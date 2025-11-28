# igev_stereo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, GeoEncoder # Modified import
from core.extractor import MultiBasicEncoder
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.backbones.hrnet import HRNet  # 统一导入
from torch.cuda.amp import autocast # Add this import for autocast

# Add this function before the OptimizedHourglass class definition
# 替换原有的build_corr函数
def build_corr(feat_left, feat_right, corr_radius):
    """向量化构建相关性体积
    
    Args:
        feat_left: 左图特征 [B, C, H, W]
        feat_right: 右图特征 [B, C, H, W]
        corr_radius: 相关性半径
        
    Returns:
        correlation volume [B, 8, D, H, W] where D = 2*corr_radius+1
    """
    B, C, H, W = feat_left.shape
    D = 2 * corr_radius + 1
    
    # 归一化特征
    feat_left = feat_left / (torch.norm(feat_left, dim=1, keepdim=True) + 1e-7)
    feat_right = feat_right / (torch.norm(feat_right, dim=1, keepdim=True) + 1e-7)
    
    # 预分配内存空间
    corr_volume = torch.zeros(B, 8, D, H, W, device=feat_left.device)
    
    # 创建视差偏移索引
    disp_indices = torch.arange(-corr_radius, corr_radius+1, device=feat_left.device)
    
    # 向量化计算所有视差的相关性
    for i, disp in enumerate(disp_indices):
        # 使用高效的移位和掩码操作
        if disp < 0:
            # 左移右图特征
            shifted_feat = torch.zeros_like(feat_right)
            shifted_feat[:, :, :, :W+disp] = feat_right[:, :, :, -disp:]
        elif disp > 0:
            # 右移右图特征
            shifted_feat = torch.zeros_like(feat_right)
            shifted_feat[:, :, :, disp:] = feat_right[:, :, :, :W-disp]
        else:
            # 不移动
            shifted_feat = feat_right
        
        # 批量计算相关性
        correlation = torch.sum(feat_left * shifted_feat, dim=1, keepdim=True)
        
        # 高效地复制到所有8个通道
        corr_volume[:, :, i] = correlation.repeat(1, 8, 1, 1)
    
    return corr_volume

class OptimizedHourglass(nn.Module):
    """优化版3D沙漏网络"""
    # 修改 __init__ 方法签名以接受 context_feature_channels_list
    def __init__(self, in_channels, context_feature_channels_list):
        super().__init__()
        # 保持原有核心结构不变，删除冗余注释
        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1)
        
        # 下采样路径
        self.downsample = nn.ModuleList([
            nn.Sequential(
                BasicConv(in_channels*(2**i), in_channels*(2**(i+1)), kernel_size=3, stride=2, padding=1, is_3d=True, relu=True), 
                BasicConv(in_channels*(2**(i+1)), in_channels*(2**(i+1)), kernel_size=3, stride=1, padding=1, is_3d=True, relu=True)
            ) for i in range(3)
        ])
        
        # 上采样路径
        self.upsample = nn.ModuleList([
            BasicConv(in_channels*(2**(3-i)), in_channels*(2**(2-i)), deconv=True, is_3d=True, relu=True, kernel_size=4, stride=2, padding=1) # Typical deconv params
            for i in range(3)
        ])
        
        # 特征融合模块统一初始化 - 修复通道数匹配问题
        self.aggregators = nn.ModuleList([
            nn.Sequential(
                # 第一个聚合器：处理 upsample[0] 和 skips[2] 的拼接
                # 根据调试输出，实际通道数是64
                BasicConv(64, in_channels*4, kernel_size=1, stride=1, padding=0, is_3d=True, relu=True),
                BasicConv(in_channels*4, in_channels*4, kernel_size=3, stride=1, padding=1, is_3d=True, relu=True),
            ),
            nn.Sequential(
                # 第二个聚合器：处理 upsample[1] 和 skips[1] 的拼接
                # 根据调试输出，实际通道数是32
                BasicConv(32, in_channels*2, kernel_size=1, stride=1, padding=0, is_3d=True, relu=True),
                BasicConv(in_channels*2, in_channels*2, kernel_size=3, stride=1, padding=1, is_3d=True, relu=True),
            ),
            nn.Sequential(
                # 第三个聚合器：处理 upsample[2] 和 x 的拼接
                BasicConv(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, is_3d=True, relu=True),
                BasicConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, is_3d=True, relu=True),
            )
        ])
        
        # 动态门控融合统一初始化
        # 确保 context_feature_channels_list 的长度与 dynamic_fusions 模块数量一致 (通常是4)
        self.dynamic_fusions = nn.ModuleList([
            DynamicHiLoFusion(
                in_channels=in_channels*(2**i),
                context_feature_channels=context_feature_channels_list[i]
            ) for i in range(len(context_feature_channels_list))
        ])

    def forward(self, x, features):
        # 简化前向传播逻辑
        # Initial convolution, e.g. 1/4 scale features
        x = self.conv0(x)
        # Potentially apply the first dynamic fusion here if it's for the initial scale
        # x = self.dynamic_fusions[0](x, features[0]) # Assuming dynamic_fusions[0] is for this scale
    
        skips = []
        current_x = x 
        for i in range(3): # Downsampling path
            current_x = self.downsample[i](current_x)
            skips.append(current_x)
        
        # current_x is now at the smallest resolution, e.g., in_channels*8
        
        for i in range(3): # Upsampling path, i = 0, 1, 2
            upsampled_x = self.upsample[i](current_x)
            
            # 确定正确的跳跃连接
            skip_to_concat = skips[1-i] if i < 2 else x
            
            # 确保空间尺寸匹配 - 使用trilinear模式处理5D张量
            if skip_to_concat.shape[2:] != upsampled_x.shape[2:]:
                skip_to_concat = F.interpolate(
                    skip_to_concat, 
                    size=upsampled_x.shape[2:], 
                    mode='trilinear',
                    align_corners=True
                )
            
    
            
            concatenated_x = torch.cat([upsampled_x, skip_to_concat], dim=1)
            aggregated_x = self.aggregators[i](concatenated_x)
            
            fusion_idx = 2 - i
            current_x = self.dynamic_fusions[fusion_idx](aggregated_x, features[fusion_idx])
        
        return current_x

class stereo(nn.Module):
    """统一的主干网络"""
    def __init__(self, args):
        super().__init__()

        # Add default for cost_volume_planes if not present
        if not hasattr(args, 'cost_volume_planes'):
            # Determine a sensible default based on corr_radius.
            # This assumes corr_radius is always present in args.
            default_cv_planes = (2 * args.corr_radius + 1)
            print(f"Warning: 'cost_volume_planes' not found in args. Defaulting to {default_cv_planes}.")
            args.cost_volume_planes = default_cv_planes

        # Add default for use_hrnet if not present
        if not hasattr(args, 'use_hrnet'):
            print(f"Warning: 'use_hrnet' not found in args. Defaulting to True (using HRNet).")
            args.use_hrnet = True

        # Add default for optimized if not present
        if not hasattr(args, 'optimized'):
            print(f"Warning: 'optimized' not found in args. Defaulting to True (using OptimizedHourglass).")
            args.optimized = True # <--- 修改这里为 True

        if len(args.hidden_dims) < 3:
            raise ValueError(f"hidden_dims需要至少3个值，当前得到{len(args.hidden_dims)}")
        
        self.args = args
        
        # 特征提取器配置 - 分开定义 backbone 和 stem
        self.feature_extractor = nn.ModuleDict({
            'backbone': HRNet(3, 96) if self.args.use_hrnet else FeatureWithHiLo(3, 96),
            'stem': nn.Sequential(
                BasicConv(3, 32, kernel_size=3, stride=2),
                nn.Sequential(
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU()
                )
            )
        })

        stem_channels = 32
        # Store backbone_feature_channels as an instance variable
        self.backbone_feature_channels = [96, 192, 384, 768] if args.use_hrnet else [48, 96, 192, 160]
        
        # 将上下文通道列表提升为实例变量
        self.context_channels_for_hourglass = [stem_channels] + self.backbone_feature_channels[:3]
        
        # 代价体聚合器初始化
        self.cost_aggregators = nn.ModuleList([
            OptimizedHourglass(8, self.context_channels_for_hourglass) if args.optimized else hourglass(8)
            for _ in range(3)
        ])
        
        # 公共组件初始化
        self._init_common_components(args)

    def _init_common_components(self, args):
        """初始化共享组件"""
        # 视差估计相关组件
        self.classifier = nn.Conv3d(8, 1, 3, padding=1)
        self.disp_fusion = nn.Sequential(
            BasicConv(3, 128, kernel_size=3, stride=1, padding=1),  # 修改为3通道输入
            nn.Conv2d(128, 1, 1),  # 修改为1通道输出，因为视差图是单通道的
            nn.Sigmoid()
        )
        
        # GRU相关组件
        disp_encoder_output_channels = 128 
        self.update_block = BasicMultiUpdateBlock(
            args,
            encoder_output_dim=disp_encoder_output_channels
        )
        
        # Initialize modules required by Combined_Geo_Encoding_Volume
        # D_channels is typically (2 * corr_radius + 1)
        D_channels = (2 * self.args.corr_radius + 1)
        # feat_lvl1_channels are the channels of the 1/4 scale features (e.g., feats_left[1])
        feat_lvl1_channels = self.backbone_feature_channels[0]

        # geo_volume2: assumed to be a GeoEncoder instance
        # It processes a geometric cost volume (D_channels) to help generate selective_weights
        geo_module_for_weights = GeoEncoder(geo_planes=D_channels)

        # init_fmap1 & init_fmap2: assumed to be Conv2d layers
        # They process 1/4 scale features (feat_lvl1_channels) to produce D_channels maps
        # These are then used to compute init_corr_val within Combined_Geo_Encoding_Volume
        fmap_module1 = nn.Conv2d(feat_lvl1_channels, D_channels, kernel_size=1, padding=0)
        fmap_module2 = nn.Conv2d(feat_lvl1_channels, D_channels, kernel_size=1, padding=0)
        
        # Initialize the geometry processor with all required arguments
        self.geometry_processor = Combined_Geo_Encoding_Volume(
            self.args, 
            self.args.cost_volume_planes, # This is cv_chan
            geo_module_for_weights,      # This is geo_volume2
            fmap_module1,                # This is init_fmap1
            fmap_module2                 # This is init_fmap2
        )

        # 修复：使用backbone的通道数而不是hidden_dims
        backbone_channels = [96, 192, 384] if args.use_hrnet else [48, 96, 192]
        # 在_init_common_components方法中
        # 修改context_convs的定义，确保输出通道数正确
        self.context_convs = nn.ModuleList([
        nn.Conv2d(backbone_channels[i], args.hidden_dims[i]*3, 3, padding=1) 
        for i in range(len(args.hidden_dims))
        ])
    
    def forward(self, left, right, max_iters=12, test=False, iters=None, test_mode=None):
        """覆盖父类的forward方法，确保只返回最终视差预测"""
        # 处理参数兼容性
        if iters is not None:
            max_iters = iters
        if test_mode is not None:
            test = test_mode
            
        # 特征提取
        feats_left = self._extract_features(left)
        feats_right = self._extract_features(right)
        
        # 代价体构建与处理
        cost_volumes = self._build_cost_volumes(feats_left, feats_right)
        disp_preds = self._process_cost_volumes(cost_volumes)
        
        # GRU迭代优化 - 这里会返回两个值，但我们只需要最终的视差预测
        _, iter_preds = self._iterative_refinement(disp_preds, feats_left, feats_right, max_iters, test)
        
        # 只返回最终的视差预测（最后一次迭代的结果）
        return iter_preds[-1]
    
    def _extract_features(self, x):
        """特征提取统一方法"""
        stem = self.feature_extractor['stem'](x)
        features = self.feature_extractor['backbone'](x)

        # print(f"Stem shape: {stem.shape}")
        # for i, f in enumerate(features):
        #     print(f"Feature {i} shape: {f.shape}")
        
        # Make sure we return exactly 4 feature levels as expected by _build_cost_volumes
        if len(features) > 3:
            # If we have more than 3 features from backbone, only use the first 3
            return [stem] + features[:3]
        elif len(features) < 3:
            # If we have fewer than 3 features, pad with duplicates of the last one
            padding = [features[-1]] * (3 - len(features))
            return [stem] + features + padding
        else:
            # If we have exactly 3 features, return as is
            return [stem] + features

    def freeze_bn(self):
        """冻结BatchNorm层"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

class GeoDynStereo(stereo):
    def __init__(self, args):
        # 在调用父类构造器前设置参数
        args.use_hrnet = True
        args.optimized = True
        super().__init__(args)  # 现在会正确继承context_channels_for_hourglass

        # 添加额外优化组件
        self.enhanced_fusion = nn.Sequential(
            BasicConv(160, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, max_iters=12, iters=None, test=False, test_mode=False):
        """前向传播，处理训练和测试模式

        参数:
            x1: 左图像
            x2: 右图像
            max_iters: 最大迭代次数
            iters: 兼容性参数，如果提供则覆盖max_iters
            test: 是否为测试模式
            test_mode: 兼容性参数，如果为True则设置test为True
        """
        # 参数兼容性处理
        if iters is not None:
            max_iters = iters

        if test_mode:
            test = True

        # 特征提取
        feats_left = self._extract_features(x1)
        feats_right = self._extract_features(x2)

        # 构建代价体
        cost_volumes = self._build_cost_volumes(feats_left, feats_right)

        # 处理代价体并生成初始视差预测
        disp_preds = self._process_cost_volumes(cost_volumes)

        # GRU迭代优化
        agg_preds, iter_preds = self._iterative_refinement(disp_preds, feats_left, feats_right, max_iters, test)

        # 根据模式返回不同的结果
        if test:
            # 测试模式只返回最终预测
            return iter_preds[-1]
        else:
            # 训练模式返回所有预测
            return agg_preds, iter_preds

    def _build_cost_volumes(self, feats_left, feats_right):
        """构建代价体"""
        # 获取特征
        stem_l, feat_l1, feat_l2, feat_l3 = feats_left
        stem_r, feat_r1, feat_r2, feat_r3 = feats_right

        # 构建多尺度代价体
        cost_volumes = []

        # 使用不同尺度的特征构建代价体
        # 通常从粗到细，使用不同分辨率的特征
        with autocast(enabled=self.args.mixed_precision):
            # 1/4 分辨率代价体
            cost_vol1 = build_corr(feat_l1, feat_r1, self.args.corr_radius)
            cost_vol1 = self.cost_aggregators[0](cost_vol1, [stem_l, feat_l1, feat_l2])

            # 1/8 分辨率代价体
            cost_vol2 = build_corr(feat_l2, feat_r2, self.args.corr_radius)
            cost_vol2 = self.cost_aggregators[1](cost_vol2, [stem_l, feat_l1, feat_l2])

            # 1/16 分辨率代价体
            cost_vol3 = build_corr(feat_l3, feat_r3, self.args.corr_radius)
            cost_vol3 = self.cost_aggregators[2](cost_vol3, [stem_l, feat_l1, feat_l2])

            cost_volumes = [cost_vol1, cost_vol2, cost_vol3]

        return cost_volumes

    def _process_cost_volumes(self, cost_volumes):
        """处理代价体并生成初始视差预测"""
        # 确保volumes是float32类型
        volumes = [vol.float() for vol in cost_volumes]

        # 对每个代价体应用分类器获取概率体
        prob_volumes = [F.softmax(self.classifier(vol), dim=2) for vol in volumes]

        # 计算每个代价体的视差预测
        disp_preds = []
        for prob_vol in prob_volumes:
            # 将概率体压缩为2D
            cost_vol = prob_vol.squeeze(1)

            # 创建视差候选值
            B, D, H, W = cost_vol.shape
            disp_candidates = torch.arange(0, D, device=cost_vol.device, dtype=cost_vol.dtype).view(1, D, 1, 1)

            # 计算期望视差
            disp_pred = torch.sum(cost_vol * disp_candidates, dim=1, keepdim=True)
            disp_preds.append(disp_pred)

        # 融合多尺度视差预测
        upsampled_disps = [F.interpolate(disp, size=disp_preds[0].shape[2:], mode='bilinear', align_corners=True)
                           for disp in disp_preds[1:]]

        all_disps = torch.cat([disp_preds[0]] + upsampled_disps, dim=1)
        fused_disp = self.disp_fusion(all_disps)

        return fused_disp

    def _iterative_refinement(self, disp_preds, feats_left, feats_right, max_iters=12, test=False):
        """使用GRU自适应迭代优化视差预测    几何编码"""
        # 初始化隐藏状态
        hidden_states = []
        for i, dim in enumerate(self.args.hidden_dims):
            hidden_states.append(torch.zeros(disp_preds.shape[0], dim,
                                           disp_preds.shape[2]//2**i,
                                           disp_preds.shape[3]//2**i,
                                           device=disp_preds.device))

        # 准备上下文特征 (uses feats_left)
        context_features = []
        for i, dim in enumerate(self.args.hidden_dims):
            feat = feats_left[min(i+1, len(feats_left)-1)]
            context = self.context_convs[i](feat)
            chunks = torch.chunk(context, 3, dim=1)
            context_features.append(chunks)

        # 准备几何输入
        geo_input_feat_l1 = feats_left[1]  # 1/4 scale left features
        geo_input_feat_r1 = feats_right[1] # 1/4 scale right features
        cost_volume_geom_list = self.geometry_processor.build_cost_volume_geom(geo_input_feat_l1, geo_input_feat_r1)
        _, init_corr_val, selective_weights_val = self.geometry_processor(
            geo_input_feat_l1,
            geo_input_feat_r1,
            cost_volume_geom_list
        )
        init_corr_for_update_block = torch.cat([init_corr_val, init_corr_val], dim=1)

        # 迭代优化
        iter_preds = []
        current_disp = disp_preds

        # 自适应迭代参数
        min_iters = 3  # 最小迭代次数
        convergence_threshold = 0.05  # 收敛阈值
        prev_disp = None

        for i in range(max_iters):
            # 更新隐藏状态和视差
            hidden_states, _, delta_disp = self.update_block(
                hidden_states, context_features,
                geo_feat0=cost_volume_geom_list[0],
                geo_feat1=cost_volume_geom_list[1],
                geo_feat2=cost_volume_geom_list[2],
                init_corr=init_corr_for_update_block,
                selective_weights=selective_weights_val,
                disp=current_disp, update=(i < max_iters-1 or not test)
            )

            # 确保delta_disp与current_disp尺寸匹配
            if delta_disp.shape[2:] != current_disp.shape[2:]:
                delta_disp = F.interpolate(
                    delta_disp,
                    size=current_disp.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 更新视差预测
            current_disp = current_disp + delta_disp
            iter_preds.append(current_disp)

            # 自适应迭代终止条件检查
            if i >= min_iters and not test:
                if prev_disp is not None:
                    # 计算相对变化
                    rel_change = torch.mean(torch.abs(current_disp - prev_disp)) / (torch.mean(torch.abs(current_disp)) + 1e-6)

                    # 如果变化小于阈值，提前终止
                    if rel_change < convergence_threshold:
                        break

                prev_disp = current_disp.clone()

        return disp_preds, iter_preds

    def freeze_bn(self):
        """冻结BatchNorm层"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
