import torch
import torch.nn as nn
import torch.nn.functional as F
from core.submodule import BasicConv

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，减少参数量和计算量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class SpatialChannelAttention(nn.Module):
    """空间-通道联合注意力机制"""
    def __init__(self, channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x

class FPNBlock(nn.Module):
    """特征金字塔网络块"""
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.output_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.attention = SpatialChannelAttention(out_channels)
        
    def forward(self, x, higher_res_feature=None):
        x = self.lateral_conv(x)
        
        if higher_res_feature is not None:
            # 上采样并融合高分辨率特征
            x = F.interpolate(x, size=higher_res_feature.shape[-2:], mode='bilinear', align_corners=False)
            x = x + higher_res_feature
        
        x = self.output_conv(x)
        x = self.attention(x)
        return x

class UnifiedBackbone(nn.Module):
    """统一的主干网络，集成特征提取和代价体处理"""
    def __init__(self, input_channels=3, feature_channels=[64, 128, 256], fpn_channels=128):
        super(UnifiedBackbone, self).__init__()
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 多尺度特征提取器
        self.feature_extractors = nn.ModuleList()
        in_ch = 64
        for out_ch in feature_channels:
            self.feature_extractors.append(
                nn.Sequential(
                    DepthwiseSeparableConv(in_ch, out_ch, stride=2),
                    DepthwiseSeparableConv(out_ch, out_ch),
                    SpatialChannelAttention(out_ch)
                )
            )
            in_ch = out_ch
        
        # 特征金字塔网络
        self.fpn_blocks = nn.ModuleList()
        for i in range(len(feature_channels)):
            self.fpn_blocks.append(FPNBlock(feature_channels[-(i+1)], fpn_channels))
        
        # 代价体构建和处理
        self.cost_processor = CostVolumeProcessor(fpn_channels)
        
    def forward(self, left_img, right_img):
        # 特征提取
        left_features = self.extract_features(left_img)
        right_features = self.extract_features(right_img)
        
        # FPN处理
        left_fpn = self.build_fpn(left_features)
        right_fpn = self.build_fpn(right_features)
        
        # 代价体构建和处理
        cost_volume = self.cost_processor(left_fpn, right_fpn)
        
        return cost_volume, left_fpn
    
    def extract_features(self, x):
        features = []
        x = self.stem(x)
        
        for extractor in self.feature_extractors:
            x = extractor(x)
            features.append(x)
        
        return features
    
    def build_fpn(self, features):
        fpn_features = []
        x = None
        
        for i, fpn_block in enumerate(self.fpn_blocks):
            feat = features[-(i+1)]  # 从最高层开始
            x = fpn_block(feat, x)
            fpn_features.insert(0, x)  # 插入到前面，保持从低到高的顺序
        
        return fpn_features

class CostVolumeProcessor(nn.Module):
    """高效的代价体构建和处理模块"""
    def __init__(self, feature_channels, max_disp=192):
        super(CostVolumeProcessor, self).__init__()
        self.max_disp = max_disp
        self.feature_channels = feature_channels
        
        # 代价体聚合网络
        self.cost_aggregation = nn.Sequential(
            DepthwiseSeparableConv(max_disp//4, 128),
            DepthwiseSeparableConv(128, 64),
            SpatialChannelAttention(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度融合
        self.multi_scale_fusion = nn.ModuleList([
            nn.Conv2d(feature_channels, 32, 1) for _ in range(3)
        ])
        
        # 最终预测头
        self.disp_head = nn.Sequential(
            nn.Conv2d(32 + 32 * 3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    
    def build_cost_volume(self, left_feat, right_feat):
        """构建4D代价体"""
        B, C, H, W = left_feat.shape
        cost_volume = torch.zeros(B, self.max_disp//4, H, W, device=left_feat.device, dtype=left_feat.dtype)
        
        for i in range(self.max_disp//4):
            if i == 0:
                cost_volume[:, i, :, :] = torch.sum(left_feat * right_feat, dim=1)
            else:
                cost_volume[:, i, :, :W-i] = torch.sum(
                    left_feat[:, :, :, i:] * right_feat[:, :, :, :W-i], dim=1
                )
        
        return cost_volume
    
    def forward(self, left_fpn, right_fpn):
        # 使用最高分辨率特征构建代价体
        main_left = left_fpn[0]  # 最高分辨率
        main_right = right_fpn[0]
        
        cost_volume = self.build_cost_volume(main_left, main_right)
        cost_features = self.cost_aggregation(cost_volume)
        
        # 多尺度特征融合
        multi_scale_features = []
        target_size = cost_features.shape[-2:]
        
        for i, fusion_conv in enumerate(self.multi_scale_fusion):
            feat = left_fpn[i]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = fusion_conv(feat)
            multi_scale_features.append(feat)
        
        # 特征融合
        fused_features = torch.cat([cost_features] + multi_scale_features, dim=1)
        
        # 视差预测
        disparity = self.disp_head(fused_features)
        
        return disparity

class GeoDynStereo(nn.Module):
    """优化后的IGEV立体匹配网络"""
    def __init__(self, args):
        super(GeoDynStereo, self).__init__()
        self.args = args
        
        # 统一主干网络
        self.backbone = UnifiedBackbone(
            input_channels=3,
            feature_channels=[64, 128, 256],
            fpn_channels=128
        )
        
    def forward(self, image1, image2, iters=12, test_mode=False):
        # 记录原始尺寸
        original_height, original_width = image1.shape[2], image1.shape[3]
        
        # 统一的特征提取和代价体处理
        disparity, features = self.backbone(image1, image2)
        
        # 上采样到原始分辨率
        disparity_upsampled = F.interpolate(
            disparity * 4.0,  # 补偿下采样的尺度
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )
        
        if test_mode:
            return disparity_upsampled
        else:
            return [disparity_upsampled] * iters
