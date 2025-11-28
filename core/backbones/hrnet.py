# 新建文件 core/backbones/hrnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# CBAM注意力机制实现
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HRModule(nn.Module):
    """ HRNet的特征融合模块 """
    def __init__(self, num_branches, blocks, num_blocks, num_channels, use_cbam=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.use_cbam = use_cbam
        
        # 为每个分支添加CBAM注意力机制
        if use_cbam:
            self.cbam_modules = nn.ModuleList([
                CBAM(num_channels[i]) for i in range(num_branches)
            ])
        
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(block, num_blocks[i], num_channels[i])
            for i in range(num_branches)
        ])
    
    def _make_one_branch(self, block, num_blocks, planes):
        layers = []
        layers.append(block(planes, planes))
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:  # 上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_channels[j], self.num_channels[i], 1, 1, 0, bias=False),
                        nn.InstanceNorm2d(self.num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)))
                elif j == i:  # 恒等映射
                    fuse_layer.append(None)
                else:  # 下采样
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[i], 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(self.num_channels[i])))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[j], 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(self.num_channels[j]),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        # 确保输入列表长度与分支数匹配
        if len(x) < self.num_branches:
            # 如果输入列表长度不足，使用最后一个元素填充
            x = x + [x[-1]] * (self.num_branches - len(x))
        
        # 预处理：确保每个分支的输入通道数与期望的通道数匹配
        for i in range(self.num_branches):
            if x[i].shape[1] != self.num_channels[i]:
                # 创建一个临时的1x1卷积层来调整通道数
                channel_adapter = nn.Conv2d(
                    x[i].shape[1], 
                    self.num_channels[i], 
                    kernel_size=1, 
                    stride=1, 
                    bias=False
                ).to(x[i].device)
                # 初始化权重
                nn.init.kaiming_normal_(channel_adapter.weight, mode='fan_out', nonlinearity='relu')
                # 调整通道数
                x[i] = channel_adapter(x[i])
        
        # 处理每个分支
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
            # 应用CBAM注意力
            if self.use_cbam:
                x[i] = self.cbam_modules[i](x[i])
        
        # 如果只有一个分支，直接返回
        if self.num_branches == 1:
            return [x[0]]
        
        # 特征融合
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 or self.fuse_layers[i][0] is None else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j or self.fuse_layers[i][j] is None:
                    # 直接相加前确保尺寸匹配
                    if y.shape[2:] != x[j].shape[2:]:
                        # 调整尺寸以匹配
                        x_j_resized = F.interpolate(x[j], size=y.shape[2:], mode='bilinear', align_corners=True)
                        y = y + x_j_resized
                    else:
                        y = y + x[j]
                else:
                    # 使用融合层处理，但仍然检查尺寸
                    fused_feature = self.fuse_layers[i][j](x[j])
                    if y.shape[2:] != fused_feature.shape[2:]:
                        # 调整尺寸以匹配
                        fused_feature = F.interpolate(fused_feature, size=y.shape[2:], mode='bilinear', align_corners=True)
                    y = y + fused_feature
            x_fuse.append(y)
            
        return x_fuse

class HRNet(nn.Module):
    """ 适用于立体匹配的4阶段HRNet """
    def __init__(self, in_channels=3, out_channels=96, use_cbam=True):
        super().__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        
        # Stage 2-4配置
        # Stage 2: 2个分支，通道数为[64, 128]
        self.transition1 = self._make_transition_layer([64], [64, 128])
        self.stage2 = HRModule(
            num_branches=2,
            blocks=BasicBlock,
            num_blocks=[4, 4],
            num_channels=[64, 128],
            use_cbam=use_cbam
        )
        
        # Stage 3: 3个分支，通道数为[64, 128, 256]
        self.transition2 = self._make_transition_layer([64, 128], [64, 128, 256])
        self.stage3 = HRModule(
            num_branches=3,
            blocks=BasicBlock,
            num_blocks=[4, 4, 4],
            num_channels=[64, 128, 256],
            use_cbam=use_cbam
        )
        
        # Stage 4: 4个分支，通道数为[64, 128, 256, 512]
        self.transition3 = self._make_transition_layer([64, 128, 256], [64, 128, 256, 512])
        self.stage4 = HRModule(
            num_branches=4,
            blocks=BasicBlock,
            num_blocks=[4, 4, 4, 4],
            num_channels=[64, 128, 256, 512],
            use_cbam=use_cbam
        )
        
        # 输出转换层
        self.transition = nn.ModuleList([
            nn.Conv2d(64, out_channels, 1),      # 1/4
            nn.Conv2d(128, out_channels*2, 1),   # 1/8
            nn.Conv2d(256, out_channels*4, 1),   # 1/16
            nn.Conv2d(512, out_channels*8, 1)    # 1/32
        ])
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, inplanes, planes, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        transition_layers = []
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.InstanceNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i - len(num_channels_pre_layer) + 1):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - len(num_channels_pre_layer) else in_channels
                    conv_downsamples.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                        nn.InstanceNorm2d(out_channels),
                        nn.ReLU(True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)
    
    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer1(x)
        
        # Stage 2
        x_list = [x]
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
        x_list = self.stage2(x_list)
        
        # Stage 3 - 修复索引错误
        x_list_stage3 = []
        for i in range(len(self.transition2)):
            if i < len(x_list):  # 确保索引有效
                if self.transition2[i] is not None:
                    x_list_stage3.append(self.transition2[i](x_list[i]))
                else:
                    x_list_stage3.append(x_list[i])
        x_list = self.stage3(x_list_stage3)
        
        # Stage 4 - 同样添加索引检查
        x_list_stage4 = []
        for i in range(len(self.transition3)):
            if i < len(x_list):  # 确保索引有效
                if self.transition3[i] is not None:
                    x_list_stage4.append(self.transition3[i](x_list[i]))
                else:
                    x_list_stage4.append(x_list[i])
        x_list = self.stage4(x_list_stage4)
        
        # 输出多尺度特征 - 同样添加索引检查
        outputs = []
        for i in range(min(len(x_list), len(self.transition))):
            outputs.append(self.transition[i](x_list[i]))
            
        return outputs  # [1/4, 1/8, 1/16, 1/32] 特征图