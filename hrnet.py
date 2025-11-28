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

class HRModule(nn.Module):
    """ HRNet的特征融合模块 """
    def __init__(self, num_branches, blocks, num_blocks, num_channels):
        super().__init__()
        self.num_branches = num_branches
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        
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
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.InstanceNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear'))
                elif j == i:  # 恒等映射
                    fuse_layer.append(None)
                else:  # 下采样
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(num_channels[i])))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(num_channels[j]),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        # 特征融合实现...
        return fused_features

class HRNet(nn.Module):
    """ 适用于立体匹配的4阶段HRNet """
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        
        # Stage 2-4配置
        self.stage2 = HRModule(...)  # 需完整配置参数
        self.stage3 = HRModule(...)
        self.stage4 = HRModule(...)
        
        # 输出转换层
        self.transition = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),    # 1/4
            nn.Conv2d(512, out_channels*2, 1),  # 1/8
            nn.Conv2d(1024, out_channels*4, 1), # 1/16
            nn.Conv2d(2048, out_channels*8, 1)  # 1/32
        ])
        
    def _make_layer(self, block, inplanes, planes, blocks):
        # 标准残差块构建...
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer1(x)
        
        # Stage 2-4
        x_list = [x]
        x_list = self.stage2(x_list)
        x_list = self.stage3(x_list)
        x_list = self.stage4(x_list)
        
        # 输出多尺度特征
        outputs = []
        for i in range(4):
            outputs.append(self.transition[i](x_list[i]))
            
        return outputs  # [1/4, 1/8, 1/16, 1/32] 特征图