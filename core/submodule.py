import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
        



def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(prob, maxdisp, interval):
    assert len(prob.shape) == 4
    disp_values = torch.arange(0, maxdisp, interval, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp//interval, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)

def disparity_variance(prob, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(prob.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(prob * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape       
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    disp = (disp_unfold*up_weights).sum(dim=1,keepdim=True)      
    return disp


class HiLo(nn.Module):
    """
    HiLo注意力机制
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)  # 每个注意力头的通道数
        self.dim = dim

        # 低频注意力头数量
        self.l_heads = int(num_heads * alpha)
        # 低频注意力通道数
        self.l_dim = self.l_heads * head_dim

        # 高频注意力头数量
        self.h_heads = num_heads - self.l_heads
        # 高频注意力通道数
        self.h_dim = self.h_heads * head_dim

        # 窗口大小
        self.ws = window_size

        # 如果窗口大小为1，则只使用低频注意力
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # 低频注意力
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # 高频注意力
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape
        
        # 计算需要的填充量，确保高度和宽度能被窗口大小整除
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        
        if pad_h > 0 or pad_w > 0:
            # 对输入进行填充
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), "constant", 0)
            x = x.permute(0, 2, 3, 1)
            
        # 更新填充后的高度和宽度
        H_padded, W_padded = H + pad_h, W + pad_w
        h_group, w_group = H_padded // self.ws, W_padded // self.ws
        total_groups = h_group * w_group

        # 重塑为窗口形式
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # QKV投影
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 注意力加权
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)

        # 重塑回原始形状
        x = attn.transpose(2, 3).reshape(B, H_padded, W_padded, self.h_dim)
        
        # 如果进行了填充，需要去除填充部分
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
            
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape
        
        # 计算需要的填充量，确保高度和宽度能被窗口大小整除
        if self.ws > 1:
            pad_h = (self.ws - H % self.ws) % self.ws
            pad_w = (self.ws - W % self.ws) % self.ws
            
            if pad_h > 0 or pad_w > 0:
                # 对输入进行填充
                x_padded = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), "constant", 0)
                x_padded = x_padded.permute(0, 2, 3, 1)
                H_padded, W_padded = H + pad_h, W + pad_w
            else:
                x_padded = x
                H_padded, W_padded = H, W
        else:
            x_padded = x
            H_padded, W_padded = H, W
            
        # 使用分块处理来减少内存使用
        # 将查询分成多个块处理
        chunk_size = 1024  # 可以根据GPU内存调整这个值
        num_chunks = (H * W + chunk_size - 1) // chunk_size
        
        # 初始化输出张量
        output = torch.zeros(B, H, W, self.l_dim, device=x.device)
        
        # Query投影
        q_full = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads)
        
        # 如果窗口尺寸大于1，执行池化操作
        if self.ws > 1:
            # 池化操作
            x_ = x_padded.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
            k, v = kv[:, :, 0], kv[:, :, 1]  # [B, H*W/ws^2, l_heads, head_dim]
        else:
            # 无池化
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
            k, v = kv[:, :, 0], kv[:, :, 1]  # [B, H*W, l_heads, head_dim]
        
        # 分块处理注意力计算
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, H * W)
            
            # 获取当前块的查询
            q_chunk = q_full[:, start_idx:end_idx].permute(0, 2, 1, 3)  # [B, l_heads, chunk_size, head_dim]
            
            # 注意力计算
            attn = (q_chunk @ k.permute(0, 2, 3, 1)) * self.scale  # [B, l_heads, chunk_size, H*W/ws^2]
            attn = attn.softmax(dim=-1)
            
            # 注意力加权
            chunk_output = (attn @ v.permute(0, 2, 1, 3))  # [B, l_heads, chunk_size, head_dim]
            chunk_output = chunk_output.permute(0, 2, 1, 3).reshape(B, end_idx-start_idx, self.l_dim)
            
            # 将结果放回输出张量
            output.reshape(B, H*W, self.l_dim)[:, start_idx:end_idx] = chunk_output
        
        # 应用投影
        output = self.l_proj(output)
        return output
    
    def forward(self, x):
        # 输入: [B, C, H, W]
        B, C, H, W = x.shape
        # 转换为[B, H, W, C]格式
        x = x.permute(0, 2, 3, 1)
        
        # 根据头分配情况选择执行路径
        if self.h_heads == 0:
            # 仅执行低频注意力
            x = self.lofi(x)
        elif self.l_heads == 0:
            # 仅执行高频注意力
            x = self.hifi(x)
        else:
            # 同时执行高频和低频注意力
            hifi_out = self.hifi(x)
            lofi_out = self.lofi(x)
            x = torch.cat((hifi_out, lofi_out), dim=-1)
        
        # 转换回[B, C, H, W]格式
        x = x.permute(0, 3, 1, 2)
        return x

class LiteHiLo(nn.Module):
    """轻量化HiLo注意力模块，使用分组卷积减少计算量"""
    def __init__(self, dim, kernel_size=3, groups=4, reduction_ratio=2):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.reduction_dim = max(dim // reduction_ratio, 16)
        
        # 高频路径 - 使用分组卷积减少参数
        self.hi_path = nn.Sequential(
            nn.Conv2d(dim, self.reduction_dim, kernel_size=1),
            nn.Conv2d(self.reduction_dim, self.reduction_dim, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=groups),
            nn.Conv2d(self.reduction_dim, dim, kernel_size=1)
        )
        
        # 低频路径 - 使用平均池化和分组卷积
        self.lo_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, self.reduction_dim, kernel_size=1),
            nn.Conv2d(self.reduction_dim, self.reduction_dim, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=groups),
            nn.Conv2d(self.reduction_dim, dim, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 特征融合
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        
    def forward(self, x):
        hi_feat = self.hi_path(x)
        lo_feat = self.lo_path(x)
        fused_feat = hi_feat + lo_feat
        return self.proj(fused_feat) + x

class Lite(nn.Module):
    """使用轻量化HiLo注意力的特征提取网络"""
    def __init__(self, in_channels=3, out_channels=96, norm_fn='instance'):
        super().__init__()
        
        if norm_fn == 'instance':
            self.norm = nn.InstanceNorm2d
        elif norm_fn == 'batch':
            self.norm = nn.BatchNorm2d
        
        # 初始特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=2, padding=1),
            self.norm(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1),
            self.norm(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 1/4尺度特征提取
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=2, padding=1),
            self.norm(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1),
            self.norm(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 轻量化HiLo注意力模块
        self.hilo1 = LiteHiLo(out_channels//2)
        
        # 1/8尺度特征提取
        self.layer1 = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1),
            self.norm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer1_hilo = LiteHiLo(out_channels)
        
        # 1/16尺度特征提取
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1),
            self.norm(out_channels*2),
            nn.ReLU(inplace=True)
        )
        self.layer2_hilo = LiteHiLo(out_channels*2)
        
        # 1/32尺度特征提取
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels*5/3, kernel_size=3, stride=2, padding=1),
            self.norm(int(out_channels*5/3)),
            nn.ReLU(inplace=True)
        )
        self.layer3_hilo = LiteHiLo(int(out_channels*5/3))
        
    def forward(self, x):
        # 1/2尺度
        x = self.conv1(x)
        
        # 1/4尺度
        x = self.conv2(x)
        x = self.hilo1(x)
        out_feature_1_4 = x
        
        # 1/8尺度
        x = self.layer1(x)
        x = self.layer1_hilo(x)
        out_feature_1_8 = x
        
        # 1/16尺度
        x = self.layer2(x)
        x = self.layer2_hilo(x)
        out_feature_1_16 = x
        
        # 1/32尺度
        x = self.layer3(x)
        x = self.layer3_hilo(x)
        out_feature_1_32 = x
        
        return [out_feature_1_4, out_feature_1_8, out_feature_1_16, out_feature_1_32]


class DynamicHiLoFusion(nn.Module):
    def __init__(self, in_channels, context_feature_channels):
        super().__init__()
        self.in_channels = in_channels
        self.context_feature_channels = context_feature_channels
        
        # 2D上下文特征处理
        self.context_conv = nn.Sequential(
            nn.Conv2d(context_feature_channels, in_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, context):
        # x: [B, C, D, H, W]
        # context: [B, C_ctx, H, W]
        
        B, C, D, H, W = x.shape
        
        # 处理上下文特征
        context_feat = self.context_conv(context)  # [B, C*2, H, W]
        
        # 将2D上下文特征扩展为3D
        # 确保扩展后的形状与x匹配
        context_3d = context_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
        
        # 确保context_3d的空间尺寸与x匹配
        if context_3d.shape[3:] != x.shape[3:]:
            # 修复：正确计算reshape的维度
            context_3d_reshaped = context_3d.reshape(B * context_3d.shape[1] * D, 1, context_3d.shape[3], context_3d.shape[4])
            context_3d_resized = F.interpolate(
                context_3d_reshaped,
                size=x.shape[3:],
                mode='bilinear',
                align_corners=True
            )
            # 恢复原始维度结构
            context_3d = context_3d_resized.reshape(B, context_3d.shape[1], D, x.shape[3], x.shape[4])
        
        # 计算门控信号
        gate_input = torch.cat([x, x], dim=1)  # 确保通道数匹配context_3d
        
        # 确保gate_input和context_3d的通道数匹配
        if gate_input.shape[1] != context_3d.shape[1]:
            # 调整context_3d的通道数以匹配gate_input
            context_3d = context_3d[:, :gate_input.shape[1]]
        
        # 以下代码应该在if语句外部，无论条件是否满足都需要执行
        gate_input = gate_input + context_3d
        gate_value = self.gate(gate_input)
        
        # 应用门控
        gated_x = x * gate_value[:, :C]
        
        # 输出处理
        out = self.output_conv(gated_x)
        
        return out