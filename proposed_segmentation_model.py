import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter
from functools import partial

from torch import Tensor

nonlinearity = partial(F.relu, inplace=True)


def conv3otherRelu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """3x3 convolution with padding and relu"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)
    )


def l2_norm(x):
    """L2 normalization"""
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

# ============================================================================
# Normalization Modules
# ============================================================================

class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization for handling domain shifts"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.contiguous().view(b, c, -1)
        mean = x_reshaped.mean(2, keepdim=True).unsqueeze(3)
        var = x_reshaped.var(2, keepdim=True).unsqueeze(3)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = x_normalized * self.weight + self.bias
        return out

class SwitchableNorm2d(nn.Module):
    """Switchable Normalization: Learns to select between IN, LN, and BN"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SwitchableNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        bn_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        bn_var = x.var(dim=(0, 2, 3), keepdim=True)
        in_mean = x.mean(dim=(2, 3), keepdim=True)
        in_var = x.var(dim=(2, 3), keepdim=True)
        ln_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        ln_var = x.var(dim=(1, 2, 3), keepdim=True)
        mean_weight = F.softmax(self.mean_weight, dim=0)
        var_weight = F.softmax(self.var_weight, dim=0)
        mean = mean_weight[0] * bn_mean + mean_weight[1] * in_mean + mean_weight[2] * ln_mean
        var = var_weight[0] * bn_var + var_weight[1] * in_var + var_weight[2] * ln_var
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = x_normalized * self.weight + self.bias
        return out
# ============================================================================
# Augmentation Modules
# ============================================================================

class MixStyleBlock(nn.Module):
    """MixStyle: Mix instance-level feature statistics"""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super(MixStyleBlock, self).__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        b, c, h, w = x.size()
        if b < 2:
            return x
        x_view = x.view(b, c, -1)
        mu = x_view.mean(dim=2, keepdim=True).unsqueeze(3)
        var = x_view.var(dim=2, keepdim=True).unsqueeze(3)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        perm = torch.randperm(b)
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]
        x_mixed = x_normed * sig_mix + mu_mix
        return x_mixed

class DropBlock2d(nn.Module):
    """DropBlock: Structured dropout"""
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        block_mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        block_mask = 1 - block_mask
        normalize_factor = block_mask.numel() / block_mask.sum()
        return x * block_mask * normalize_factor
# ============================================================================
# Original Modules (kept as is)
# ============================================================================

class BasicConv2d(nn.Module):
    """Basic convolution block with BN and ReLU"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWCon(nn.Module):
    """Depthwise Separable Convolution - Lightweight"""
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, dilation=1, use_ibn=False):
        super(DWCon, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation,
                                   groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)

        if use_ibn:
            self.norm = AdaptiveInstanceNorm2d(out_planes)
        else:
            self.groupN = nn.GroupNorm(4, out_planes)
        self.use_ibn = use_ibn
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_ibn:
            x = self.norm(x)
        else:
            x = self.groupN(x)
        x = self.relu(x)
        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Channel shuffle operation"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)

class MFA_E(nn.Module):
    """Multiscale Feature Aggregation - Enhanced"""

    def __init__(self, use_mixstyle=True):
        super(MFA_E, self).__init__()

        self.dwconv1_small = DWCon(128, 128, kernel_size=3, padding=1, use_ibn=True)
        self.dwconv1_large = DWCon(128, 128, kernel_size=5, padding=2, use_ibn=True)
        self.dwconv2 = DWCon(256, 320, use_ibn=True)
        self.groupN = nn.GroupNorm(4, 128)
        self.cv1 = nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.mixstyle = MixStyleBlock(p=0.5, alpha=0.1) if use_mixstyle else None
        self.dropblock = DropBlock2d(drop_prob=0.1, block_size=7)

    def forward(self, x5):
        x5 = self.cv1(x5)

        if self.mixstyle is not None:
            x5 = self.mixstyle(x5)
        x5 = channel_shuffle(x5, 4)
        b, c, H, W = x5.size()
        channels_per_group = c // 2
        x51, x52 = torch.split(x5, channels_per_group, dim=1)
        x51_small = self.groupN(x51)
        x51_small = self.dwconv1_small(x51_small)
        x51_small = self.sigmoid(x51_small)
        x51_small = x51 * x51_small
        x52_large = self.groupN(x52)
        x52_large = self.dwconv1_large(x52_large)
        x52_large = self.sigmoid(x52_large)
        x52_large = x52 * x52_large
        x5_ = torch.cat([x51_small, x52_large], dim=1)
        out_MFA = self.dwconv2(x5_)
        out_MFA = self.dropblock(out_MFA)
        return out_MFA

class ESA_block(nn.Module):
    """Enhanced Spatial Attention Block"""

    def __init__(self, in_ch, split_factor=16, dropout_prob=0.2, use_mixstyle=True):
        super(ESA_block, self).__init__()

        self.conv2d_2 = DWCon(in_ch // 2, in_ch // 2, kernel_size=3, padding=2, dilation=2, use_ibn=True)
        self.conv2d_4 = DWCon(in_ch // 2, in_ch // 2, kernel_size=3, padding=4, dilation=4, use_ibn=True)
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.groupN = nn.GroupNorm(2, in_ch // 2)
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(
            BasicConv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.mixstyle = MixStyleBlock(p=0.3, alpha=0.1) if use_mixstyle else None

    def forward(self, x):
        if self.mixstyle is not None:
            x = self.mixstyle(x)

        x = channel_shuffle(x, 4)

        b, c, H, W = x.size()
        channels_per_group = c // 2
        x41, x42 = torch.split(x, channels_per_group, dim=1)

        x41 = self.groupN(x41)
        x41 = self.conv2d_2(x41)
        x42 = self.groupN(x42)
        x42 = self.conv2d_4(x42)
        s1 = self.SA1(x41)
        s2 = self.SA2(x42)
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1]
        out = self.sa_fusion(s_all) * x + x
        return out


class Att_block_1(nn.Module):
    """Attention Block with Edge Enhancement"""

    def __init__(self, in_channel, dropout_prob=0.5):
        super(Att_block_1, self).__init__()
        self.conv1 = conv3otherRelu(in_channel, in_channel)
        self.avg_pool = nn.AvgPool2d((5, 5), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=0)
        self.norm = SwitchableNorm2d(in_channel)

        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        edge = x - self.avg_pool(x)
        weight = self.sigmoid(self.norm(self.conv_1(edge)))
        out = weight * x + x
        out = self.PReLU(out)
        out = self.dropout(out)
        return out


# ============================================================================
# NEW: Lightweight Decoder Blocks
# ============================================================================

class LightweightDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_dropblock=True):
        super(LightweightDecoderBlock, self).__init__()

        # Depthwise separable conv with Instance Norm
        self.dwconv = DWCon(in_channels, out_channels, kernel_size=3, padding=1, use_ibn=True)
        # DropBlock for structured regularization
        self.dropblock = DropBlock2d(drop_prob=0.1, block_size=5) if use_dropblock else None
        # Upsample with bilinear + 1x1 conv (lighter than transposed conv)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1)
        # Adaptive Instance Norm for OOD robustness
        self.norm = AdaptiveInstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dwconv(x)

        # Apply DropBlock during training
        if self.dropblock is not None:
            x = self.dropblock(x)
        x = self.upsample(x)
        x = self.conv1x1(x)
        x = self.norm(x)  # Adaptive Instance Norm
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels, use_mixstyle=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = SwitchableNorm2d(channels)  # Switchable Norm instead of BN
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = SwitchableNorm2d(channels)  # Switchable Norm
        # MixStyle for domain generalization
        self.mixstyle = MixStyleBlock(p=0.3, alpha=0.1) if use_mixstyle else None

    def forward(self, x):
        residual = x
        # Apply MixStyle before processing
        if self.mixstyle is not None:
            x = self.mixstyle(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.relu(out)
        return out


# ============================================================================
# Main Model with Lightweight Decoder
# ============================================================================

class CFFANet_OOD(nn.Module):

    def __init__(self, num_channels=3, num_classes=1, channel=32,
                 split_factor=16, pretrained=True, use_uncertainty=False,
                 use_mixstyle=True):
        super(CFFANet_OOD, self).__init__()
        self.use_uncertainty = use_uncertainty

        # MobileNetV2 Backbone (lightweight already)
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.layer1 = mobilenet.features[0]
        self.layer2 = mobilenet.features[1]
        self.layer3 = nn.Sequential(mobilenet.features[2], mobilenet.features[3])
        self.layer4 = nn.Sequential(mobilenet.features[4], mobilenet.features[5],
                                    mobilenet.features[6])
        self.layer5 = nn.Sequential(mobilenet.features[7], mobilenet.features[8],
                                    mobilenet.features[9], mobilenet.features[10])
        self.layer6 = nn.Sequential(mobilenet.features[11], mobilenet.features[12],
                                    mobilenet.features[13])
        self.layer7 = nn.Sequential(mobilenet.features[14], mobilenet.features[15],
                                    mobilenet.features[16])
        self.layer8 = nn.Sequential(mobilenet.features[17])

        # Bridge and attention modules
        self.MFA_block = MFA_E(use_mixstyle=use_mixstyle)
        self.esa_4 = ESA_block(96, use_mixstyle=use_mixstyle)
        self.esa_3 = ESA_block(32, use_mixstyle=use_mixstyle)
        self.attention2 = Att_block_1(24)
        self.attention1 = Att_block_1(16)

        self.decoder5 = LightweightDecoderBlock(320, 64, use_dropblock=True)  # 320 -> 64 (was 128)
        self.refine5 = ResidualBlock(64, use_mixstyle=use_mixstyle)
        self.decoder4 = LightweightDecoderBlock(160, 48, use_dropblock=True)  # 64+96=160 -> 48 (was 96)
        self.refine4 = ResidualBlock(48, use_mixstyle=use_mixstyle)
        self.decoder3 = LightweightDecoderBlock(80, 24, use_dropblock=True)  # 48+32=80 -> 24
        self.decoder2 = LightweightDecoderBlock(48, 16, use_dropblock=True)  # 24+24=48 -> 16
        self.decoder1 = LightweightDecoderBlock(32, 16, use_dropblock=False)  # 16+16=32 -> 16 (no dropblock at final layer)
        # Final layers - use 1x1 conv instead of 3x3 for final prediction
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.finalconv = nn.Conv2d(16, num_classes, 1)  # 1x1 instead of 3x3

    def forward(self, input):

        # Encoder
        x = self.layer1(input)  # (B, 16, 256, 256)
        e1 = self.layer2(x)  # (B, 16, 256, 256)
        e2 = self.layer3(e1)  # (B, 24, 128, 128)
        e3 = self.layer4(e2)  # (B, 32, 64, 64)
        l5 = self.layer5(e3)  # (B, 64, 32, 32)
        e4 = self.layer6(l5)  # (B, 96, 32, 32)
        l7 = self.layer7(e4)  # (B, 160, 16, 16)
        e5 = self.layer8(l7)  # (B, 320, 16, 16)
        # Bridge (no PPM - removed)
        encoder_booster = self.MFA_block(e5)  # (B, 320, 16, 16)

        # NEW Lightweight Decoder
        d5 = self.decoder5(encoder_booster)  # (B, 64, 32, 32)
        d5 = self.refine5(d5)  # Refine features
        d4_ESAM_4 = self.esa_4(e4)  # (B, 96, 32, 32)
        d4_cat = torch.cat([d5, d4_ESAM_4], dim=1)  # (B, 160, 32, 32)
        d4 = self.decoder4(d4_cat)  # (B, 48, 64, 64)
        d4 = self.refine4(d4)
        d3_ESAM_3 = self.esa_3(e3)  # (B, 32, 64, 64)
        d3_cat = torch.cat([d4, d3_ESAM_3], dim=1)  # (B, 80, 64, 64)
        d3 = self.decoder3(d3_cat)  # (B, 24, 128, 128)
        e2_att = self.attention2(e2)  # (B, 24, 128, 128)
        d2_cat = torch.cat([d3, e2_att], dim=1)  # (B, 48, 128, 128)
        d2 = self.decoder2(d2_cat)  # (B, 16, 256, 256)
        e1_att = self.attention1(e1)  # (B, 16, 256, 256)
        d1_cat = torch.cat([d2, e1_att], dim=1)  # (B, 32, 256, 256)
        d1 = self.decoder1(d1_cat)  # (B, 16, 512, 512)
        # Final output
        out = self.final_upsample(d1)  # (B, 16, 1024, 1024) -> resize to 512
        out = self.finalconv(out)  # (B, 1, 512, 512)
        # Resize to exact input size if needed
        if out.size()[2:] != input.size()[2:]:
            out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)

        out = torch.sigmoid(out)
        return out


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    num_classes = 1
    in_batch, inchannel, in_h, in_w = 2, 3, 512, 512
    x = torch.randn(in_batch, inchannel, in_h, in_w)

    print("=" * 70)
    print("Testing Lightweight CFFANet_OOD")
    print("=" * 70)

    # Test model
    net = CFFANet_OOD(use_uncertainty=False, use_mixstyle=True, pretrained=False)
    net.eval()

    with torch.no_grad():
        out = net(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    num_params = count_parameters(net)
    print(f"\n✅ Total trainable parameters: {num_params:,}")
    print(f"   (~{num_params / 1e6:.2f}M parameters)")
