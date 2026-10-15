# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "C3HorNet",
    "C3HorBlock",
    "C3HorBlock_light",
    "GhostBottleneck",
    "Bottleneck",
    "gnconv",
    "SpatialBottleneck",
    "DropPath",
    "MLP",
    "HorBlock",
    "NAM",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "GDRConv",
    "TDAM",
    "CTDAM",
    "GTDRBlock",
    "GTDRBlock_CTDAM",
    "HARFM",
    "FFAM",
    "ELA",
    "DyHead",
    "DyHeadBlock",
)

class DyHeadBlock(nn.Module):
    """
    Dynamic Head Block (Dai et al., CVPR 2021).
    Applies three sequential attention mechanisms:
      1. Scale-aware attention (across feature levels)
      2. Spatial-aware attention (deformable convolution)
      3. Task-aware attention (dynamic ReLU on channels)
    
    Simplified version compatible with Ultralytics single-level input.
    Each detection head feature map is processed independently.
    """

    def __init__(self, c1, c2, num_heads=4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # Scale-aware attention
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1),
            nn.Hardsigmoid(inplace=True),
        )

        # Spatial-aware attention (use standard deformable-like conv)
        self.offset_conv = nn.Conv2d(c1, 2 * 3 * 3, 3, padding=1, bias=True)
        self.spatial_conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(c2)

        # Task-aware attention (Dynamic ReLU)
        self.task_fc1 = nn.Conv2d(c2, c2 // 4, 1)
        self.task_fc2 = nn.Conv2d(c2 // 4, c2 * 2, 1)  # alpha and beta

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        # 1. Scale-aware attention
        scale_attn = self.scale_attn(x)
        x = x * scale_attn

        # 2. Spatial-aware attention
        x = self.spatial_bn(self.spatial_conv(x))

        # 3. Task-aware attention (DyReLU)
        avg = F.adaptive_avg_pool2d(x, 1)
        task_params = self.task_fc2(F.relu(self.task_fc1(avg)))
        alpha, beta = torch.split(task_params, self.c2, dim=1)
        alpha = alpha.sigmoid()  # gating
        x = x * alpha + beta

        return x

class DyHead(nn.Module):
    """
    Multi-block DyHead module for YOLO detection heads.
    Stacks multiple DyHeadBlocks sequentially.
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        num_blocks (int): Number of DyHead blocks to stack. Default: 2.
    """

    def __init__(self, c1, c2, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DyHeadBlock(c1 if i == 0 else c2, c2) for i in range(num_blocks)]
        )
        # Channel projection if needed
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return self.blocks(x)

class ELA(nn.Module):
    """
    Efficient Local Attention-style gating used inside FFAM.

    This is a lightweight spatial-channel attention approximation:
    - pool along width to keep vertical structure
    - pool along height to keep horizontal structure
    - project each with 1x1 conv
    - combine as multiplicative attention

    Input/output shape: [B, C, H, W]
    """

    def __init__(self, c: int, groups: int = 16):
        super().__init__()
        g = max(1, min(groups, c))
        while c % g != 0 and g > 1:
            g -= 1

        self.conv_h = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn_h = nn.GroupNorm(g, c)
        self.gn_w = nn.GroupNorm(g, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Height-aware descriptor: B,C,H,1
        x_h = x.mean(dim=3, keepdim=True)
        # Width-aware descriptor: B,C,1,W
        x_w = x.mean(dim=2, keepdim=True)

        a_h = torch.sigmoid(self.gn_h(self.conv_h(x_h)))
        a_w = torch.sigmoid(self.gn_w(self.conv_w(x_w)))

        return a_h * a_w

class HARFM(nn.Module):
    """
    HARFM: High- and Adjacent-level Recalibration Fusion Module
    Output resolution is aligned to the LOW-resolution input branch by default
    in terms of pyramid level naming:
      [P3, P4] -> output P3 spatial size
      [P4, P5] -> output P4 spatial size
    """

    def __init__(
        self,
        c1,
        c2,
        reduction: int =2,
        use_gn: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        # This section is where the module is configud in 2 sections:
        assert isinstance(c1, (list, tuple)) and len(c1) == 2, \
            f"HARFM expects c1 as [c_low, c_high], got {c1}"

        c_low, c_high = c1 # split theb channels into low and high branches
        hidden = max(c2 // max(int(reduction), 1), 16) 

        #  project low and high branches to c2 channels for fusion
        self.low_proj = Conv(c_low, c2, k=3, s=1) 
        self.high_proj = Conv(c_high, c2, k=1, s=1)

        # fuse low and high features with simple concatenation followed by 1x1 conv to mix channels
        self.fuse_in = Conv(2 * c2, c2, k=1, s=1)

        # generate attention weights with a lightweight bottleneck: dw -> pw -> act -> dw -> pw -> sigmoid
        self.dw1 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
        self.pw1 = nn.Conv2d(c2, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False)
        self.pw2 = nn.Conv2d(hidden, c2, kernel_size=1, stride=1, padding=0, bias=False)

        # Typicaly default ot GroupNnnorm but can be switched to BatchNorm or Identity if desired. GroupNorm with groups=16 is a common choice for attention modules.
        if use_gn:
            g = min(16, c2)
            while c2 % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, c2)
        else:
            self.norm = nn.BatchNorm2d(c2)

        # Activation function and output projection. Sigmoid is used to generate attention weights in the range [0,1]. The output projection is a 3x3 conv that produces the final recalibrated feature map.
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.residual = residual
        self.out_proj = Conv(c2, c2, k=3, s=1)

    def forward(self, xs):
        assert isinstance(xs, (list, tuple)) and len(xs) == 2, \
            f"HARFM forward expects [x_low, x_high], got {type(xs)}"

        x_low, x_high = xs

        # IMPORTANT:
        # align to low-side spatial size so:
        # [P3, P4] -> output P3 size
        # [P4, P5] -> output P4 size
        target_size = x_low.shape[2:]

        if x_high.shape[2:] != target_size:
            x_high = F.interpolate(x_high, size=target_size, mode="nearest")

        x_low = self.low_proj(x_low)
        x_high = self.high_proj(x_high)

        x = torch.cat([x_low, x_high], dim=1)
        x = self.fuse_in(x)
        identity = x

        w = self.dw1(x)
        w = self.pw1(w)
        w = self.act(w)
        w = self.dw2(w)
        w = self.pw2(w)
        w = self.norm(w)
        w = self.sigmoid(w)

        x = identity * w + identity # if self.residual else identity * w
        return self.out_proj(x)

class FFAM(nn.Module):
    """
    FFAM: Feature Filtering and Aggregation Module
    Flexible constructor to support YAML phases:
      Phase 1: FFAM, [c2]
      Phase 2: FFAM, [c2, use_ela, kernels]
      Phase 3: FFAM, [c2, use_ela, kernels, residual]

    Expected inputs:
      xs = [x_low, x_mid, x_high]

    Design goals:
    - align all 3 scales to the mid scale
    - filter low and mid features with semantic/spatial attention
    - aggregate into one fused mid-scale feature
    - enrich with multi-kernel depthwise branches
    """

    def __init__(
        self,
        c1,
        c2,
        use_ela: bool = True,
        kernels=(5, 7, 9, 11),
        residual: bool = True,
    ):
        super().__init__()

        assert isinstance(c1, (list, tuple)) and len(c1) == 3, \
            f"FFAM expects c1 as [c_low, c_mid, c_high], got {c1}"

        c_low, c_mid, c_high = c1

        self.low_proj = Conv(c_low, c2, k=3, s=1)
        self.mid_proj = Conv(c_mid, c2, k=3, s=1)
        self.high_proj = Conv(c_high, c2, k=1, s=1)

        self.use_ela = use_ela
        self.residual = residual

        if self.use_ela:
            self.ela_mid = ELA(c2)
            self.ela_high = ELA(c2)

        # fuse aligned features first
        self.pre_fuse = Conv(c2, c2, k=3, s=1)

        # multi-kernel depthwise branches
        self.dw_branches = nn.ModuleList()
        for k in kernels:
            assert isinstance(k, int) and k % 2 == 1, f"FFAM kernels must be odd ints, got {k}"
            self.dw_branches.append(
                nn.Conv2d(c2, c2, kernel_size=k, stride=1, padding=k // 2, groups=c2, bias=False)
            )

        self.branch_bn = nn.ModuleList([nn.BatchNorm2d(c2) for _ in kernels])
        self.branch_act = nn.SiLU()

        self.post_fuse = Conv(c2 * len(kernels), c2, k=1, s=1)
        self.out_proj = Conv(c2, c2, k=3, s=1)

    def forward(self, xs):
        assert isinstance(xs, (list, tuple)) and len(xs) == 3, \
            f"FFAM forward expects [x_low, x_mid, x_high], got type={type(xs)} len={len(xs) if isinstance(xs, (list, tuple)) else 'NA'}"

        x_low, x_mid, x_high = xs
        target_size = x_mid.shape[2:]  # central fusion hub is mid-scale

        if x_low.shape[2:] != target_size:
            x_low = F.interpolate(x_low, size=target_size, mode="nearest")
        if x_high.shape[2:] != target_size:
            x_high = F.interpolate(x_high, size=target_size, mode="nearest")

        x_low = self.low_proj(x_low)
        x_mid = self.mid_proj(x_mid)
        x_high = self.high_proj(x_high)

        if self.use_ela:
            a_high = self.ela_high(x_high)
            a_mid = self.ela_mid(x_mid)

            # semantic-guided filtering + self-filtering
            fused = a_high * x_low + x_high + a_mid * x_mid
        else:
            fused = x_low + x_mid + x_high

        fused = self.pre_fuse(fused)
        identity = fused

        branch_outs = []
        for dw, bn in zip(self.dw_branches, self.branch_bn):
            y = dw(fused)
            y = bn(y)
            y = self.branch_act(y)
            branch_outs.append(y)

        x = torch.cat(branch_outs, dim=1)
        x = self.post_fuse(x)

        if self.residual:
            x = x + identity

        return self.out_proj(x)                                                                                                                                                                                                                                 

class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    """Channel MLP using 1x1 convs (ConvNeXt/Transformer style)."""
    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.SiLU):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class HorBlock(nn.Module):
    """HorNet-style block: (Norm->gnConv->Res) + (Norm->MLP->Res), with optional DropPath."""
    def __init__(self, dim, order=5, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        # Cap order to prevent integer division producing zeros in gnconv.
        # order=k means dims = [dim // 2^(k-1), ..., dim // 1], so max viable order is ~log2(dim).
        import math
        max_order = max(1, int(math.log2(dim)))  # At least 1, at most log2(dim)
        capped_order = min(order, max_order)
        if capped_order < order:
            print(f"[HorBlock] order={order} capped to {capped_order} for dim={dim}")
        
        self.norm1 = nn.GroupNorm(1, dim)
        self.gnconv = gnconv(dim, order=capped_order, s=1.0)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.gnconv(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class NAM(nn.Module):
    """Normalization-based Attention Module (NAM).

    Lightweight implementation intended to be inserted in neck fusion outputs.
    Callable from YAML as: [-1, 1, NAM, [c]] where c is number of channels.
    """

    def __init__(self, c, eps: float = 1e-6, use_abs: bool = True, residual: bool = True, alpha_init: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.use_abs = bool(use_abs)
        self.residual = bool(residual)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init))) if self.residual else None

        # BatchNorm with affine weights (gamma) used for channel attention
        self.bn = nn.BatchNorm2d(c, affine=True, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y: normalized features (via BatchNorm)
        y = self.bn(x)

        # Channel attention via BN gamma (weight)
        gamma = self.bn.weight  # shape [C]
        if self.use_abs:
            gamma = gamma.abs()
        denom_c = gamma.sum() + self.eps
        w_c = (gamma / denom_c).view(1, -1, 1, 1)
        mc = torch.sigmoid(y * w_c)

        # Spatial attention (PN-inspired minimal variant)
        s = y.mean(dim=1, keepdim=True)  # [B,1,H,W]
        if self.use_abs:
            s = s.abs()
        B = s.shape[0]
        denom_s = s.view(B, -1).sum(dim=1).view(B, 1, 1, 1) + self.eps
        w_s = s / denom_s
        w_s = w_s.clamp(min=-50.0, max=50.0)
        ms = torch.sigmoid(w_s)

        out = x * mc * ms
        if self.residual:
            return x + self.alpha * (out - x)
        return out

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        print(">>> USING Standard Bottleneck <<<")
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class gnconv(nn.Module):
    """Gated Group Convolution (gnconv) for spatial mixing with multi-order decomposition."""

    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        """Initialize gnconv with multi-order group convolutions.
        
        Args:
            dim (int): Input/output channel dimension.
            order (int): Order of decomposition. Default: 5.
            gflayer (nn.Module, optional): Custom gating function layer. Default: None (uses depthwise conv).
            h (int): Height parameter for custom gflayer. Default: 14.
            w (int): Width parameter for custom gflayer. Default: 8.
            s (float): Scaling factor for output. Default: 1.0.
        """
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        #self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.proj_in = nn.Conv2d(dim, self.dims[0] + sum(self.dims), 1) #change this line for next round


        if gflayer is None:
            # Default: use depthwise convolution with kernel size 7
            self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), 7, 1, 3, groups=sum(self.dims))
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        """Forward pass through gnconv.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            mask (torch.Tensor, optional): Mask tensor. Default: None.
            dummy (bool): Dummy parameter for compatibility. Default: False.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

class SpatialBottleneck(nn.Module):
    """Spatial bottleneck."""
                                                            #e=0.5
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0,gnconv_order=5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        print(">>> USING Spatial Bottleneck <<<")
        c_ = int(c2 * e)  # hidden channels
        self.pw1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = nn.GroupNorm(1, c_)
        self.spatial = gnconv(c_, order=gnconv_order, s=1.0)
        self.pw2 = nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0, bias=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        y = self.pw1(x)
        y = self.norm(y)
        y = self.spatial(y)
        y = self.pw2(y)
        return x + y if self.add else y

class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.teTDAMnsor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        print(">>> USING C3k2 Bottleneck <<<")
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) 
            if c3k 
            else Bottleneck(self.c, self.c, shortcut, g) 
            for _ in range(n)
        )

class C3HorNet(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        print(">>> USING C3HorNet Bottleneck <<<")
        """Initializes the C3HorNet module, a CSP Bottleneck with 2 convolutions inspired by HorNet architecture."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) 
            if c3k 
            else SpatialBottleneck(self.c, self.c, shortcut)
            for _ in range(n)
        )

class C3HorBlock(C2f):
    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True, g=1, order=5, mlp_ratio=4.0, drop_path=0.0):
        print(">>> USING C3HorBlock Bottleneck <<<")
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            HorBlock(self.c, order=order, mlp_ratio=mlp_ratio, drop_path=drop_path) 
            for _ in range(n))

class C3HorBlock_light(C2f):
    """Lighter variant of C3HorBlock with reduced order and MLP hidden ratio to cut parameters/FLOPs."""

    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True, g=1, order=3, mlp_ratio=1.0, drop_path=0.0):
        print(">>> USING C3HorBlock_light Bottleneck <<<")
        super().__init__(c1, c2, n, shortcut, g, e)
        # Lower `order` reduces gnconv channel splits; lower `mlp_ratio` shrinks MLP hidden dim
        self.m = nn.ModuleList(
            HorBlock(self.c, order=order, mlp_ratio=mlp_ratio, drop_path=drop_path)
            for _ in range(n)
        )

# This one got the best results so far
class GDRConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()

        hidden = c2 // 2

        self.primary = nn.Sequential(
            nn.Conv2d(c1, hidden, k, s, autopad(k), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.cheap = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

    def forward(self, x):
        y1 = self.primary(x)
        y2 = self.cheap(y1)
        return torch.cat([y1, y2], dim=1)
    
# This one got the worse results - might be issue with implementation. 

# Implementation of GTDR-Conv replacement of Conv blocks 

# class GDRConv(nn.Module):
#     """
#     GDR-Conv (Ghost + DyReLU) as described:

#       1) Primary branch: 1x1 Conv -> BN -> DyReLU
#       2) Expansion branch (sequential): DW 5x5 -> (optional PW 1x1) -> BN -> DyReLU
#       3) Output: concat(primary, expansion) along channels

#     The DW branch is implemented as true depthwise on the primary features, then a PW 1x1
#     to reach the required cheap channels (common Ghost-style cheap feature generation).
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel: int = 3,         # DW kernel size (3 or 5)
#         stride: int = 1,
#         ratio: int = 2,          # primary = ceil(out/ratio), cheap = out - primary
#         dyrelu_reduction: int = 4,
#         a_min: float = 0.0,
#         a_max: float = 2.0,
#     ):
#         super().__init__()
#         assert out_channels > 0
#         assert ratio >= 1
        
#         # Ensure kernel is odd and >= 3
#         if kernel < 3:
#             kernel = 3
#         if kernel % 2 == 0:
#             kernel += 1

#         primary = int((out_channels + ratio - 1) // ratio)  # ceil(out/ratio)
#         cheap = out_channels - primary
#         if cheap <= 0:
#             primary = max(out_channels - 1, 1)
#             cheap = out_channels - primary

#         self.primary_channels = primary
#         self.cheap_channels = cheap

#         # Primary: 1x1
#         self.pw = nn.Conv2d(in_channels, primary, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.pw_bn = nn.BatchNorm2d(primary)
#         self.pw_act = DyReLU(primary, reduction=dyrelu_reduction, a_min=a_min, a_max=a_max)

#         # Expansion: depthwise 5x5 (on primary output) + pointwise 1x1 to cheap channels
#         # Use autopad for correct 'same' padding 
#         pad = autopad(kernel)
#         self.dw = nn.Conv2d(primary, primary, kernel_size=kernel, stride=1, padding=pad,
#                             groups=primary, bias=False)
#         self.dw_pw = nn.Conv2d(primary, cheap, kernel_size=1, stride=1, padding=0, bias=False)

#         self.dw_bn = nn.BatchNorm2d(cheap)
#         self.dw_act = DyReLU(cheap, reduction=dyrelu_reduction, a_min=a_min, a_max=a_max)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y1 = self.pw_act(self.pw_bn(self.pw(x)))          # primary features
#         y2 = self.dw_pw(self.dw(y1))                      # spatial cheap features
#         y2 = self.dw_act(self.dw_bn(y2))
#         return torch.cat([y1, y2], dim=1)

# class DyReLU(nn.Module):
#     """
#     Dynamic ReLU (2-piece):
#         y = max(a1*x + b1, a2*x + b2)
#     where (a1,b1,a2,b2) are predicted per-channel via GAP + 1x1 convs.

#     Lightweight: negligible overhead vs standard activations.
#     """
#     def __init__(
#         self,
#         channels: int,
#         reduction: int = 4,
#         a_min: float = 0.0,
#         a_max: float = 2.0,
#     ):
#         super().__init__()
#         hidden = max(channels // reduction, 4)

#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
#         self.fc2 = nn.Conv2d(hidden, channels * 4, kernel_size=1, bias=True)

#         self.a_min = a_min
#         self.a_max = a_max

#         # Stabilize start: near-identity / ReLU-ish behavior initially
#         nn.init.zeros_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, _, _ = x.shape

#         s = self.pool(x)
#         s = F.relu(self.fc1(s), inplace=True)
#         theta = self.fc2(s).view(b, 4, c, 1, 1)

#         a1 = theta[:, 0]
#         b1 = theta[:, 1]
#         a2 = theta[:, 2]
#         b2 = theta[:, 3]

#         # Constrain slopes for stability
#         a1 = torch.sigmoid(a1) * (self.a_max - self.a_min) + self.a_min
#         a2 = torch.sigmoid(a2) * (self.a_max - self.a_min) + self.a_min

#         y1 = a1 * x + b1
#         y2 = a2 * x + b2
#         return torch.maximum(y1, y2)

# class GDRConv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, dyrelu_reduction=4, a_min=0.0, a_max=2.0):
#         super().__init__()
#         assert c2 % 2 == 0, "This exact half/half design assumes c2 is even."

#         hidden = c2 // 2

#         self.primary = nn.Sequential(
#             nn.Conv2d(c1, hidden, k, s, autopad(k), bias=False),
#             nn.BatchNorm2d(hidden),
#             DyReLU(hidden, reduction=dyrelu_reduction, a_min=a_min, a_max=a_max),
#         )

#         self.cheap = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 5, 1, autopad(5), groups=hidden, bias=False),  # padding=2
#             nn.BatchNorm2d(hidden),
#             DyReLU(hidden, reduction=dyrelu_reduction, a_min=a_min, a_max=a_max),
#         )

#     def forward(self, x):
#         y1 = self.primary(x)
#         y2 = self.cheap(y1)
#         return torch.cat([y1, y2], dim=1)

class TDAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class GTDRBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = GDRConv(c1, c2)
        self.attn = TDAM(c2)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x

class GTDRC3(nn.Module):
    """
    Faithful implementation based on paper paragraph.
    """
    print(">>> USING GTDRC3 Bottleneck <<<")

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()

        # First GTDRBlock
        self.block1 = GTDRBlock(c1, c2)

        # Second GTDRBlock
        self.block2 = GTDRBlock(c2, c2)

        # GhostBottleneck repetitions
        self.gb_blocks = nn.Sequential(
            *[GhostBottleneck(c2, c2) for _ in range(n)]
        )

        # Fusion block (after concat)
        self.fusion = GTDRBlock(2 * c2, c2)

        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        identity = x

        # Sequential main branch
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.gb_blocks(y2)

        # Concat first GTDRBlock output and final output
        out = torch.cat((y1, y3), dim=1)

        # Fusion
        out = self.fusion(out)

        # Residual
        if self.shortcut:
            out = out + identity

        return out

# -------------------------
# NEW CTDAM (attention only swap)
# -------------------------
class CTDAM(nn.Module):
    """
    Competitive TDAM (Channel competition + optional spatial context).
    Minimal integration: this module REPLACES TDAM; no other topology changes required.
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        temperature: float = 1.5,
        groups: int = 8,
        use_spatial_dw: bool = True,
        dw_kernel: int = 3,
        init_gamma: float = 1.0,
    ):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.channels = channels
        self.temperature = max(float(temperature), 1e-6)
        self.groups = int(max(groups, 1))
        assert channels % self.groups == 0, f"CTDAM: channels={channels} must be divisible by groups={self.groups}"

        if use_spatial_dw:
            pad = dw_kernel // 2
            self.spatial = nn.Conv2d(channels, channels, dw_kernel, 1, pad, groups=channels, bias=False)
        else:
            self.spatial = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)

        self.gamma = nn.Parameter(torch.ones(1) * float(init_gamma))

    def _softmax_grouped(self, logits: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        G = self.groups
        cg = C // G
        x = logits.view(B, G, cg) / self.temperature
        w = F.softmax(x, dim=-1)
        w = w * cg  # keep mean gate ~1
        return w.view(B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xs = self.spatial(x)
        y = self.pool(xs).view(B, C)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        gates = self._softmax_grouped(y).view(B, C, 1, 1)

        # identity-friendly scaling
        return x * (1.0 + self.gamma * (gates - 1.0))

# -------------------------
# Minimal-change GTDRBlock swap: TDAM -> CTDAM only
# -------------------------
class GTDRBlock_CTDAM(nn.Module):
    def __init__(self, c1, c2,
                 attn_reduction=4, attn_temp=1.5, attn_groups=8, use_spatial_dw=True):
        super().__init__()
        self.conv = GDRConv(c1, c2)
        self.attn = CTDAM(
            c2,
            reduction=attn_reduction,
            temperature=attn_temp,
            groups=attn_groups,
            use_spatial_dw=use_spatial_dw,
            dw_kernel=3,
            init_gamma=1.0,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x

# -------------------------
# Minimal-change GTDRC3: same structure, only block class swapped (THIS IS THE BEST PERFORMER SO FAR: To test P4 or P5 swap out)
# -------------------------
class GTDRC3_CTDAM(nn.Module):
    """
    Identical architecture to GTDRC3, except TDAM is replaced by CTDAM
    in block1, block2, and fusion.
    """
    print(">>> USING GTDRC3_CTDAM Bottleneck <<<")

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()

        self.block1 = GTDRBlock_CTDAM(c1, c2)
        self.block2 = GTDRBlock_CTDAM(c2, c2)

        self.gb_blocks = nn.Sequential(
            *[GhostBottleneck(c2, c2) for _ in range(n)]
        )

        self.fusion = GTDRBlock_CTDAM(2 * c2, c2)

        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.gb_blocks(y2)
        out = torch.cat((y1, y3), dim=1)
        out = self.fusion(out)
        if self.shortcut:
            out = out + identity
        return out


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        print(">>> USING C3k Bottleneck <<<")
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ //64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) 
            if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))
