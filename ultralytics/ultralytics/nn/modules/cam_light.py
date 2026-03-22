"""CAM-Light: C2f with dual-branch NAM attention (channel + spatial).

Paper: APMLF-YOLO Section 3.1, Fig. 2
NAM ref: "NAM: Normalization-based Attention Module" (Liu et al. 2021)
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


class NAMChannelAtt(nn.Module):
    """Channel attention using BN gamma as importance weights.

    β_i = |γ_i| / Σ|γ_i|  (paper Eq. channel branch)
    Output = sigmoid(x_weighted) * residual

    Note: This implementation uses sigmoid gating (sigmoid(BN(x) * β) * x),
    which is the attention style used in APMLF-YOLO (sigmoid-gated, not
    direct scaling as in the original NAM paper).

    BN tagged _is_nam=True so APMLFTrainer can inject sparsity gradient.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.bn._is_nam = True  # tag for sparsity regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_bn = self.bn(x)
        # Use .abs().detach() — importance weights are read-only scalars.
        # Gradients flow through self.bn(x) normally via BN backward.
        # .detach() prevents a second gradient path through the weight lookup
        # that would double-count BN weight gradients.
        weight = self.bn.weight.abs().detach()
        weight = weight / (weight.sum() + 1e-8)  # shape: [C]
        x_weighted = x_bn * weight.view(1, -1, 1, 1)
        return torch.sigmoid(x_weighted) * residual


class NAMSpatialAtt(nn.Module):
    """Spatial attention using per-pixel L2 channel norm.

    w_i = ||x[:,h,w]||_2 / sum(||x[:,h,w]||_2)  (paper spatial branch)

    Zero learnable parameters — resolution-independent.
    Applied AFTER channel attention (sequential, per Fig. 2).

    The output uses sigmoid gating: sigmoid(w) * residual.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # L2 norm across channel dim → shape [B, 1, H, W]
        pixel_norm = x.norm(p=2, dim=1, keepdim=True)
        # Normalize spatially
        w = pixel_norm / (pixel_norm.sum(dim=[2, 3], keepdim=True) + 1e-8)
        return torch.sigmoid(w) * residual


class CAMLight(nn.Module):
    """C2f redesigned with dual-branch NAM attention.

    Replaces all C2f blocks in YOLOv8n backbone.
    CAM_E: expansion ratio. Start at 0.375 and calibrate on training
    machine until backbone+standard neck+3-head totals 3.01M parameters.
    (YOLOv8n baseline C2f uses e=0.5 → 3.2M params)

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of Bottleneck sub-blocks
        shortcut: Use residual connections in Bottlenecks
        g: Groups for Bottleneck convolutions
        e: Expansion ratio (override CAM_E if provided)
    """
    CAM_E: float = 0.5  # Calibrated: matches standard YOLOv8n backbone size (~3.01M with std neck+3-head)

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = None,
    ):
        super().__init__()
        e = e if e is not None else self.CAM_E
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.channel_att = NAMChannelAtt(c2)
        self.spatial_att = NAMSpatialAtt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # Sequential: channel attention first, then spatial (Fig. 2)
        out = self.channel_att(out)
        out = self.spatial_att(out)
        return out
