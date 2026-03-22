"""AFPN: Asymptotic Feature Pyramid Network with ASFF adaptive fusion.

Paper: APMLF-YOLO Section 3.2, Fig. 3
AFPN ref: "AFPN: Asymptotic Feature Pyramid Network for Object Detection" (Yang et al. 2023)

Architecture:
  Stage 1: ASFF_2 fuses adjacent P3<->P4 (P5 isolated — asymptotic)
  Stage 2: ASFF_3 fuses all 3 scales (P5 joins asymptotically)
  BasicBlock refines each output at each stage.
  compress_c=8 for efficient weight computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class ASFF2(nn.Module):
    """Adaptive Spatial Feature Fusion for 2 inputs.

    Learns spatially-varying softmax weights per location.
    compress_c=8: internal channel reduction for weight computation.
    """
    def __init__(self, inter_dim: int, compress_c: int = 8):
        super().__init__()
        self.weight_l1 = Conv(inter_dim, compress_c, 1)
        self.weight_l2 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, 1, bias=True)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        w_cat = torch.cat([self.weight_l1(x1), self.weight_l2(x2)], dim=1)
        w = F.softmax(self.weight_levels(w_cat), dim=1)
        fused = x1 * w[:, 0:1] + x2 * w[:, 1:2]
        return self.conv(fused)


class ASFF3(nn.Module):
    """Adaptive Spatial Feature Fusion for 3 inputs."""
    def __init__(self, inter_dim: int, compress_c: int = 8):
        super().__init__()
        self.weight_l1 = Conv(inter_dim, compress_c, 1)
        self.weight_l2 = Conv(inter_dim, compress_c, 1)
        self.weight_l3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, bias=True)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        w_cat = torch.cat(
            [self.weight_l1(x1), self.weight_l2(x2), self.weight_l3(x3)], dim=1
        )
        w = F.softmax(self.weight_levels(w_cat), dim=1)
        fused = x1 * w[:, 0:1] + x2 * w[:, 1:2] + x3 * w[:, 2:3]
        return self.conv(fused)


class AFPNBasicBlock(nn.Module):
    """Residual refinement block used after each ASFF fusion stage."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.conv2(self.conv1(x))) + x)


def _make_blocks(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[AFPNBasicBlock(channels) for _ in range(num_blocks)])


class BlockBodyP345(nn.Module):
    """Two-stage asymptotic feature fusion for P3/P4/P5.

    Stage 1 (asymptotic: P5 isolated):
      new_P3 = ASFF2(P3, upsample(P4))
      new_P4 = ASFF2(downsample(P3), P4)
      -> BasicBlock refinement

    Stage 2 (P5 joins):
      fin_P3 = ASFF3(P3, upsample(P4), upsample x2(P5))
      fin_P4 = ASFF3(downsample(P3), P4, upsample(P5))
      fin_P5 = ASFF3(downsample x2(P3), downsample(P4), P5)
      -> BasicBlock refinement

    Args:
        channels: [c_p3, c_p4, c_p5] internal channel counts
        num_blocks: AFPNBasicBlock count per output per stage
    """
    def __init__(self, channels: list, num_blocks: int = 4):
        super().__init__()
        c3, c4, c5 = channels

        # Stage 1: resize + project cross-scale features to match target channel count
        # P4 -> P3 resolution: upsample (spatial) + project (channel c4 -> c3)
        self.up_p4_to_p3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Conv(c4, c3, 1))
        # P3 -> P4 resolution: downsample with stride conv (also adjusts channels c3 -> c4)
        self.dn_p3_to_p4 = Conv(c3, c4, 3, 2)

        # Stage 1 ASFF (both inputs have matching channels)
        self.asff2_p3_s1 = ASFF2(c3)
        self.asff2_p4_s1 = ASFF2(c4)

        # Stage 1 refinement
        self.blocks_p3_s1 = _make_blocks(c3, num_blocks)
        self.blocks_p4_s1 = _make_blocks(c4, num_blocks)

        # Stage 2: resize + project cross-scale features to match target channel count
        # For P3 output: need P4@c4->c3 and P5@c5->c3 at P3 spatial size
        self.up_p4_to_p3_s2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Conv(c4, c3, 1))
        self.up_p5_to_p3_s2 = nn.Sequential(nn.Upsample(scale_factor=4, mode='nearest'), Conv(c5, c3, 1))
        # For P4 output: need P3@c3->c4 and P5@c5->c4 at P4 spatial size
        self.dn_p3_to_p4_s2 = Conv(c3, c4, 3, 2)
        self.up_p5_to_p4_s2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Conv(c5, c4, 1))
        # For P5 output: need P3@c3->c5 and P4@c4->c5 at P5 spatial size
        self.dn_p3_to_p5_s2 = nn.Sequential(Conv(c3, c4, 3, 2), Conv(c4, c5, 3, 2))
        self.dn_p4_to_p5_s2 = Conv(c4, c5, 3, 2)

        # Stage 2 ASFF
        self.asff3_p3_s2 = ASFF3(c3)
        self.asff3_p4_s2 = ASFF3(c4)
        self.asff3_p5_s2 = ASFF3(c5)

        # Stage 2 refinement
        self.blocks_p3_s2 = _make_blocks(c3, num_blocks)
        self.blocks_p4_s2 = _make_blocks(c4, num_blocks)
        self.blocks_p5_s2 = _make_blocks(c5, num_blocks)

    def forward(self, x: list) -> list:
        p3, p4, p5 = x

        # Stage 1: adjacent fusion (all cross-scale inputs projected to target channels)
        new_p3 = self.asff2_p3_s1(p3, self.up_p4_to_p3(p4))
        new_p4 = self.asff2_p4_s1(self.dn_p3_to_p4(p3), p4)
        p3 = self.blocks_p3_s1(new_p3)
        p4 = self.blocks_p4_s1(new_p4)

        # Stage 2: full 3-scale fusion (all inputs projected to target channels)
        fin_p3 = self.asff3_p3_s2(p3, self.up_p4_to_p3_s2(p4), self.up_p5_to_p3_s2(p5))
        fin_p4 = self.asff3_p4_s2(self.dn_p3_to_p4_s2(p3), p4, self.up_p5_to_p4_s2(p5))
        fin_p5 = self.asff3_p5_s2(self.dn_p3_to_p5_s2(p3), self.dn_p4_to_p5_s2(p4), p5)

        return [
            self.blocks_p3_s2(fin_p3),
            self.blocks_p4_s2(fin_p4),
            self.blocks_p5_s2(fin_p5),
        ]


class AFPNNeck(nn.Module):
    """Full AFPN neck: project -> asymptotic fusion -> project back.

    Args:
        in_channels: [c_p3, c_p4, c_p5] from backbone (e.g. [64,128,256] for YOLOv8n at n-scale)
        out_channels: unified output channels for detection heads (256)
        factor: channel reduction factor for internal AFPN computation (start: 4)
        num_blocks: AFPNBasicBlock count per stage per output (start: 4)

    CALIBRATION NOTE: Adjust factor and num_blocks on training machine until
    full model (backbone + AFPNNeck + APMLFDetect heads) reaches 5.1M params.
    """
    def __init__(
        self,
        in_channels: list = None,
        out_channels: int = 256,
        factor: int = 4,
        num_blocks: int = 4,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256]
        ic = [c // factor for c in in_channels]  # e.g. [16, 32, 64]

        self.conv_in = nn.ModuleList(
            [Conv(c, ic[i], 1) for i, c in enumerate(in_channels)]
        )
        self.body = BlockBodyP345(ic, num_blocks=num_blocks)
        self.conv_out = nn.ModuleList(
            [Conv(ic[i], out_channels, 1) for i in range(3)]
        )

    def forward(self, x: list) -> list:
        projected = [self.conv_in[i](x[i]) for i in range(3)]
        fused = self.body(projected)
        return [self.conv_out[i](fused[i]) for i in range(3)]
