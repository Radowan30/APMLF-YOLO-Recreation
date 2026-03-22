"""APMLFDetect: Combined AFPN neck + MHSA + 4-scale detection heads.

Receives [P2, P3, P4, P5] from backbone.
Pipeline:
  1. AFPN fuses P3/P4/P5 -> [P3', P4', P5'] (all at out_channels=256)
  2. MHSA applied to P5' (full 20x20 self-attention)
  3. 4 detection heads: P2 (direct from backbone), P3'/P4'/P5' (from AFPN+MHSA)

Inherits from Ultralytics Detect for compatibility with trainer/loss/export.
Input ch = (c_p2, c_p3, c_p4, c_p5) (e.g. (32, 64, 128, 256) for YOLOv8n at n-scale).
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.afpn import AFPNNeck
from ultralytics.nn.modules.mhsa import MHSAContentPosition


AFPN_OUT = 256  # Unified output channels from AFPN for P3'/P4'/P5'


class APMLFDetect(Detect):
    """4-scale detection: P2 (direct) + AFPN(P3,P4,P5) + MHSA on P5'.

    Args:
        nc: Number of classes (6 for PKU PCB dataset)
        ch: Input channel tuple (c_p2, c_p3, c_p4, c_p5)
    """
    def __init__(self, nc: int = 80, ch: tuple = ()):
        # Detect parent needs channels for all 4 detection heads:
        # P2 keeps its own channels, P3'/P4'/P5' all use AFPN_OUT
        ch_for_detect = (ch[0], AFPN_OUT, AFPN_OUT, AFPN_OUT)
        super().__init__(nc=nc, ch=ch_for_detect)

        # AFPN takes P3/P4/P5 channels
        self.afpn = AFPNNeck(
            in_channels=list(ch[1:]),   # [c_p3, c_p4, c_p5]
            out_channels=AFPN_OUT,
            factor=2,      # Calibrated: ic=[32,64,128] → 4.93M total params
            num_blocks=4,  # Calibrated: closest integer to 5.1M paper target
        )
        # MHSA on P5' (20x20 at 640px input, AFPN_OUT channels)
        self.mhsa = MHSAContentPosition(
            dim=AFPN_OUT,
            num_heads=8,
            dim_head=32,
            H=20,
            W=20,
        )

    def forward(self, x: list):
        p2, p3, p4, p5 = x

        # 1. AFPN fuses P3/P4/P5
        p3_fused, p4_fused, p5_fused = self.afpn([p3, p4, p5])

        # 2. MHSA refines P5' (full 20x20 self-attention)
        p5_attn = self.mhsa(p5_fused)

        # 3. Feed into Detect parent: [P2, P3', P4', P5_attn]
        return super().forward([p2, p3_fused, p4_fused, p5_attn])
