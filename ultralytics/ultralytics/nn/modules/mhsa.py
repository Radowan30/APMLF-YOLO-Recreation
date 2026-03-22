"""MHSA with BoTNet-style decomposed 2D position encoding.

Paper: APMLF-YOLO Section 3.2, Fig. 5
Applied to P5 feature map (20x20 = 400 tokens) after AFPN fusion.
Full H×W attention — NOT axial (Fig. 5 shows full HxW token grid).

Attention formula:
  Attention = softmax((QK^T + QR^T) / sqrt(d_head)) @ V
  r = pos_h + pos_w   (decomposed, broadcast to [heads, H*W, d_head])

Args:
  dim: Feature channels (256 for YOLOv8n at P5)
  num_heads: Number of attention heads (8 per paper)
  dim_head: Channels per head (32, so 8x32=256)
  H, W: Spatial size of input feature map (20x20 for P5 at 640px input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSAContentPosition(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dim_head: int = 32,
        H: int = 20,
        W: int = 20,
    ):
        super().__init__()
        assert dim == num_heads * dim_head, \
            f"dim ({dim}) must equal num_heads ({num_heads}) x dim_head ({dim_head})"
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.H = H
        self.W = W

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

        # Decomposed learnable 2D position encoding (BoTNet style)
        self.pos_h = nn.Parameter(torch.randn(num_heads, H, 1, dim_head) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(num_heads, 1, W, dim_head) * 0.02)

    def _get_pos_bias(self, H: int, W: int) -> torch.Tensor:
        """Return position bias r of shape [nh, H*W, dh], interpolating if needed."""
        nh, dh = self.num_heads, self.dim_head
        if H == self.H and W == self.W:
            return (self.pos_h + self.pos_w).reshape(nh, H * W, dh)
        # Interpolate pos_h: [nh, self.H, 1, dh] -> [nh, H, 1, dh]
        # Treat nh*dh as channels, self.H as height, 1 as width
        pos_h = self.pos_h.permute(0, 3, 1, 2)  # [nh, dh, self.H, 1]
        pos_h = pos_h.reshape(1, nh * dh, self.H, 1)
        pos_h = F.interpolate(pos_h, size=(H, 1), mode="bilinear", align_corners=False)
        pos_h = pos_h.reshape(nh, dh, H, 1).permute(0, 2, 3, 1)  # [nh, H, 1, dh]

        pos_w = self.pos_w.permute(0, 3, 1, 2)  # [nh, dh, 1, self.W]
        pos_w = pos_w.reshape(1, nh * dh, 1, self.W)
        pos_w = F.interpolate(pos_w, size=(1, W), mode="bilinear", align_corners=False)
        pos_w = pos_w.reshape(nh, dh, 1, W).permute(0, 2, 3, 1)  # [nh, 1, W, dh]

        return (pos_h + pos_w).reshape(nh, H * W, dh)  # [nh, H*W, dh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, dh = self.num_heads, self.dim_head
        N = H * W  # number of tokens

        def reshape(t):
            # [B, C, H, W] -> [B, nh, N, dh]
            return t.reshape(B, nh, dh, N).permute(0, 1, 3, 2)

        Q = reshape(self.to_q(x))   # [B, nh, N, dh]
        K = reshape(self.to_k(x))   # [B, nh, N, dh]
        V = reshape(self.to_v(x))   # [B, nh, N, dh]

        # Content attention: [B, nh, N, N]
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Decomposed position bias: r[nh, N, dh] (interpolated if H/W differ from init)
        r = self._get_pos_bias(H, W)  # [nh, N, dh]
        # Position attention: Q @ r^T -> [B, nh, N, N]
        pos_attn = torch.matmul(Q, r.unsqueeze(0).transpose(-2, -1)) * self.scale
        attn = F.softmax(attn + pos_attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, nh, N, dh]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.to_out(out)
