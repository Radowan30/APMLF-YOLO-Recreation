import pytest
import torch
from ultralytics.nn.modules.mhsa import MHSAContentPosition


class TestMHSAContentPosition:
    def test_output_shape(self):
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        x = torch.randn(1, 256, 20, 20)
        assert m(x).shape == x.shape

    def test_batch_invariant(self):
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        x = torch.randn(3, 256, 20, 20)
        assert m(x).shape == (3, 256, 20, 20)

    def test_position_params_shape(self):
        """Decomposed position: pos_h[heads,H,1,d_head], pos_w[heads,1,W,d_head]."""
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        assert m.pos_h.shape == (8, 20, 1, 32)
        assert m.pos_w.shape == (8, 1, 20, 32)

    def test_differentiable(self):
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        x = torch.randn(1, 256, 20, 20, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_total_params_reasonable(self):
        """256-dim, 8 heads: 4x(256x256) conv + pos params ~263K."""
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        params = sum(p.numel() for p in m.parameters())
        assert 250_000 < params < 300_000, f"Unexpected param count: {params}"
