import pytest
import torch
from ultralytics.nn.modules.afpn import ASFF2, ASFF3, AFPNBasicBlock, BlockBodyP345, AFPNNeck


class TestASFF2:
    def test_output_shape(self):
        """Two inputs of same spatial size → same size output."""
        m = ASFF2(inter_dim=64)
        x1 = torch.randn(2, 64, 20, 20)
        x2 = torch.randn(2, 64, 20, 20)
        out = m(x1, x2)
        assert out.shape == (2, 64, 20, 20)

    def test_weights_sum_to_one(self):
        """Softmax weights across 2 inputs must sum to 1 per location."""
        import torch.nn.functional as F
        m = ASFF2(inter_dim=32)
        m.eval()
        x1 = torch.randn(1, 32, 10, 10)
        x2 = torch.randn(1, 32, 10, 10)
        with torch.no_grad():
            wc = torch.cat([m.weight_l1(x1), m.weight_l2(x2)], 1)
            w = F.softmax(m.weight_levels(wc), dim=1)
        assert torch.allclose(w.sum(dim=1), torch.ones(1, 10, 10), atol=1e-5)


class TestASFF3:
    def test_output_shape(self):
        m = ASFF3(inter_dim=64)
        x1 = torch.randn(2, 64, 20, 20)
        x2 = torch.randn(2, 64, 20, 20)
        x3 = torch.randn(2, 64, 20, 20)
        assert m(x1, x2, x3).shape == (2, 64, 20, 20)


class TestAFPNBasicBlock:
    def test_residual_shape(self):
        m = AFPNBasicBlock(64)
        x = torch.randn(2, 64, 20, 20)
        assert m(x).shape == x.shape


class TestBlockBodyP345:
    def test_output_shapes(self):
        """Must return 3 tensors matching P3/P4/P5 spatial sizes."""
        m = BlockBodyP345(channels=[16, 32, 64], num_blocks=2)
        p3 = torch.randn(1, 16, 80, 80)
        p4 = torch.randn(1, 32, 40, 40)
        p5 = torch.randn(1, 64, 20, 20)
        out = m([p3, p4, p5])
        assert len(out) == 3
        assert out[0].shape == p3.shape
        assert out[1].shape == p4.shape
        assert out[2].shape == p5.shape


class TestAFPNNeck:
    def test_forward(self):
        """Full AFPN: takes [P3,P4,P5] at YOLOv8n scales → 3 unified outputs."""
        neck = AFPNNeck(in_channels=[64, 128, 256], out_channels=256, factor=4, num_blocks=4)
        p3 = torch.randn(1, 64, 80, 80)
        p4 = torch.randn(1, 128, 40, 40)
        p5 = torch.randn(1, 256, 20, 20)
        outs = neck([p3, p4, p5])
        assert len(outs) == 3
        assert outs[0].shape == (1, 256, 80, 80)
        assert outs[1].shape == (1, 256, 40, 40)
        assert outs[2].shape == (1, 256, 20, 20)
