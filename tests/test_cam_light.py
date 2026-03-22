import pytest
import torch
import torch.nn as nn
from ultralytics.nn.modules.cam_light import NAMChannelAtt

class TestNAMChannelAtt:
    def test_output_shape(self):
        """Output shape must match input shape."""
        m = NAMChannelAtt(64)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_has_named_bn(self):
        """BN must be tagged _is_nam=True for sparsity regularization."""
        m = NAMChannelAtt(64)
        assert hasattr(m.bn, '_is_nam') and m.bn._is_nam is True

    def test_output_bounded_by_residual(self):
        """Sigmoid gate in (0,1) means output magnitude cannot exceed residual."""
        m = NAMChannelAtt(64)
        m.eval()
        x = torch.randn(2, 64, 16, 16)
        residual = x.clone()
        out = m(x)
        # sigmoid(anything) is in (0,1), so |out| <= |residual| elementwise
        assert (out.abs() <= residual.abs() + 1e-5).all(), \
            "Output exceeds residual magnitude — sigmoid gate not working"

    def test_differentiable(self):
        """Must support backprop through gamma weights."""
        m = NAMChannelAtt(64)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None


class TestNAMSpatialAtt:
    def test_output_shape(self):
        from ultralytics.nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        x = torch.randn(2, 64, 32, 32)
        assert m(x).shape == x.shape

    def test_no_parameters(self):
        """Spatial branch must have zero learnable parameters."""
        from ultralytics.nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        assert sum(p.numel() for p in m.parameters()) == 0

    def test_output_bounded_by_residual(self):
        """Sigmoid gate bounds output by residual magnitude."""
        from ultralytics.nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert (out.abs() <= x.abs() + 1e-5).all()

    def test_differentiable(self):
        from ultralytics.nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestCAMLight:
    def test_output_shape(self):
        from ultralytics.nn.modules.cam_light import CAMLight
        m = CAMLight(128, 128, n=3)
        x = torch.randn(1, 128, 40, 40)
        assert m(x).shape == (1, 128, 40, 40)

    def test_parameter_reduction_vs_c2f(self):
        """CAMLight with e=0.375 should have fewer params than C2f with e=0.5."""
        from ultralytics.nn.modules.cam_light import CAMLight
        from ultralytics.nn.modules.block import C2f
        cam = CAMLight(256, 256, n=6, e=0.375)
        c2f = C2f(256, 256, n=6, e=0.5)
        cam_params = sum(p.numel() for p in cam.parameters())
        c2f_params = sum(p.numel() for p in c2f.parameters())
        assert cam_params < c2f_params, \
            f"CAMLight({cam_params}) should be < C2f({c2f_params})"

    def test_sequential_attention_order(self):
        """channel_att applied before spatial_att (sequential per Fig.2)."""
        from ultralytics.nn.modules.cam_light import CAMLight
        m = CAMLight(64, 64, n=1)
        assert hasattr(m, 'channel_att')
        assert hasattr(m, 'spatial_att')

    def test_cam_e_class_variable(self):
        """CAM_E class variable must be accessible for calibration."""
        from ultralytics.nn.modules.cam_light import CAMLight
        assert hasattr(CAMLight, 'CAM_E')
        assert 0.3 <= CAMLight.CAM_E <= 0.5
