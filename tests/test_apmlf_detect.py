import pytest
import torch
from ultralytics.nn.modules.apmlf_detect import APMLFDetect


class TestAPMLFDetect:
    def _make_inputs(self, batch=2):
        """Simulate YOLOv8n backbone outputs at 640px input (after n-scale: x0.25 channels)."""
        return [
            torch.randn(batch, 32, 160, 160),   # P2
            torch.randn(batch, 64, 80, 80),     # P3
            torch.randn(batch, 128, 40, 40),    # P4
            torch.randn(batch, 256, 20, 20),    # P5
        ]

    def test_has_afpn(self):
        m = APMLFDetect(nc=6, ch=(32, 64, 128, 256))
        assert hasattr(m, 'afpn'), "APMLFDetect must have 'afpn' attribute"

    def test_has_mhsa(self):
        m = APMLFDetect(nc=6, ch=(32, 64, 128, 256))
        assert hasattr(m, 'mhsa'), "APMLFDetect must have 'mhsa' attribute"

    def test_four_detection_heads(self):
        """Must have 4 detection heads (cv2 and cv3 each have 4 entries)."""
        m = APMLFDetect(nc=6, ch=(32, 64, 128, 256))
        assert len(m.cv2) == 4, f"Expected 4 cv2 heads, got {len(m.cv2)}"
        assert len(m.cv3) == 4, f"Expected 4 cv3 heads, got {len(m.cv3)}"

    def test_forward_returns_output(self):
        """Forward pass must not crash and return something."""
        m = APMLFDetect(nc=6, ch=(32, 64, 128, 256))
        m.eval()
        feats = self._make_inputs()
        with torch.no_grad():
            out = m(feats)
        assert out is not None
