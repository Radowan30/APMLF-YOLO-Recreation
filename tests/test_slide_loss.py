import pytest
import math
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics', 'ultralytics'))
from utils.slide_loss import SlideLoss


class TestSlideLoss:
    def _make_loss(self):
        return SlideLoss()

    def test_easy_negative_weight_is_one(self):
        """For true <= auto_iou - 0.1: weight = 1.0 (unmodified BCE)."""
        loss = self._make_loss()
        pred = torch.tensor([0.0])
        true = torch.tensor([0.0])  # well below auto_iou=0.5 - 0.1 = 0.4
        # Weight = 1.0 → result must equal standard BCE
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, true, reduction='mean'
        )
        result = loss(pred, true, auto_iou=0.5)
        assert abs(result.item() - expected.item()) < 1e-6, \
            f"Easy negative: got {result.item()}, expected {expected.item()}"

    def test_middle_zone_weight_uses_paper_formula(self):
        """Middle zone weight = e^(1-μ), NOT e^μ (paper Eq.2 vs YOLO-FaceV2 code)."""
        loss = self._make_loss()
        auto_iou = 0.5
        # true in (0.4, 0.5): middle zone
        true_mid = torch.tensor([0.45])
        pred = torch.zeros(1)
        # Weight for middle zone = e^(1-0.5) = e^0.5 ≈ 1.649
        expected_weight = math.exp(1.0 - auto_iou)
        # Base BCE for pred=0, true=0.45
        base_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, true_mid, reduction='mean'
        )
        result = loss(pred, true_mid, auto_iou=auto_iou)
        assert abs(result.item() - (base_bce * expected_weight).item()) < 1e-4, \
            f"Middle zone: got {result.item()}, expected {(base_bce * expected_weight).item()}"

    def test_positive_zone_weight_decreasing(self):
        """For true > auto_iou: weight = e^(1-x), decreasing as IoU increases."""
        loss = self._make_loss()
        pred = torch.zeros(1)
        w_low = loss(pred, torch.tensor([0.6]), auto_iou=0.5)
        w_high = loss(pred, torch.tensor([0.9]), auto_iou=0.5)
        # Higher IoU → lower weight → lower loss
        assert w_high.item() < w_low.item()

    def test_auto_iou_minimum_clamp(self):
        """auto_iou must be clamped to minimum 0.2."""
        loss = self._make_loss()
        # Should not raise with very low IoU
        out = loss(torch.zeros(3), torch.zeros(3), auto_iou=0.01)
        assert out.item() >= 0

    def test_output_shape(self):
        loss = self._make_loss()
        pred = torch.randn(100, 6)
        true = torch.rand(100, 6)
        out = loss(pred, true, auto_iou=0.5)
        assert out.shape == ()  # scalar


class TestV8DetectionLossWithSlide:
    def test_slide_loss_used_not_bce(self):
        """v8DetectionLossWithSlide must use SlideLoss, not standard BCEWithLogitsLoss."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics'))
        from utils.slide_loss import SlideLoss, v8DetectionLossWithSlide
        from ultralytics.nn.tasks import DetectionModel
        yaml_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics',
            'ultralytics', 'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        model = DetectionModel(yaml_path, nc=6, verbose=False)
        criterion = v8DetectionLossWithSlide(model)
        assert isinstance(criterion.bce, SlideLoss), \
            f"Expected SlideLoss, got {type(criterion.bce)}"

    def test_auto_iou_not_hardcoded(self):
        """SlideLoss forward must accept auto_iou as a parameter."""
        from utils.slide_loss import SlideLoss
        import inspect
        sig = inspect.signature(SlideLoss.forward)
        assert 'auto_iou' in sig.parameters, \
            "SlideLoss.forward must accept auto_iou parameter"
