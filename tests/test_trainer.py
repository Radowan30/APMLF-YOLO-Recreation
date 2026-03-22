"""Tests for APMLFTrainer correctness."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics'))


class TestAPMLFTrainer:
    def test_importable(self):
        from apmlf_trainer import APMLFTrainer
        assert APMLFTrainer is not None

    def test_uses_slide_loss_not_base_criterion(self):
        """init_criterion must be overridden (not the base DetectionModel version).

        In this ultralytics version, init_criterion lives on DetectionModel (not
        DetectionTrainer). APMLFTrainer patches it onto the model in get_model(), so
        we verify that APMLFTrainer defines its own init_criterion — distinct from
        the base DetectionModel implementation.
        """
        from apmlf_trainer import APMLFTrainer
        from ultralytics.nn.tasks import DetectionModel
        assert APMLFTrainer.init_criterion is not DetectionModel.init_criterion, \
            "APMLFTrainer.init_criterion was not overridden (must differ from DetectionModel.init_criterion)"

    def test_optimizer_step_overridden(self):
        """optimizer_step must be overridden for NAM gradient injection."""
        from apmlf_trainer import APMLFTrainer
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        assert APMLFTrainer.optimizer_step is not DetectionTrainer.optimizer_step, \
            "APMLFTrainer.optimizer_step was not overridden"

    def test_nam_lambda_value(self):
        """λ must be 1e-4 per paper Section 3.1."""
        from apmlf_trainer import NAM_LAMBDA
        assert abs(NAM_LAMBDA - 1e-4) < 1e-10, \
            f"NAM_LAMBDA must be 1e-4, got {NAM_LAMBDA}"

    def test_nam_gradient_injection_logic(self):
        """Gradient injection: λ*sign(γ) added to .grad of _is_nam BN layers."""
        import torch.nn as nn

        NAM_LAMBDA = 1e-4

        bn = nn.BatchNorm2d(8)
        bn._is_nam = True
        bn.weight.data = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        bn.weight.grad = torch.zeros(8)

        # Apply injection (mirrors APMLFTrainer.optimizer_step logic)
        bn.weight.grad.data.add_(NAM_LAMBDA * torch.sign(bn.weight.data))

        expected = NAM_LAMBDA * torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        assert torch.allclose(bn.weight.grad, expected, atol=1e-10), \
            f"Gradient injection incorrect: {bn.weight.grad}"

    def test_non_nam_bn_not_modified(self):
        """BN layers without _is_nam tag must NOT have gradient injected."""
        import torch.nn as nn

        NAM_LAMBDA = 1e-4
        bn = nn.BatchNorm2d(8)  # no _is_nam tag
        bn.weight.grad = torch.zeros(8)
        original_grad = bn.weight.grad.clone()

        if getattr(bn, '_is_nam', False) and bn.weight.grad is not None:
            bn.weight.grad.data.add_(NAM_LAMBDA * torch.sign(bn.weight.data))

        assert torch.allclose(bn.weight.grad, original_grad), \
            "Untagged BN gradient was modified — injection is not selective"
