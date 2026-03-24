"""APMLFTrainer: Custom Ultralytics trainer for APMLF-YOLO.

Extends DetectionTrainer with:
1. v8DetectionLossWithSlide — Slide Loss for classification branch
2. NAM sparsity regularization — gradient injection on BN γ tagged _is_nam=True
   Formula: L_total = L_task + λ*Σ|γ|, applied via γ.grad += λ*sign(γ)
   λ = 1e-4 per paper Section 3.1

All paper training hyperparameters (SGD, lr=0.01, etc.) are set in train.py
via the 'args' override — NOT here. Trainer responsibilities: loss + optimizer step.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics'))

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.slide_loss import v8DetectionLossWithSlide


NAM_LAMBDA = 1e-4  # Sparsity regularization coefficient (paper Section 3.1)


def _slide_init_criterion(self):
    """Module-level named function so torch.save/pickle can serialize it.

    Used to patch DetectionModel.init_criterion on the model instance.
    Must be defined at module level (not as a lambda) to be picklable.
    """
    return v8DetectionLossWithSlide(self)


class APMLFTrainer(DetectionTrainer):
    """DetectionTrainer with Slide Loss and NAM sparsity regularization."""

    def init_criterion(self):
        """Return Slide Loss criterion bound to self.model.

        In this ultralytics version, init_criterion lives on DetectionModel, not
        DetectionTrainer.  APMLFTrainer defines it here so the class attribute
        exists for test introspection, and also patches it onto the model inside
        get_model() so the model's lazy criterion initialisation picks it up.
        """
        return v8DetectionLossWithSlide(self.model)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load APMLF-YOLO model and patch its init_criterion to use Slide Loss."""
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(
            cfg or 'ultralytics/cfg/models/v8/apmlf_yolo.yaml',
            nc=self.data['nc'],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        # Patch the model instance so model.loss() uses v8DetectionLossWithSlide.
        # Uses a module-level named function (not a lambda) so torch.save/pickle
        # can serialize the model without raising AttributeError on load.
        import types
        model.init_criterion = types.MethodType(_slide_init_criterion, model)
        return model

    def optimizer_step(self):
        """Standard optimizer step + NAM sparsity gradient injection.

        Injects λ·sign(γ) into gradient of every BN layer tagged _is_nam=True.
        This implements subgradient of L1 penalty Σ|γ| without modifying loss directly.
        Must run BEFORE super().optimizer_step() so gradients are clipped with injection.
        """
        for module in self.model.modules():
            if (
                isinstance(module, torch.nn.BatchNorm2d)
                and getattr(module, '_is_nam', False)
                and module.weight.grad is not None
            ):
                module.weight.grad.data.add_(
                    NAM_LAMBDA * torch.sign(module.weight.data)
                )
        super().optimizer_step()
