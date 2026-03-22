"""Tests that CAMLight and APMLFDetect are registered in parse_model."""
import pytest
import torch
import os


class TestModuleRegistration:
    def test_cam_light_importable_from_modules(self):
        from ultralytics.nn.modules import CAMLight
        assert CAMLight is not None

    def test_apmlf_detect_importable_from_modules(self):
        from ultralytics.nn.modules import APMLFDetect
        assert APMLFDetect is not None

    def test_yaml_parses_without_error(self):
        """parse_model must build the full model from YAML."""
        from ultralytics.nn.tasks import DetectionModel
        yaml_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics',
            'ultralytics', 'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        assert m is not None

    def test_model_has_cam_light_not_c2f(self):
        """Backbone must contain CAMLight, not C2f."""
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import CAMLight
        from ultralytics.nn.modules.block import C2f
        yaml_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics',
            'ultralytics', 'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        has_cam = any(isinstance(mod, CAMLight) for mod in m.modules())
        has_c2f = any(isinstance(mod, C2f) for mod in m.modules())
        assert has_cam, "CAMLight not found in model"
        assert not has_c2f, "C2f still present — CAMLight did not replace it"

    def test_model_forward_pass(self):
        """Full model forward pass at 640x640."""
        from ultralytics.nn.tasks import DetectionModel
        yaml_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..', 'ultralytics',
            'ultralytics', 'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        m.eval()
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 640, 640))
        assert out is not None
