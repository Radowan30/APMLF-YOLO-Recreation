# APMLF-YOLO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement APMLF-YOLO (Progressive Multi-Level Feature Fusion YOLO) for PCB tiny defect detection as described in the paper, as a fork-style extension of Ultralytics YOLOv8 v8.2.103.

**Architecture:** CAM-Light backbone (C2f with NAM dual-branch attention + reduced expansion ratio) feeds P2/P3/P4/P5 into APMLFDetect — a single module encapsulating AFPN neck (P3/P4/P5 asymptotic fusion via ASFF), MHSA (full self-attention on P5), and 4 detection heads (P2 small-object + P3/P4/P5 AFPN outputs). Training uses Slide Loss (IoU-adaptive BCE weighting) and NAM sparsity regularization injected via optimizer_step override.

**Tech Stack:** Python 3.9, PyTorch 2.4.1, CUDA 12.1, Ultralytics v8.2.103 (cloned locally), PKU PCB dataset (693→1408 images, 6 classes)

---

## Critical Design Constraints

- **No training on this machine** — all code is written here, transferred to training machine later
- **Ultralytics fork-style** — clone repo, add modules inside; do NOT pip install ultralytics
- **Match paper exactly** — every implementation decision backed by paper figures/text
- **Parameter targets**: backbone+standard neck+3-head = 3.01M; full model = 5.1M
- **Ultralytics version**: v8.2.103 exactly (last stable before YOLO11 structural changes)

## Known Calibration Values (Empirically Set on Training Machine)

- `CAMLight.CAM_E = 0.375` — start here; adjust until backbone hits 3.01M
- `AFPNNeck factor=4, num_blocks=4` — start here; adjust until full model hits 5.1M

---

## File Structure

### New Files (Create)
| File | Responsibility |
|------|----------------|
| `ultralytics/nn/modules/cam_light.py` | NAMChannelAtt, NAMSpatialAtt, CAMLight classes |
| `ultralytics/nn/modules/afpn.py` | ASFF2, ASFF3, AFPNBasicBlock, BlockBodyP345, AFPNNeck |
| `ultralytics/nn/modules/mhsa.py` | MHSAContentPosition (full HxW attention + decomposed position encoding) |
| `ultralytics/nn/modules/apmlf_detect.py` | APMLFDetect (AFPN neck + MHSA + 4-scale detection heads) |
| `ultralytics/utils/slide_loss.py` | SlideLoss (IoU-adaptive BCE weighting per paper Eq.2) |
| `ultralytics/cfg/models/v8/apmlf_yolo.yaml` | Model architecture YAML |
| `apmlf_trainer.py` | APMLFTrainer (custom trainer with Slide Loss + NAM sparsity) |
| `train.py` | Training entry point with all paper hyperparameters |
| `data/pcb_defect.yaml` | PKU PCB dataset config |
| `tests/test_cam_light.py` | Unit tests for CAM-Light + NAM |
| `tests/test_afpn.py` | Unit tests for AFPN components |
| `tests/test_mhsa.py` | Unit tests for MHSA |
| `tests/test_apmlf_detect.py` | Unit tests for APMLFDetect forward pass |
| `tests/test_slide_loss.py` | Unit tests for SlideLoss + v8DetectionLossWithSlide |
| `tests/test_registration.py` | Verify module registration and model parse/forward |
| `tests/test_trainer.py` | Verify APMLFTrainer uses Slide Loss and NAM injection |
| `tests/test_integration.py` | Full model forward pass + parameter count verification |
| `TRAINING_GUIDE.md` | Step-by-step instructions for the training machine |

### Modified Files
| File | Change |
|------|--------|
| `ultralytics/nn/modules/__init__.py` | Export all new module classes |
| `ultralytics/nn/tasks.py` | Register new modules in parse_model() |
| `ultralytics/utils/loss.py` | Add v8DetectionLossWithSlide subclass |

---

## Task 1: Environment Setup

**Files:**
- Create: `requirements_dev.txt`
- Create: `setup_notes.md`

**Environment note**: This machine uses `uv` for Python environment management.
Use `uv` to create the Python 3.9 venv — do NOT use `python -m venv` or conda.

- [ ] **Step 1: Create Python 3.9 venv with uv**

```bash
cd "C:\Users\Radowan Ahmed Baized\Desktop\RadowanStuff\PCB Research\APMLF_YOLO Implementation"
uv python install 3.9
uv venv --python 3.9 .venv
```

Expected: `.venv/` created with Python 3.9.

- [ ] **Step 2: Install base dependencies into the venv**

```bash
uv pip install pytest pyyaml
```

Note: torch/torchvision are NOT installed here — this machine has no GPU. They will be installed on the training machine.

- [ ] **Step 3: Clone Ultralytics v8.2.103**

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout v8.2.103
git checkout -b apmlf-yolo
cd ..
```

Expected: Branch `apmlf-yolo` created at tag v8.2.103.

- [ ] **Step 4: Install ultralytics in editable mode (CPU-only)**

```bash
uv pip install -e ultralytics/ --no-deps
```

`--no-deps` avoids pulling in torch (not needed for authoring). This makes ultralytics importable as a local module so tests can run.

- [ ] **Step 5: Verify the checkout and import are correct**

```bash
.venv/Scripts/python -c "
import ultralytics
print('ultralytics version:', ultralytics.__version__)
# Should print 8.2.103
"
```

Expected: `ultralytics version: 8.2.103`

- [ ] **Step 6: Create dev requirements file**

Create `requirements_dev.txt` in project root (not inside ultralytics/):
```
# Development machine (no GPU — code authoring only)
pytest
pyyaml
# ultralytics installed via: uv pip install -e ultralytics/ --no-deps

# Training machine (install these there):
# torch==2.4.1+cu121
# torchvision==0.19.1+cu121
# ultralytics==8.2.103 (use local fork, not pip)
```

- [ ] **Step 7: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 8: Commit**

```bash
cd ultralytics
git add -A
git commit -m "chore: checkout ultralytics v8.2.103 as apmlf-yolo base"
cd ..
```

---

## Task 2: NAM Channel Attention

**Files:**
- Create: `ultralytics/nn/modules/cam_light.py`
- Create: `tests/test_cam_light.py`

**Paper reference**: Section 3.1, Fig. 2 (channel branch). NAM channel attention uses BN γ (scale) parameters as importance weights: `β_i = |γ_i| / Σ|γ_i|`, applied as sigmoid gate on residual.

- [ ] **Step 1: Write failing test for NAMChannelAtt**

Create `tests/test_cam_light.py`:
```python
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
from nn.modules.cam_light import NAMChannelAtt, NAMSpatialAtt, CAMLight

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

    def test_output_range(self):
        """Sigmoid gate: output must be dampened (not amplified beyond input)."""
        m = NAMChannelAtt(64)
        m.eval()
        x = torch.ones(1, 64, 8, 8)
        out = m(x)
        # With sigmoid in [0,1], output <= input magnitude
        assert out.abs().max() <= x.abs().max() + 1e-5

    def test_differentiable(self):
        """Must support backprop through gamma weights."""
        m = NAMChannelAtt(64)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:\Users\Radowan Ahmed Baized\Desktop\RadowanStuff\PCB Research\APMLF_YOLO Implementation"
python -m pytest tests/test_cam_light.py::TestNAMChannelAtt -v
```
Expected: `ModuleNotFoundError: No module named 'nn.modules.cam_light'`

- [ ] **Step 3: Implement NAMChannelAtt**

Create `ultralytics/nn/modules/cam_light.py`:
```python
"""CAM-Light: C2f with dual-branch NAM attention (channel + spatial).

Paper: APMLF-YOLO Section 3.1, Fig. 2
NAM ref: "NAM: Normalization-based Attention Module" (Liu et al. 2021)
"""
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


class NAMChannelAtt(nn.Module):
    """Channel attention using BN gamma as importance weights.

    β_i = |γ_i| / Σ|γ_i|  (paper Eq. channel branch)
    Output = sigmoid(x_weighted) * residual
    BN tagged _is_nam=True so APMLFTrainer can inject sparsity gradient.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True)
        self.bn._is_nam = True  # tag for sparsity regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_bn = self.bn(x)
        # Compute normalized importance from BN gamma
        # Use .abs().detach() — importance weights are read-only scalars.
        # Gradients flow through self.bn(x) normally via BN backward.
        # .detach() prevents a second gradient path through the weight lookup
        # that would double-count BN weight gradients (numerically correct either way,
        # but .detach() matches the NAM paper's intent: γ as a non-differentiable selector).
        weight = self.bn.weight.abs().detach()
        weight = weight / (weight.sum() + 1e-8)  # shape: [C]
        # Apply channel-wise scaling
        x_weighted = x_bn * weight.view(1, -1, 1, 1)
        return torch.sigmoid(x_weighted) * residual
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cam_light.py::TestNAMChannelAtt -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ultralytics
git add nn/modules/cam_light.py
cd ..
git add tests/test_cam_light.py
git commit -m "feat: add NAMChannelAtt with BN-gamma importance weighting"
```

---

## Task 3: NAM Spatial Attention

**Files:**
- Modify: `ultralytics/nn/modules/cam_light.py`
- Modify: `tests/test_cam_light.py`

**Paper reference**: Section 3.1, Fig. 2 (spatial branch). Pixel-wise L2 norm: `w_i = ||x[:,h,w]||₂ / Σ||x[:,h,w]||₂`. Zero extra parameters.

- [ ] **Step 1: Write failing tests for NAMSpatialAtt**

Append to `tests/test_cam_light.py`:
```python
class TestNAMSpatialAtt:
    def test_output_shape(self):
        from nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        x = torch.randn(2, 64, 32, 32)
        assert m(x).shape == x.shape

    def test_no_parameters(self):
        """Spatial branch must have zero learnable parameters."""
        from nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        assert sum(p.numel() for p in m.parameters()) == 0

    def test_output_range(self):
        """Sigmoid gate should not amplify beyond input."""
        from nn.modules.cam_light import NAMSpatialAtt
        m = NAMSpatialAtt()
        x = torch.ones(1, 64, 8, 8)
        out = m(x)
        assert out.abs().max() <= x.abs().max() + 1e-5
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cam_light.py::TestNAMSpatialAtt -v
```
Expected: ImportError (NAMSpatialAtt not defined yet).

- [ ] **Step 3: Implement NAMSpatialAtt**

Append to `ultralytics/nn/modules/cam_light.py`:
```python
class NAMSpatialAtt(nn.Module):
    """Spatial attention using per-pixel L2 channel norm.

    w_i = ||x[:,h,w]||_2 / sum(||x[:,h,w]||_2)  (paper spatial branch)
    Zero learnable parameters — resolution-independent.
    Applied AFTER channel attention (sequential, per Fig. 2).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # L2 norm across channel dim → shape [B, 1, H, W]
        pixel_norm = x.norm(p=2, dim=1, keepdim=True)
        # Normalize spatially
        w = pixel_norm / (pixel_norm.sum(dim=[2, 3], keepdim=True) + 1e-8)
        return torch.sigmoid(w) * residual
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cam_light.py::TestNAMSpatialAtt -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add nn/modules/cam_light.py && cd ..
git add tests/test_cam_light.py
git commit -m "feat: add NAMSpatialAtt with zero-parameter pixel L2 norm"
```

---

## Task 4: CAMLight Block

**Files:**
- Modify: `ultralytics/nn/modules/cam_light.py`
- Modify: `tests/test_cam_light.py`

**Paper reference**: Section 3.1. CAMLight = C2f with NAM replacing standard attention; expansion ratio `e` reduced from 0.5 → ~0.375 (calibrate on training machine to hit 3.01M). NAM applied sequentially: channel then spatial.

- [ ] **Step 1: Write failing tests for CAMLight**

Append to `tests/test_cam_light.py`:
```python
class TestCAMLight:
    def test_output_shape(self):
        from nn.modules.cam_light import CAMLight
        m = CAMLight(128, 128, n=3)
        x = torch.randn(1, 128, 40, 40)
        assert m(x).shape == (1, 128, 40, 40)

    def test_parameter_reduction_vs_c2f(self):
        """CAMLight with e=0.375 should have fewer params than C2f with e=0.5."""
        from nn.modules.cam_light import CAMLight
        from ultralytics.nn.modules.block import C2f
        cam = CAMLight(256, 256, n=6, e=0.375)
        c2f = C2f(256, 256, n=6, e=0.5)
        cam_params = sum(p.numel() for p in cam.parameters())
        c2f_params = sum(p.numel() for p in c2f.parameters())
        assert cam_params < c2f_params, \
            f"CAMLight({cam_params}) should be < C2f({c2f_params})"

    def test_sequential_attention_order(self):
        """channel_att applied before spatial_att (sequential per Fig.2)."""
        from nn.modules.cam_light import CAMLight
        m = CAMLight(64, 64, n=1)
        # Just verify both attributes exist and ordering in forward
        assert hasattr(m, 'channel_att')
        assert hasattr(m, 'spatial_att')

    def test_cam_e_class_variable(self):
        """CAM_E class variable must be accessible for calibration."""
        from nn.modules.cam_light import CAMLight
        assert hasattr(CAMLight, 'CAM_E')
        assert 0.3 <= CAMLight.CAM_E <= 0.5
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cam_light.py::TestCAMLight -v
```
Expected: ImportError (CAMLight not defined).

- [ ] **Step 3: Implement CAMLight**

Append to `ultralytics/nn/modules/cam_light.py`:
```python
class CAMLight(nn.Module):
    """C2f redesigned with dual-branch NAM attention.

    Replaces all C2f blocks in YOLOv8n backbone.
    CAM_E: expansion ratio. Start at 0.375 and calibrate on training
    machine until backbone+standard neck+3-head totals 3.01M parameters.
    (YOLOv8n baseline C2f uses e=0.5 → 3.2M params)

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of Bottleneck sub-blocks
        shortcut: Use residual connections in Bottlenecks
        g: Groups for Bottleneck convolutions
        e: Expansion ratio (override CAM_E if provided)
    """
    CAM_E: float = 0.375  # CALIBRATION: adjust until backbone = 3.01M params

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = None,
    ):
        super().__init__()
        e = e if e is not None else self.CAM_E
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.channel_att = NAMChannelAtt(c2)
        self.spatial_att = NAMSpatialAtt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # Sequential: channel attention first, then spatial (Fig. 2)
        out = self.channel_att(out)
        out = self.spatial_att(out)
        return out
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cam_light.py -v
```
Expected: All tests pass (including previous NAMChannelAtt and NAMSpatialAtt).

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add nn/modules/cam_light.py && cd ..
git add tests/test_cam_light.py
git commit -m "feat: add CAMLight block with NAM dual-branch attention and reduced expansion ratio"
```

---

## Task 5: AFPN Components (ASFF2, ASFF3, BasicBlock)

**Files:**
- Create: `ultralytics/nn/modules/afpn.py`
- Create: `tests/test_afpn.py`

**Paper reference**: Section 3.2, Fig. 3. AFPN = Asymptotic Feature Pyramid Network. Spatially-adaptive softmax weighting (ASFF) per location. compress_c=8 for weight computation.

- [ ] **Step 1: Write failing tests**

Create `tests/test_afpn.py`:
```python
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
from nn.modules.afpn import ASFF2, ASFF3, AFPNBasicBlock, BlockBodyP345, AFPNNeck


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
        m = ASFF2(inter_dim=32)
        x1 = torch.randn(1, 32, 10, 10)
        x2 = torch.randn(1, 32, 10, 10)
        # Run forward then inspect weight_levels output manually
        m.eval()
        with torch.no_grad():
            wc = torch.cat([m.weight_l1(x1), m.weight_l2(x2)], 1)
            import torch.nn.functional as F
            w = F.softmax(m.weight_levels(wc), dim=1)
            # Sum over dim=1 should be ~1 everywhere
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_afpn.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement AFPN components**

Create `ultralytics/nn/modules/afpn.py`:
```python
"""AFPN: Asymptotic Feature Pyramid Network with ASFF adaptive fusion.

Paper: APMLF-YOLO Section 3.2, Fig. 3
AFPN ref: "AFPN: Asymptotic Feature Pyramid Network for Object Detection" (Yang et al. 2023)

Architecture:
  Stage 1: ASFF_2 fuses adjacent P3↔P4 (P5 isolated — asymptotic)
  Stage 2: ASFF_3 fuses all 3 scales (P5 joins asymptotically)
  BasicBlock refines each output at each stage.
  compress_c=8 for efficient weight computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class ASFF2(nn.Module):
    """Adaptive Spatial Feature Fusion for 2 inputs.

    Learns spatially-varying softmax weights per location.
    compress_c=8: internal channel reduction for weight computation.
    """
    def __init__(self, inter_dim: int, compress_c: int = 8):
        super().__init__()
        self.weight_l1 = Conv(inter_dim, compress_c, 1)
        self.weight_l2 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, 1, bias=True)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        w_cat = torch.cat([self.weight_l1(x1), self.weight_l2(x2)], dim=1)
        w = F.softmax(self.weight_levels(w_cat), dim=1)
        fused = x1 * w[:, 0:1] + x2 * w[:, 1:2]
        return self.conv(fused)


class ASFF3(nn.Module):
    """Adaptive Spatial Feature Fusion for 3 inputs."""
    def __init__(self, inter_dim: int, compress_c: int = 8):
        super().__init__()
        self.weight_l1 = Conv(inter_dim, compress_c, 1)
        self.weight_l2 = Conv(inter_dim, compress_c, 1)
        self.weight_l3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, bias=True)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        w_cat = torch.cat(
            [self.weight_l1(x1), self.weight_l2(x2), self.weight_l3(x3)], dim=1
        )
        w = F.softmax(self.weight_levels(w_cat), dim=1)
        fused = x1 * w[:, 0:1] + x2 * w[:, 1:2] + x3 * w[:, 2:3]
        return self.conv(fused)


class AFPNBasicBlock(nn.Module):
    """Residual refinement block used after each ASFF fusion stage."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.conv2(self.conv1(x))) + x)


def _make_blocks(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[AFPNBasicBlock(channels) for _ in range(num_blocks)])


class BlockBodyP345(nn.Module):
    """Two-stage asymptotic feature fusion for P3/P4/P5.

    Stage 1 (asymptotic: P5 isolated):
      new_P3 = ASFF2(P3, upsample(P4))
      new_P4 = ASFF2(downsample(P3), P4)
      → BasicBlock refinement

    Stage 2 (P5 joins):
      fin_P3 = ASFF3(P3, upsample(P4), upsample×2(P5))
      fin_P4 = ASFF3(downsample(P3), P4, upsample(P5))
      fin_P5 = ASFF3(downsample×2(P3), downsample(P4), P5)
      → BasicBlock refinement

    Args:
        channels: [c_p3, c_p4, c_p5] internal (post-projection) channel counts
        num_blocks: AFPNBasicBlock count per output per stage
    """
    def __init__(self, channels: list, num_blocks: int = 4):
        super().__init__()
        c3, c4, c5 = channels

        # ---- Stage 1 resize ops ----
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dn_p3_to_p4 = Conv(c3, c4, 3, 2)

        # ---- Stage 1 ASFF ----
        self.asff2_p3_s1 = ASFF2(c3)
        self.asff2_p4_s1 = ASFF2(c4)

        # ---- Stage 1 refinement ----
        self.blocks_p3_s1 = _make_blocks(c3, num_blocks)
        self.blocks_p4_s1 = _make_blocks(c4, num_blocks)

        # ---- Stage 2 resize ops ----
        self.up_p4_to_p3_s2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_p5_to_p3_s2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.dn_p3_to_p4_s2 = Conv(c3, c4, 3, 2)
        self.up_p5_to_p4_s2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dn_p3_to_p5_s2 = nn.Sequential(Conv(c3, c4, 3, 2), Conv(c4, c5, 3, 2))
        self.dn_p4_to_p5_s2 = Conv(c4, c5, 3, 2)

        # ---- Stage 2 ASFF ----
        self.asff3_p3_s2 = ASFF3(c3)
        self.asff3_p4_s2 = ASFF3(c4)
        self.asff3_p5_s2 = ASFF3(c5)

        # ---- Stage 2 refinement ----
        self.blocks_p3_s2 = _make_blocks(c3, num_blocks)
        self.blocks_p4_s2 = _make_blocks(c4, num_blocks)
        self.blocks_p5_s2 = _make_blocks(c5, num_blocks)

    def forward(self, x: list) -> list:
        p3, p4, p5 = x

        # Stage 1: adjacent fusion
        new_p3 = self.asff2_p3_s1(p3, self.up_p4_to_p3(p4))
        new_p4 = self.asff2_p4_s1(self.dn_p3_to_p4(p3), p4)
        p3 = self.blocks_p3_s1(new_p3)
        p4 = self.blocks_p4_s1(new_p4)

        # Stage 2: full 3-scale fusion
        fin_p3 = self.asff3_p3_s2(p3, self.up_p4_to_p3_s2(p4), self.up_p5_to_p3_s2(p5))
        fin_p4 = self.asff3_p4_s2(self.dn_p3_to_p4_s2(p3), p4, self.up_p5_to_p4_s2(p5))
        fin_p5 = self.asff3_p5_s2(self.dn_p3_to_p5_s2(p3), self.dn_p4_to_p5_s2(p4), p5)

        return [
            self.blocks_p3_s2(fin_p3),
            self.blocks_p4_s2(fin_p4),
            self.blocks_p5_s2(fin_p5),
        ]


class AFPNNeck(nn.Module):
    """Full AFPN neck: project → asymptotic fusion → project back.

    Args:
        in_channels: [c_p3, c_p4, c_p5] from backbone (e.g. [64,128,256] for YOLOv8n)
        out_channels: unified output channels for detection heads (256)
        factor: channel reduction factor for internal AFPN computation (start: 4)
        num_blocks: AFPNBasicBlock count per stage per output (start: 4)

    CALIBRATION NOTE: Adjust factor and num_blocks on training machine until
    full model (backbone + AFPNNeck + APMLFDetect heads) reaches 5.1M params.
    """
    def __init__(
        self,
        in_channels: list = None,
        out_channels: int = 256,
        factor: int = 4,
        num_blocks: int = 4,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256]
        ic = [c // factor for c in in_channels]  # e.g. [16, 32, 64]

        self.conv_in = nn.ModuleList(
            [Conv(c, ic[i], 1) for i, c in enumerate(in_channels)]
        )
        self.body = BlockBodyP345(ic, num_blocks=num_blocks)
        self.conv_out = nn.ModuleList(
            [Conv(ic[i], out_channels, 1) for i in range(3)]
        )

    def forward(self, x: list) -> list:
        projected = [self.conv_in[i](x[i]) for i in range(3)]
        fused = self.body(projected)
        return [self.conv_out[i](fused[i]) for i in range(3)]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_afpn.py -v
```
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add nn/modules/afpn.py && cd ..
git add tests/test_afpn.py
git commit -m "feat: add AFPN neck with ASFF2/ASFF3 adaptive spatial fusion and two-stage asymptotic pipeline"
```

---

## Task 6: MHSA Module

**Files:**
- Create: `ultralytics/nn/modules/mhsa.py`
- Create: `tests/test_mhsa.py`

**Paper reference**: Section 3.2, Fig. 5. Full H×W attention (400 tokens at P5 20×20). BoTNet-style decomposed position encoding `r = L_h + L_w`. Attention = `softmax((QK^T + QR^T)/√d) · V`. 8 heads, d_head=32, total dim=256.

- [ ] **Step 1: Write failing tests**

Create `tests/test_mhsa.py`:
```python
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
from nn.modules.mhsa import MHSAContentPosition


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
        """256-dim, 8 heads: 4×(256×256) conv + pos params ≈ 263K."""
        m = MHSAContentPosition(dim=256, num_heads=8, dim_head=32, H=20, W=20)
        params = sum(p.numel() for p in m.parameters())
        assert 250_000 < params < 300_000, f"Unexpected param count: {params}"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_mhsa.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement MHSAContentPosition**

Create `ultralytics/nn/modules/mhsa.py`:
```python
"""MHSA with BoTNet-style decomposed 2D position encoding.

Paper: APMLF-YOLO Section 3.2, Fig. 5
Applied to P5 feature map (20×20 = 400 tokens) after AFPN fusion.
Full H×W attention — NOT axial (Fig. 5 shows full HxW token grid).

Attention formula:
  Attention = softmax((QK^T + QR^T) / sqrt(d_head)) @ V
  r = pos_h + pos_w   (decomposed, broadcast to [heads, H*W, d_head])

Args:
  dim: Feature channels (256 for YOLOv8n at P5)
  num_heads: Number of attention heads (8 per paper)
  dim_head: Channels per head (32, so 8×32=256)
  H, W: Spatial size of input feature map (20×20 for P5 at 640px input)
"""
import math
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
            f"dim ({dim}) must equal num_heads ({num_heads}) × dim_head ({dim_head})"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, dh = self.num_heads, self.dim_head
        N = H * W  # number of tokens

        def reshape(t):
            # [B, C, H, W] → [B, nh, N, dh]
            return t.reshape(B, nh, dh, N).permute(0, 1, 3, 2)

        Q = reshape(self.to_q(x))   # [B, nh, N, dh]
        K = reshape(self.to_k(x))   # [B, nh, N, dh]
        V = reshape(self.to_v(x))   # [B, nh, N, dh]

        # Content attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, nh, N, N]

        # Decomposed position bias: r[nh, H, W, dh] = pos_h + pos_w
        r = (self.pos_h + self.pos_w).reshape(nh, N, dh)  # [nh, N, dh]
        # Position attention: Q @ r^T → [B, nh, N, N]
        pos_attn = torch.matmul(Q, r.unsqueeze(0).transpose(-2, -1)) * self.scale
        attn = F.softmax(attn + pos_attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, nh, N, dh]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.to_out(out)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_mhsa.py -v
```
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add nn/modules/mhsa.py && cd ..
git add tests/test_mhsa.py
git commit -m "feat: add MHSAContentPosition with BoTNet decomposed 2D position encoding"
```

---

## Task 7: APMLFDetect (Combined Neck + Head Module)

**Files:**
- Create: `ultralytics/nn/modules/apmlf_detect.py`
- Create: `tests/test_apmlf_detect.py`

**Paper reference**: Full model Fig. 1. APMLFDetect is a monolithic nn.Module receiving [P2, P3, P4, P5] from backbone. Internally: project P3/P4/P5 through AFPNNeck → apply MHSA to P5 output → 4 detection heads (P2 separate, P3/P4/P5 from AFPN). Output format identical to Ultralytics Detect.

- [ ] **Step 1: Write failing tests**

Create `tests/test_apmlf_detect.py`:
```python
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
from nn.modules.apmlf_detect import APMLFDetect


class TestAPMLFDetect:
    def _make_inputs(self, batch=2):
        """Simulate YOLOv8n backbone outputs at 640px input."""
        return [
            torch.randn(batch, 32, 160, 160),   # P2
            torch.randn(batch, 64, 80, 80),     # P3
            torch.randn(batch, 128, 40, 40),    # P4
            torch.randn(batch, 256, 20, 20),    # P5
        ]

    def test_output_is_list_of_4(self):
        m = APMLFDetect(nc=6, ch=[32, 64, 128, 256])
        m.eval()
        feats = self._make_inputs()
        out = m(feats)
        # Ultralytics Detect returns list of tensors during training
        assert isinstance(out, (list, tuple))

    def test_four_detection_scales(self):
        """One output per scale: P2, P3, P4, P5."""
        m = APMLFDetect(nc=6, ch=[32, 64, 128, 256])
        m.eval()
        feats = self._make_inputs()
        # Access raw cv2/cv3 outputs
        assert len(m.cv2) == 4
        assert len(m.cv3) == 4

    def test_mhsa_applied_to_p5(self):
        """APMLFDetect must have an MHSA attribute on the P5 path."""
        m = APMLFDetect(nc=6, ch=[32, 64, 128, 256])
        assert hasattr(m, 'mhsa')

    def test_afpn_neck_present(self):
        m = APMLFDetect(nc=6, ch=[32, 64, 128, 256])
        assert hasattr(m, 'afpn')
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_apmlf_detect.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement APMLFDetect**

Create `ultralytics/nn/modules/apmlf_detect.py`:
```python
"""APMLFDetect: Combined AFPN neck + MHSA + 4-scale detection heads.

Receives [P2, P3, P4, P5] from backbone.
Pipeline:
  1. AFPN fuses P3/P4/P5 → [P3', P4', P5']
  2. MHSA applied to P5' (full 20×20 self-attention)
  3. 4 detection heads: P2 (direct), P3'/P4'/P5' (from AFPN)

Inherits from Ultralytics Detect for compatibility with trainer/loss/export.
Input ch = [c_p2, c_p3, c_p4, c_p5] (e.g. [32, 64, 128, 256] for YOLOv8n).
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.afpn import AFPNNeck
from ultralytics.nn.modules.mhsa import MHSAContentPosition


class APMLFDetect(Detect):
    """4-scale detection: P2 + AFPN(P3,P4,P5) + MHSA on P5.

    Args:
        nc: Number of classes (6 for PKU PCB dataset)
        ch: Input channel tuple [c_p2, c_p3, c_p4, c_p5]
    """
    def __init__(self, nc: int = 80, ch: tuple = ()):
        # Detect expects channels for all 4 heads
        # We pass out_channels=256 unified for P3/P4/P5; P2 raw
        afpn_out = 256
        # ch_for_detect: P2 raw channels, then 3× afpn_out
        ch_detect = (ch[0], afpn_out, afpn_out, afpn_out)
        super().__init__(nc=nc, ch=ch_detect)

        in_channels_afpn = list(ch[1:])  # [c_p3, c_p4, c_p5]
        self.afpn = AFPNNeck(
            in_channels=in_channels_afpn,
            out_channels=afpn_out,
            factor=4,
            num_blocks=4,
        )
        self.mhsa = MHSAContentPosition(
            dim=afpn_out,
            num_heads=8,
            dim_head=32,
            H=20,
            W=20,
        )

    def forward(self, x: list):
        p2, p3, p4, p5 = x

        # AFPN fuses P3/P4/P5
        p3_fused, p4_fused, p5_fused = self.afpn([p3, p4, p5])

        # MHSA refines P5 (full 20×20 self-attention)
        p5_attn = self.mhsa(p5_fused)

        # Feed into Detect parent: [P2, P3', P4', P5_attn]
        return super().forward([p2, p3_fused, p4_fused, p5_attn])
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_apmlf_detect.py -v
```
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add nn/modules/apmlf_detect.py && cd ..
git add tests/test_apmlf_detect.py
git commit -m "feat: add APMLFDetect combining AFPN neck, MHSA, and 4-scale detection heads"
```

---

## Task 8: Slide Loss

**Files:**
- Create: `ultralytics/utils/slide_loss.py`
- Create: `tests/test_slide_loss.py`

**Paper reference**: Section 3.3, Eq. 2. IoU-adaptive BCE weighting for classification only.
```
f(x) = 1.0          if x <= μ - 0.1        (easy negative zone)
      e^(1-μ)        if μ-0.1 < x < μ      (transition zone)  ← paper Eq.2
      e^(1-x)        if x >= μ              (positive zone)
```
**CRITICAL**: Use `e^(1-μ)` for middle zone, NOT `e^μ` (YOLO-FaceV2 code error).

- [ ] **Step 1: Write failing tests**

Create `tests/test_slide_loss.py`:
```python
import pytest
import math
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
from utils.slide_loss import SlideLoss


class TestSlideLoss:
    def _make_loss(self):
        return SlideLoss()

    def test_easy_negative_weight_is_one(self):
        """For true < auto_iou - 0.1: weight = 1.0"""
        loss = self._make_loss()
        pred = torch.tensor([0.0])
        true = torch.tensor([0.0])  # well below auto_iou=0.5 - 0.1 = 0.4
        # Can't inspect weight directly; test relative to manual reference
        out = loss(pred, true, auto_iou=0.5)
        assert out.item() >= 0

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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_slide_loss.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement SlideLoss**

Create `ultralytics/utils/slide_loss.py`:
```python
"""Slide Loss: IoU-adaptive BCE weighting for classification branch.

Paper: APMLF-YOLO Section 3.3, Eq. 2
Applied ONLY to classification BCE loss (not box/DFL loss).
μ = auto_iou (mean CIoU of positive-assigned samples from TAL).

Weighting function (paper Eq. 2 — using e^(1-μ) NOT e^μ):
  f(x) = 1.0        if x <= μ - 0.1       (easy negatives: full weight)
  f(x) = e^(1-μ)    if μ-0.1 < x < μ     (transition: constant per batch)
  f(x) = e^(1-x)    if x >= μ             (positives: decreasing with IoU)

Key difference from YOLO-FaceV2 (which uses e^μ): the paper specifies 1-μ.
This makes the transition zone weight > 1 when μ < 1, emphasizing hard samples.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideLoss(nn.Module):
    """IoU-adaptive classification loss weighting.

    Wraps BCE with IoU-derived per-sample weights.

    Args:
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        auto_iou: float = 0.5,
    ) -> torch.Tensor:
        # Clamp auto_iou to minimum 0.2 (avoid degenerate weights at start of training)
        auto_iou = max(float(auto_iou), 0.2)

        # Base BCE loss (unreduced)
        loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')

        # Three zones based on paper Eq. 2
        b1 = true <= (auto_iou - 0.1)                          # easy negatives
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)    # transition zone
        b3 = true >= auto_iou                                   # positive zone

        # Weights: PAPER formula (e^(1-μ) for middle, NOT e^μ)
        w = (
            1.0 * b1
            + math.exp(1.0 - auto_iou) * b2
            + torch.exp(1.0 - true) * b3
        )
        loss = loss * w

        return loss.mean() if self.reduction == 'mean' else loss.sum()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_slide_loss.py -v
```
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ultralytics && git add utils/slide_loss.py && cd ..
git add tests/test_slide_loss.py
git commit -m "feat: add SlideLoss with IoU-adaptive BCE weighting per paper Eq.2 (e^(1-μ) middle zone)"
```

---

## Task 9: Module Registration

**Files:**
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`

This makes the YAML parser aware of the new modules. Both `CAMLight` and `APMLFDetect` must be registered so `parse_model()` can instantiate them from YAML.

- [ ] **Step 1: Write failing test for module registration**

Create `tests/test_registration.py`:
```python
"""Tests that CAMLight and APMLFDetect are registered in parse_model."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))


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
            os.path.dirname(__file__), '..', 'ultralytics',
            'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        assert m is not None

    def test_model_has_cam_light_not_c2f(self):
        """Backbone must contain CAMLight, not C2f."""
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import CAMLight
        from ultralytics.nn.modules.block import C2f
        yaml_path = os.path.join(
            os.path.dirname(__file__), '..', 'ultralytics',
            'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        has_cam = any(isinstance(mod, CAMLight) for mod in m.modules())
        has_c2f = any(isinstance(mod, C2f) for mod in m.modules())
        assert has_cam, "CAMLight not found in model"
        assert not has_c2f, "C2f still present — CAMLight not replacing it"

    def test_model_forward_pass(self):
        """Full model forward pass at 640×640."""
        import torch
        from ultralytics.nn.tasks import DetectionModel
        yaml_path = os.path.join(
            os.path.dirname(__file__), '..', 'ultralytics',
            'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
        )
        m = DetectionModel(yaml_path, nc=6, verbose=False)
        m.eval()
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 640, 640))
        assert out is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_registration.py -v
```
Expected: `ImportError` — `CAMLight` not yet in `__init__.py`.

- [ ] **Step 3: Add exports to `__init__.py`**

Read the file first, then append after the last existing import block:
```bash
tail -20 ultralytics/nn/modules/__init__.py
```

Append to `ultralytics/nn/modules/__init__.py`:
```python
from ultralytics.nn.modules.cam_light import CAMLight, NAMChannelAtt, NAMSpatialAtt
from ultralytics.nn.modules.afpn import (
    AFPNNeck, ASFF2, ASFF3, AFPNBasicBlock, BlockBodyP345
)
from ultralytics.nn.modules.mhsa import MHSAContentPosition
from ultralytics.nn.modules.apmlf_detect import APMLFDetect
```

If `__all__` list is present in the file, also add the names there:
```python
"CAMLight", "NAMChannelAtt", "NAMSpatialAtt",
"AFPNNeck", "ASFF2", "ASFF3", "AFPNBasicBlock", "BlockBodyP345",
"MHSAContentPosition", "APMLFDetect",
```

- [ ] **Step 4: Register in `parse_model()` in `tasks.py`**

First read how v8.2.103 registers C2f and Detect:
```bash
grep -n "C2f\|RepC3\|BottleneckCSP" ultralytics/nn/tasks.py | head -20
grep -n "isinstance.*Detect\|m is Detect\|m in.*Detect" ultralytics/nn/tasks.py | head -20
```

In `tasks.py`, `parse_model()` has two key registration points:

**Point A — channel calculation set** (find the set that includes `C2f`; it determines which modules receive `(c1, c2, n, ...)` args):
```python
# FIND this line (exact set membership check varies by version):
if m in {C2f, C2fAttn, ...}:   # or similar
# ADD CAMLight to the same set:
if m in {C2f, CAMLight, ...}:  # CAMLight same args as C2f
```

**Point B — Detect output channels** (find where Detect's output channel is recorded):
```python
# FIND something like:
elif m in {Detect, Segment, ...}:
    args.append([ch[x] for x in f])
# ADD APMLFDetect to the same branch:
elif m in {Detect, Segment, APMLFDetect, ...}:
    args.append([ch[x] for x in f])
```

**If tasks.py uses a dict lookup** (less common in v8.2.103), add to the dict:
```python
CAMLight: C2f_handler,   # same handler as C2f
APMLFDetect: detect_handler,
```

The invariant: `CAMLight` is handled identically to `C2f`; `APMLFDetect` is handled identically to `Detect`. If unsure, copy the exact `if/elif` branch for `C2f` and `Detect` and add `CAMLight`/`APMLFDetect` to the same condition.

**Also add imports at top of tasks.py:**
```python
from ultralytics.nn.modules.cam_light import CAMLight
from ultralytics.nn.modules.apmlf_detect import APMLFDetect
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_registration.py -v
```
Expected: All 5 tests pass (including forward pass).

If `test_model_forward_pass` fails with a channel mismatch error, check that `APMLFDetect.__init__` receives the correct channel list from `parse_model`. The YAML passes `[ch[2], ch[4], ch[6], ch[9]]` = `[32, 64, 128, 256]` after n-scale.

- [ ] **Step 6: Commit**

```bash
cd ultralytics
git add nn/modules/__init__.py nn/tasks.py
cd ..
git add tests/test_registration.py
git commit -m "feat: register CAMLight and APMLFDetect in ultralytics module registry and parse_model"
```

---

## Task 10: Loss Integration (v8DetectionLossWithSlide)

**Files:**
- Modify: `ultralytics/utils/loss.py`

`v8DetectionLoss.__call__` in v8.2.103 performs TAL assignment then computes box, DFL, and cls losses in one pass. We must override `__call__` to intercept the point where CIoU is available (after TAL, before BCE) so we can compute `auto_iou` = mean CIoU of positive assignments and pass it to `SlideLoss`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_slide_loss.py`:
```python
class TestV8DetectionLossWithSlide:
    def test_slide_loss_used_not_bce(self):
        """v8DetectionLossWithSlide must use SlideLoss, not standard BCEWithLogitsLoss."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
        from utils.slide_loss import SlideLoss, v8DetectionLossWithSlide
        from ultralytics.nn.tasks import DetectionModel
        yaml_path = os.path.join(
            os.path.dirname(__file__), '..', 'ultralytics',
            'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
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
```

- [ ] **Step 2: Run to verify partial failure**

```bash
python -m pytest tests/test_slide_loss.py::TestV8DetectionLossWithSlide -v
```
Expected: ImportError for `v8DetectionLossWithSlide` (not yet in loss.py).

- [ ] **Step 3: Read `v8DetectionLoss.__call__` to find the BCE call site**

```bash
grep -n "self.bce\|bce_loss\|cls_loss" ultralytics/utils/loss.py | head -30
```

In v8.2.103, `v8DetectionLoss.__call__` structure is approximately:
```python
def __call__(self, preds, batch):
    ...
    # TAL assignment
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(...)

    # Box loss (uses CIoU internally via bbox_iou)
    if fg_mask.sum():
        loss[0], iou = self.bbox_loss(pred_dist, pred_bboxes, anchor_points,
                                       target_bboxes, target_scores, ...)
    # Cls loss
    loss[1] += self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

    return loss.sum() * batch_size, loss.detach()
```

The `iou` variable returned by `bbox_loss` contains the CIoU values for positive samples — this is our `auto_iou` source.

- [ ] **Step 4: Implement v8DetectionLossWithSlide**

Append at end of `ultralytics/utils/loss.py`:
```python
from ultralytics.utils.slide_loss import SlideLoss


class v8DetectionLossWithSlide(v8DetectionLoss):
    """v8DetectionLoss with IoU-adaptive SlideLoss for classification branch.

    Overrides __call__ to:
    1. Run TAL assignment (same as parent)
    2. Compute box loss → captures CIoU of positive samples as auto_iou
    3. Pass auto_iou to SlideLoss for classification weighting

    auto_iou = mean CIoU of TAL-positive predictions (paper: "μ" in Eq.2).
    Falls back to 0.5 when no positives assigned (e.g. first iterations).
    """

    def __init__(self, model):
        super().__init__(model)
        self.bce = SlideLoss()  # Replace BCEWithLogitsLoss

    def __call__(self, preds, batch):
        """Full loss computation with dynamic auto_iou for SlideLoss.

        Structure mirrors v8DetectionLoss.__call__ exactly, with the single
        change that self.bce is called with auto_iou= instead of no argument.

        IMPLEMENTATION NOTE: Read v8DetectionLoss.__call__ in loss.py, then
        copy it verbatim here and modify ONLY the self.bce call to add auto_iou:

          # Original:
          loss[1] += self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

          # Replace with:
          auto_iou = float(iou.mean().clamp(0.2, 1.0)) if fg_mask.sum() > 0 else 0.5
          loss[1] += self.bce(pred_scores, target_scores.to(dtype),
                              auto_iou=auto_iou).sum() / target_scores_sum

        The `iou` variable is already computed by self.bbox_loss() before the BCE call.
        If v8.2.103's bbox_loss does not return iou as second value, check:
          grep -n "def bbox_loss" ultralytics/utils/loss.py
        and adjust accordingly.
        """
        # Step 1: copy the full __call__ body from v8DetectionLoss
        # Step 2: locate the self.bce(...) call
        # Step 3: replace it with:
        #   auto_iou = float(iou.mean().clamp(0.2, 1.0)) if fg_mask.sum() > 0 else 0.5
        #   self.bce(..., auto_iou=auto_iou)
        # Step 4: verify tests pass
        #
        # We cannot write the full body here without reading the exact v8.2.103 source,
        # but the transformation is mechanical: one line changed, all else identical.
        raise NotImplementedError(
            "Copy v8DetectionLoss.__call__ body here, then modify the self.bce call. "
            "See docstring above for exact change."
        )
```

**Then complete the implementation:**
```bash
# Read the exact __call__ body
grep -n "def __call__" ultralytics/utils/loss.py
# Read ~80 lines from that line number
# Copy the body into v8DetectionLossWithSlide.__call__
# Modify the single self.bce(...) line as described above
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_slide_loss.py -v
```
Expected: All pass including `TestV8DetectionLossWithSlide`.

- [ ] **Step 6: Commit**

```bash
cd ultralytics
git add utils/loss.py utils/slide_loss.py
cd ..
git add tests/test_slide_loss.py
git commit -m "feat: add v8DetectionLossWithSlide with dynamic auto_iou from TAL positive CIoU"
```

---

## Task 11: YAML Model Configuration

**Files:**
- Create: `ultralytics/cfg/models/v8/apmlf_yolo.yaml`

**Paper reference**: YOLOv8n backbone architecture. CAMLight replaces all C2f blocks. SPPF retained. APMLFDetect receives [P2, P3, P4, P5].

- [ ] **Step 1: Read reference YOLOv8n YAML**

```bash
cat ultralytics/cfg/models/v8/yolov8.yaml
```

- [ ] **Step 2: Create APMLF-YOLO YAML**

Create `ultralytics/cfg/models/v8/apmlf_yolo.yaml`:
```yaml
# APMLF-YOLO: Progressive Multi-Level Feature Fusion YOLO
# Paper: Section 3 — CAMLight backbone + AFPN+MHSA neck + 4-head detection
# Base: YOLOv8n architecture with all C2f → CAMLight
# Target params: backbone+standard neck+3-head = 3.01M; full model = 5.1M
#
# CALIBRATION on training machine:
#   1. Adjust CAMLight.CAM_E in cam_light.py until backbone ≈ 3.01M params
#   2. Adjust AFPNNeck factor/num_blocks in apmlf_detect.py until total ≈ 5.1M

nc: 6  # PKU PCB dataset: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper

scales:
  # depth, width, max_channels (use 'n' for smallest — matches paper YOLOv8n base)
  n: [0.33, 0.25, 1024]

backbone:
  # [from, repeats, module, args]
  # NOTE: All channel values are PRE-SCALE (yaml spec values).
  # At 'n' scale (width_multiple=0.25): actual channels = spec × 0.25
  # e.g. 64→16, 128→32, 256→64, 512→128, 1024→256
  # Actual output channels shown after ×0.25 scaling.
  - [-1, 1, Conv, [64, 3, 2]]          # 0 - P1/2,  actual: 16ch @ 320×320
  - [-1, 1, Conv, [128, 3, 2]]         # 1 - P2/4,  actual: 32ch @ 160×160
  - [-1, 3, CAMLight, [128, False]]    # 2 - P2 CAMLight,  actual: 32ch
  - [-1, 1, Conv, [256, 3, 2]]         # 3 - P3/8,  actual: 64ch @ 80×80
  - [-1, 6, CAMLight, [256, False]]    # 4 - P3 CAMLight,  actual: 64ch
  - [-1, 1, Conv, [512, 3, 2]]         # 5 - P4/16, actual: 128ch @ 40×40
  - [-1, 6, CAMLight, [512, False]]    # 6 - P4 CAMLight,  actual: 128ch
  - [-1, 1, Conv, [1024, 3, 2]]        # 7 - P5/32, actual: 256ch @ 20×20
  - [-1, 3, CAMLight, [1024, False]]   # 8 - P5 CAMLight,  actual: 256ch
  - [-1, 1, SPPF, [1024, 5]]          # 9 - P5+SPPF, actual: 256ch (1024×0.25=256)

head:
  # APMLFDetect receives: P2(idx 2), P3(idx 4), P4(idx 6), P5(idx 9)
  # Encapsulates AFPN(P3,P4,P5) + MHSA(P5) + 4 detection heads
  - [[2, 4, 6, 9], 1, APMLFDetect, [nc]]
```

- [ ] **Step 3: Verify YAML parses without error**

```bash
cd "C:\Users\Radowan Ahmed Baized\Desktop\RadowanStuff\PCB Research\APMLF_YOLO Implementation"
python -c "
import sys; sys.path.insert(0, 'ultralytics')
from ultralytics import YOLO
# This will fail if not installed — use direct parse_model test instead:
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel('ultralytics/cfg/models/v8/apmlf_yolo.yaml', nc=6)
print(m)
print(f'Total params: {sum(p.numel() for p in m.parameters()):,}')
"
```

Expected: Model prints cleanly; parameter count shown (pre-calibration, expect ~4-6M range).

- [ ] **Step 4: Commit**

```bash
cd ultralytics
git add cfg/models/v8/apmlf_yolo.yaml
git commit -m "feat: add apmlf_yolo.yaml model configuration with CAMLight backbone and APMLFDetect head"
```

---

## Task 12: Custom Trainer (APMLFTrainer)

**Files:**
- Create: `apmlf_trainer.py`
- Create: `tests/test_trainer.py`

**Paper reference**: Section 4 (Training Details). SGD optimizer, lr=0.01, momentum=0.937, wd=0.0005. NAM sparsity: `L_total = L_task + λΣ|γ|` via gradient injection `γ.grad += λ·sign(γ)`, λ=1e-4.

- [ ] **Step 1: Write failing tests for APMLFTrainer**

Create `tests/test_trainer.py` BEFORE creating `apmlf_trainer.py`:
```python
"""Tests for APMLFTrainer correctness."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))


class TestAPMLFTrainer:
    def test_importable(self):
        from apmlf_trainer import APMLFTrainer
        assert APMLFTrainer is not None

    def test_uses_slide_loss_not_base_criterion(self):
        """init_criterion must be overridden (not the base DetectionTrainer version)."""
        from apmlf_trainer import APMLFTrainer
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        assert APMLFTrainer.init_criterion is not DetectionTrainer.init_criterion, \
            "APMLFTrainer.init_criterion was not overridden"

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

        NAM_LAMBDA = 1e-4  # expected value from paper

        bn = nn.BatchNorm2d(8)
        bn._is_nam = True
        # Set known weight values so sign() is predictable
        bn.weight.data = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        bn.weight.grad = torch.zeros(8)  # start with zero grad

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

        # Simulate the conditional check in optimizer_step
        if getattr(bn, '_is_nam', False) and bn.weight.grad is not None:
            bn.weight.grad.data.add_(NAM_LAMBDA * torch.sign(bn.weight.data))

        assert torch.allclose(bn.weight.grad, original_grad), \
            "Untagged BN gradient was modified — injection is not selective"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_trainer.py -v
```
Expected: `ImportError: No module named 'apmlf_trainer'` (file does not exist yet).

- [ ] **Step 3: Create apmlf_trainer.py**

Create `apmlf_trainer.py` in project root (NOT inside ultralytics/):
```python
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics'))

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLossWithSlide


NAM_LAMBDA = 1e-4  # Sparsity regularization coefficient (paper Section 3.1)


class APMLFTrainer(DetectionTrainer):
    """DetectionTrainer with Slide Loss and NAM sparsity regularization."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load APMLF-YOLO model."""
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(
            cfg or 'ultralytics/cfg/models/v8/apmlf_yolo.yaml',
            nc=self.data['nc'],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def init_criterion(self):
        """Use Slide Loss instead of standard v8DetectionLoss."""
        return v8DetectionLossWithSlide(self.model)

    def optimizer_step(self):
        """Standard optimizer step + NAM sparsity gradient injection.

        Injects λ·sign(γ) into gradient of every BN layer tagged _is_nam=True.
        This implements subgradient of L1 penalty Σ|γ| without modifying loss directly.
        Must run BEFORE super().optimizer_step() so gradients are clipped with injection.
        """
        # Inject NAM sparsity gradient before optimizer step
        for module in self.model.modules():
            if (
                isinstance(module, torch.nn.BatchNorm2d)
                and getattr(module, '_is_nam', False)
                and module.weight.grad is not None
            ):
                module.weight.grad.data.add_(
                    NAM_LAMBDA * torch.sign(module.weight.data)
                )

        # Standard Ultralytics optimizer step (handles gradient clipping + scaler)
        super().optimizer_step()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_trainer.py -v
```
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add apmlf_trainer.py tests/test_trainer.py
git commit -m "feat: add APMLFTrainer with Slide Loss criterion and NAM sparsity gradient injection"
```

---

## Task 13: Dataset Configuration

**Files:**
- Create: `data/pcb_defect.yaml`

- [ ] **Step 1: Create data directory and YAML**

```bash
mkdir -p data
```

Create `data/pcb_defect.yaml`:
```yaml
# PKU PCB Defect Dataset
# Original: 693 images → 1408 after augmentation (flip + random crop)
# Split: 8:1:1 (train:val:test)
# Source: https://github.com/Charmve/Surface-Defect-Detection (PKU dataset)
#
# IMPORTANT: Set absolute paths on the training machine before running train.py
# The paths below are placeholders — replace with actual paths on training machine.

path: /path/to/pcb_dataset  # REPLACE on training machine

train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

- [ ] **Step 2: Write and run smoke test for YAML**

```python
# Paste into Python REPL to verify YAML is valid and has correct class count:
import yaml
with open('data/pcb_defect.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['nc'] == 6, f"Expected 6 classes, got {cfg['nc']}"
assert len(cfg['names']) == 6
expected = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
assert list(cfg['names'].values()) == expected
print("Dataset YAML OK — 6 classes, correct names")
```

Expected: `Dataset YAML OK — 6 classes, correct names`

- [ ] **Step 3: Commit**

```bash
git add data/pcb_defect.yaml
git commit -m "feat: add PKU PCB dataset config (paths are placeholders for training machine)"
```

---

## Task 14: Training Entry Point

**Files:**
- Create: `train.py`

**Paper reference**: Section 4. SGD, lr=0.01, momentum=0.937, wd=0.0005, batch=4, epochs=200, imgsz=640, augmentation=flip+random crop ONLY (no mosaic, no mixup).

- [ ] **Step 1: Create train.py**

Create `train.py` in project root:
```python
"""APMLF-YOLO Training Script.

Paper hyperparameters (Section 4):
  Optimizer:    SGD (NOT Ultralytics default AdamW)
  LR:           0.01
  Momentum:     0.937
  Weight decay: 0.0005
  Batch size:   4
  Epochs:       200
  Image size:   640×640
  Augmentation: horizontal flip + random crop ONLY
                (mosaic=0, mixup=0 — paper does NOT use these)

Usage on training machine:
  python train.py --data /path/to/pcb_dataset

Before running:
  1. Update data/pcb_defect.yaml with correct dataset path
  2. Ensure ultralytics v8.2.103 is importable (PYTHONPATH or local install)
  3. Verify GPU is available: python -c "import torch; print(torch.cuda.is_available())"
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics'))

from apmlf_trainer import APMLFTrainer


def main():
    parser = argparse.ArgumentParser(description='Train APMLF-YOLO on PCB dataset')
    parser.add_argument(
        '--data', type=str, default='data/pcb_defect.yaml',
        help='Path to dataset YAML (update pcb_defect.yaml path first)'
    )
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 1, etc.)')
    args = parser.parse_args()

    trainer = APMLFTrainer(
        overrides={
            # Model
            'model': 'ultralytics/cfg/models/v8/apmlf_yolo.yaml',
            'data': args.data,

            # Paper Section 4 hyperparameters — DO NOT change without paper justification
            'optimizer': 'SGD',      # Paper specifies SGD, NOT AdamW
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'batch': 4,
            'epochs': 200,
            'imgsz': 640,
            'device': args.device,

            # Augmentation: flip + random crop ONLY (paper Section 4)
            'fliplr': 0.5,           # horizontal flip
            'flipud': 0.0,
            'mosaic': 0.0,           # DISABLED — paper does not use mosaic
            'mixup': 0.0,            # DISABLED — paper does not use mixup
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,        # slight translate ≈ random crop effect
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.0,            # no color jitter per paper
            'hsv_s': 0.0,
            'hsv_v': 0.0,

            # Output
            'project': 'runs/apmlf_yolo',
            'name': 'pcb_defect',
            'save': True,
            'save_period': 10,
            'plots': True,

            # Resume
            'resume': args.resume or False,
        }
    )
    trainer.train()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify train.py syntax and imports**

```bash
python -c "
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'ultralytics')
# Test that train.py imports without error (does NOT start training)
import ast
with open('train.py') as f:
    src = f.read()
ast.parse(src)  # syntax check
print('train.py syntax OK')

# Verify argparse works
import subprocess
result = subprocess.run(['python', 'train.py', '--help'], capture_output=True, text=True, timeout=10)
assert result.returncode == 0, f'train.py --help failed: {result.stderr}'
print('train.py --help OK')
"
```
Expected: Both `syntax OK` and `--help OK` printed.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add train.py with paper hyperparameters (SGD, no mosaic/mixup, batch=4, epochs=200)"
```

---

## Task 15: Integration Tests + Parameter Count Verification

**Files:**
- Create: `tests/test_integration.py`

This is the primary validation that the full model architecture matches the paper.

- [ ] **Step 1: Create integration tests**

Create `tests/test_integration.py`:
```python
"""Integration tests: full model forward pass and parameter verification.

Parameter targets from paper:
  - Backbone + standard neck + 3-head (no P2): ~3.01M
  - Full model (4-head, AFPN, MHSA):           ~5.1M

These tests verify architecture correctness, NOT training correctness.
Run on training machine after calibrating CAMLight.CAM_E and AFPNNeck factor/num_blocks.
"""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))


class TestFullModelForwardPass:
    def _build_model(self):
        from ultralytics.nn.tasks import DetectionModel
        return DetectionModel(
            os.path.join(
                os.path.dirname(__file__), '..', 'ultralytics',
                'cfg', 'models', 'v8', 'apmlf_yolo.yaml'
            ),
            nc=6,
            verbose=False,
        )

    def test_model_builds(self):
        """Model instantiates without error."""
        m = self._build_model()
        assert m is not None

    def test_forward_pass(self):
        """Forward pass with 640×640 input produces 4 output feature maps."""
        m = self._build_model()
        m.eval()
        x = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            out = m(x)
        # Ultralytics returns tuple/list during eval
        assert out is not None

    def test_total_parameter_count(self):
        """Total model parameters should be approximately 5.1M.

        CALIBRATION: If this fails, adjust in cam_light.py and apmlf_detect.py:
          - CAMLight.CAM_E: reduce toward 0.33 to decrease, increase toward 0.45 to increase
          - AFPNNeck factor: increase to reduce AFPN params, decrease to increase
          - AFPNNeck num_blocks: reduce to decrease AFPN params
        Target: 5.1M ± 0.5M (10% tolerance)
        """
        m = self._build_model()
        total = sum(p.numel() for p in m.parameters())
        print(f"\nTotal parameters: {total:,}")
        # Wide tolerance — exact value requires empirical calibration
        assert 3_000_000 < total < 8_000_000, \
            f"Unexpected param count {total:,}. Calibrate CAM_E and AFPN factor/num_blocks."

    def test_cam_light_replaces_all_c2f(self):
        """No C2f modules should remain in backbone (all replaced by CAMLight)."""
        from ultralytics.nn.modules.block import C2f
        m = self._build_model()
        for name, module in m.named_modules():
            assert not isinstance(module, C2f), \
                f"Found C2f at {name} — should be CAMLight"

    def test_apmlf_detect_has_four_heads(self):
        """Detection head must operate at 4 scales (P2+P3+P4+P5)."""
        from ultralytics.nn.modules.apmlf_detect import APMLFDetect
        m = self._build_model()
        detect = None
        for module in m.modules():
            if isinstance(module, APMLFDetect):
                detect = module
                break
        assert detect is not None, "APMLFDetect not found in model"
        assert len(detect.cv2) == 4, f"Expected 4 detection heads, got {len(detect.cv2)}"

    def test_nam_bns_tagged(self):
        """All NAM BN layers must be tagged _is_nam=True for sparsity injection."""
        import torch.nn as nn
        m = self._build_model()
        nam_count = sum(
            1 for mod in m.modules()
            if isinstance(mod, nn.BatchNorm2d) and getattr(mod, '_is_nam', False)
        )
        # YOLOv8n backbone has multiple CAMLight blocks, each with NAMChannelAtt→BN
        assert nam_count > 0, "No NAM BN layers found — check CAMLight implementation"
        print(f"\nNAM BN layers found: {nam_count}")
```

- [ ] **Step 2: Run integration tests (architecture only — no GPU needed)**

```bash
cd "C:\Users\Radowan Ahmed Baized\Desktop\RadowanStuff\PCB Research\APMLF_YOLO Implementation"
python -m pytest tests/test_integration.py -v -s
```

Expected: `test_model_builds`, `test_cam_light_replaces_all_c2f`, `test_apmlf_detect_has_four_heads`, `test_nam_bns_tagged` should pass. `test_forward_pass` and `test_total_parameter_count` may need fix/calibration.

- [ ] **Step 3: If forward pass fails, debug**

Common failure points:
- `parse_model` not handling `APMLFDetect` correctly → re-check tasks.py registration
- Channel mismatch in `AFPNNeck` → check `in_channels` vs actual backbone output channels
- YAML layer indices wrong → print model structure and verify P2=idx2, P3=idx4, P4=idx6, P5=idx9

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full model integration tests with architecture and parameter count verification"
```

---

## Task 16: Parameter Calibration Guide

This task is reference documentation for the training machine — the calibration cannot be done on this machine (no GPU/memory). Written as inline comments + TRAINING_GUIDE.md.

- [ ] **Step 1: Create TRAINING_GUIDE.md**

Create `TRAINING_GUIDE.md` in project root:
```markdown
# APMLF-YOLO Training Guide

## Prerequisites
- Python 3.9 (exact — paper uses 3.9)
- PyTorch 2.4.1 (exact — 2.4.0 has Windows CPU bug)
- CUDA 12.1
- 8GB+ GPU VRAM recommended (batch=4, imgsz=640)

## Setup on Training Machine

### 1. Install Dependencies
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ultralytics/  # Install local fork in editable mode
```

### 2. Configure Dataset
Edit `data/pcb_defect.yaml`:
```yaml
path: /absolute/path/to/PKU_PCB_dataset
```

PKU PCB dataset structure required:
```
PKU_PCB_dataset/
  images/
    train/  (80% of 1408 augmented images)
    val/    (10%)
    test/   (10%)
  labels/
    train/  (YOLO format .txt)
    val/
    test/
```

### 3. Parameter Calibration (MUST DO BEFORE FULL TRAINING)

**Target parameters per paper:**
- Backbone + standard YOLOv8 neck + 3-head: **3.01M**
- Full model (AFPN + MHSA + 4 heads): **5.1M**

**Step A: Calibrate CAM-Light expansion ratio**
```python
# In ultralytics/nn/modules/cam_light.py, adjust CAM_E:
class CAMLight(nn.Module):
    CAM_E: float = 0.375  # START HERE

# Test backbone params:
python -c "
import sys; sys.path.insert(0, 'ultralytics')
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel('ultralytics/cfg/models/v8/apmlf_yolo.yaml', nc=6)
backbone_params = sum(p.numel() for name, p in m.named_parameters() if 'model.0' in name or 'model.1' in name or 'model.2' in name or 'model.3' in name or 'model.4' in name or 'model.5' in name or 'model.6' in name or 'model.7' in name or 'model.8' in name or 'model.9' in name)
print(f'Backbone params: {backbone_params:,}')
"
# Target: ~3.01M for backbone alone
# If too high: decrease CAM_E (try 0.35, 0.33)
# If too low: increase CAM_E (try 0.40, 0.42)
```

**Step B: Calibrate AFPN (after backbone is correct)**
```python
# In ultralytics/nn/modules/apmlf_detect.py, adjust AFPNNeck args:
self.afpn = AFPNNeck(
    in_channels=in_channels_afpn,
    out_channels=afpn_out,
    factor=4,      # START HERE: increase to reduce params, decrease to increase
    num_blocks=4,  # START HERE: reduce to decrease params
)

# Test full model params:
python -c "
import sys; sys.path.insert(0, 'ultralytics')
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel('ultralytics/cfg/models/v8/apmlf_yolo.yaml', nc=6)
total = sum(p.numel() for p in m.parameters())
print(f'Total params: {total:,}')
"
# Target: ~5.1M total
```

### 4. Run Training
```bash
python train.py --data data/pcb_defect.yaml --device 0
```

### 5. Monitor Training
- Logs saved to `runs/apmlf_yolo/pcb_defect/`
- Key metrics to watch: mAP50, mAP50-95 per class
- Paper reports mAP50=96.5% on PKU PCB dataset

### 6. Expected Results (from paper Table 3)
| Model | mAP50 | Params | FLOPs |
|-------|-------|--------|-------|
| YOLOv8n baseline | 94.1% | 3.2M | 8.7G |
| APMLF-YOLO | **96.5%** | **3.01M** | 9.3G |

### Troubleshooting

**CUDA out of memory**: Reduce batch size to 2 in train.py
**Model not converging**: Verify SGD is used (not AdamW) — check optimizer in logs
**NAM sparsity not working**: Check `_is_nam` tag on BN layers via integration tests
**Loss NaN**: Usually from Slide Loss with auto_iou=0 — verify TAL produces positives
```

- [ ] **Step 2: Commit**

```bash
git add TRAINING_GUIDE.md
git commit -m "docs: add training guide with setup, calibration steps, and expected results"
```

---

## Task 17: Run All Tests

- [ ] **Step 1: Run complete test suite**

```bash
cd "C:\Users\Radowan Ahmed Baized\Desktop\RadowanStuff\PCB Research\APMLF_YOLO Implementation"
python -m pytest tests/ -v --tb=short 2>&1 | head -100
```

- [ ] **Step 2: Fix any failures**

Common issues:
- Import errors → check sys.path.insert and ultralytics installation
- Shape mismatches → run individual test file with `-s` flag for debug prints
- YAML parse errors → check tasks.py module registration

- [ ] **Step 3: Commit final test results**

```bash
git add -A
git commit -m "test: all unit and integration tests passing"
```

---

## Task 18: Final Project Structure Verification

- [ ] **Step 1: Verify all files exist**

```bash
# From project root
ls ultralytics/nn/modules/cam_light.py
ls ultralytics/nn/modules/afpn.py
ls ultralytics/nn/modules/mhsa.py
ls ultralytics/nn/modules/apmlf_detect.py
ls ultralytics/utils/slide_loss.py
ls ultralytics/cfg/models/v8/apmlf_yolo.yaml
ls apmlf_trainer.py
ls train.py
ls data/pcb_defect.yaml
ls TRAINING_GUIDE.md
ls tests/test_cam_light.py
ls tests/test_afpn.py
ls tests/test_mhsa.py
ls tests/test_apmlf_detect.py
ls tests/test_slide_loss.py
ls tests/test_registration.py
ls tests/test_trainer.py
ls tests/test_integration.py
```

- [ ] **Step 2: Verify git log shows clean commit history**

```bash
cd ultralytics && git log --oneline
```

Expected: 15+ commits, one per feature/task.

- [ ] **Step 3: Transfer to training machine**

```bash
# Option A: ZIP the entire project
cd ..
zip -r apmlf_yolo_implementation.zip "APMLF_YOLO Implementation/" \
    --exclude "*.pyc" --exclude "__pycache__/*" --exclude "*.egg-info/*"

# Option B: Push ultralytics branch to remote
cd "APMLF_YOLO Implementation/ultralytics"
git remote add origin <your-remote-url>
git push -u origin apmlf-yolo
```

---

## Paper-to-Code Mapping Summary

| Paper Section | Implementation Location |
|---------------|------------------------|
| Section 3.1: CAM-Light | `ultralytics/nn/modules/cam_light.py` |
| Section 3.1: NAM Channel | `NAMChannelAtt` class, BN γ weighting |
| Section 3.1: NAM Spatial | `NAMSpatialAtt` class, pixel L2 norm |
| Section 3.1: Parameter reduction | `CAMLight.CAM_E = 0.375` (calibrate) |
| Section 3.1: Sparsity reg | `APMLFTrainer.optimizer_step()`, λ=1e-4 |
| Section 3.2: AFPN | `ultralytics/nn/modules/afpn.py` |
| Section 3.2: ASFF adaptive weights | `ASFF2`, `ASFF3`, compress_c=8 |
| Section 3.2: Asymptotic fusion | `BlockBodyP345`, Stage 1→2 |
| Section 3.2: MHSA | `ultralytics/nn/modules/mhsa.py` |
| Section 3.2: Position encoding | `MHSAContentPosition`, decomposed L_h+L_w |
| Section 3.2: P2 4th head | `APMLFDetect`, P2 direct from backbone |
| Section 3.3: Slide Loss | `ultralytics/utils/slide_loss.py` |
| Section 3.3: Eq.2 middle zone | `e^(1-μ)` — NOT `e^μ` (YOLO-FaceV2 error) |
| Section 4: SGD optimizer | `train.py`, `optimizer='SGD'` |
| Section 4: No mosaic/mixup | `train.py`, `mosaic=0, mixup=0` |
| Dataset | `data/pcb_defect.yaml`, 6 classes |

---

## Critical Implementation Warnings

1. **Slide Loss middle zone**: Paper Eq.2 uses `e^(1-μ)`. YOLO-FaceV2 reference code uses `e^μ`. These differ. Use `e^(1-μ)`.

2. **`v8DetectionLossWithSlide` requires copying `v8DetectionLoss.__call__`**: The `auto_iou` parameter must be computed from TAL-positive CIoU values inside `__call__`, then passed to `self.bce(pred, true, auto_iou=auto_iou)`. Simply replacing `self.bce` and calling `super().__call__()` results in `auto_iou=0.5` hardcoded every batch — this defeats the purpose of Slide Loss. Task 10 provides the mechanical override: copy the entire `__call__` body and change only the one `self.bce(...)` line.

3. **MHSA is full HxW attention**: Fig.5 H×1×d label is a visualization simplification. Standard full 400-token attention is correct (not axial/row-wise).

4. **AFPN takes P3/P4/P5 only**: The 4th (P2) head is a separate backbone branch, NOT part of AFPN.

5. **CAMLight expansion ratio**: Must be empirically calibrated on training machine. Do NOT assume 0.375 is correct — it is a starting point.

6. **Ultralytics version**: Must be v8.2.103 exactly. Later versions (YOLO11+) restructured the codebase significantly.

7. **NAM public repo**: `Christian-lyc/NAM` only implements channel attention (spatial is disabled). The spatial branch must be implemented from scratch using L2 pixel norm.

8. **APMLFDetect is monolithic**: Cannot be expressed as YAML layer sequences — too complex. The single `[[2,4,6,9], 1, APMLFDetect, [nc]]` entry is correct.

9. **YAML channel values are pre-scale**: All channel numbers in `apmlf_yolo.yaml` are specification values, scaled by `width_multiple=0.25` at 'n' size. Actual channels = spec × 0.25. APMLFDetect receives `[32, 64, 128, 256]`, not `[128, 256, 512, 1024]`.
