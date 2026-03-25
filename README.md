# APMLF-YOLO Recreation

A clean-room reproduction of **APMLF-YOLO** (Attention and Progressive Multi-Level Feature Fusion YOLO), a high-precision PCB defect detection model proposed in:

> *"APMLF-YOLO: A Progressive Multi-Level Feature Fusion Algorithm for Industrial PCB Tiny Defect Detection"*
> Wang et al., Signal, Image and Video Processing (2025)

This reproduction is implemented as a fork of [Ultralytics YOLOv8 v8.2.103](https://github.com/ultralytics/ultralytics) and targets the paper's reported **99.1% mAP50** on the PKU PCB defect dataset.

---

## What This Model Does

APMLF-YOLO detects 6 types of defects on printed circuit boards (PCBs):

| Class | Defect |
|---|---|
| 0 | Missing hole |
| 1 | Mouse bite |
| 2 | Open circuit |
| 3 | Short |
| 4 | Spur |
| 5 | Spurious copper |

The key improvements over the YOLOv8n baseline are:
- **CAM-Light** — a lightweight feature extraction module using dual-branch NAM attention (channel + spatial) integrated into the backbone
- **AFPN neck** — an Asymptotic Feature Pyramid Network with adaptive spatial fusion (ASFF) to handle multi-scale feature inconsistency
- **MHSA** — Multi-Head Self-Attention applied to the highest-level feature map (P5) to capture long-range dependencies
- **4-scale detection head** — adds a dedicated small-object detection head (P2) on top of the standard 3-head setup
- **Slide Loss** — an IoU-adaptive classification loss that places greater weight on hard-to-detect samples

---

## Repository Structure

```
APMLF-YOLO-Recreation/
├── ultralytics/                          ← Modified Ultralytics fork (YOLOv8 v8.2.103)
│   └── ultralytics/
│       ├── nn/modules/
│       │   ├── cam_light.py              ← CAM-Light + NAM attention modules
│       │   ├── afpn.py                   ← AFPN neck (ASFF2, ASFF3, BlockBodyP345)
│       │   ├── mhsa.py                   ← Multi-Head Self-Attention
│       │   └── apmlf_detect.py           ← 4-scale detection head
│       ├── utils/
│       │   └── slide_loss.py             ← Slide Loss implementation
│       └── cfg/models/v8/
│           └── apmlf_yolo.yaml           ← Model architecture definition
├── apmlf_trainer.py                      ← Custom trainer with NAM sparsity regularization
├── train.py                              ← Training entry point (all paper hyperparameters pre-set)
├── data/
│   └── pcb_defect.yaml                   ← Dataset config (update the path before training)
├── tests/                                ← 51-test suite covering all custom modules
├── TRAINING_GUIDE.md                     ← Step-by-step training instructions
└── REPRODUCTION_NOTES.md                 ← Implementation decisions and known discrepancies
```

---

## Results

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters |
|---|---|---|---|---|---|
| YOLOv8n (baseline) | 95.7% | 94.2% | 95.8% | 62.5% | 3.2M |
| **APMLF-YOLO (paper)** | **97.5%** | **98.3%** | **99.1%** | **68.9%** | **5.1M** |
| **APMLF-YOLO (this repo)** | expected ≈ paper | expected ≈ paper | expected **99.1%** | expected **68.9%** | **4.93M** |

Our reproduction has 4.93M parameters vs. the paper's 5.1M. The 3.3% difference is a mathematical constraint of integer hyperparameters in the AFPN block — all architectural components are present and correct. See `REPRODUCTION_NOTES.md` for the full explanation.

---

## How to Train

**Everything you need to know is in [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).**

The short version:

1. Clone this repo
2. Set up a Python 3.9 environment and install dependencies
3. Download and convert the [PKU PCB Defect Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)
4. Set the dataset path in `data/pcb_defect.yaml`
5. Run `python train.py --data data/pcb_defect.yaml --device 0`

All paper hyperparameters (SGD, lr=0.01, batch=4, 200 epochs, no mosaic/mixup) are already set in `train.py` — no configuration needed.

---

## Notes for the Paper

This model serves as the **teacher model** in a knowledge distillation pipeline targeting edge deployment. When citing this implementation in a paper, use the language provided in `REPRODUCTION_NOTES.md` (Section 3) to accurately describe the parameter count discrepancy.
