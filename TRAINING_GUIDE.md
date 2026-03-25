# APMLF-YOLO Training Guide

This guide walks you through everything needed to train the APMLF-YOLO model on the PKU PCB defect dataset — from cloning the repository to evaluating the final results. No prior experience with YOLO or Ultralytics is assumed.

**Approximate total time (first-time setup):** 1–2 hours of setup, then 8–20 hours of unattended training depending on your GPU.

---

## What You Need

Before starting, make sure your **training machine** has:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 (exact) | Other versions may cause compatibility issues |
| NVIDIA GPU | Any with CUDA support | CPU training is possible but impractically slow |
| GPU VRAM | 4 GB or more | The paper trained on a GTX 1650 (4 GB) at batch=4, 640×640 |
| Disk space | ~5 GB | Dataset (~1.3 GB) + checkpoints + logs |

---

## Step 1 — Clone the Repository

This is the first thing to do. Clone the project onto your training machine:

```bash
git clone https://github.com/Radowan30/APMLF-YOLO-Recreation.git
cd "APMLF-YOLO-Recreation"
```

After cloning, the folder will contain all the project files:

```
APMLF_YOLO Implementation/
├── ultralytics/          ← the modified Ultralytics fork (REQUIRED)
├── apmlf_trainer.py      ← custom trainer
├── train.py              ← training entry point
├── data/
│   └── pcb_defect.yaml   ← dataset config (you will edit this in Step 4)
└── tests/                ← verify everything works
```

> **Note:** The dataset itself is **not** included in the repository — it is too large to store on GitHub. You will download and prepare it in Step 3.

All remaining steps in this guide assume you are working from inside the `APMLF_YOLO Implementation/` folder.

---

## SStep 2 — Set Up the Python Environment

### 2a. Install Python 3.9 and create a virtual environment

The paper used Python 3.9 exactly. Other versions may work but are untested.

We use **uv** to manage the Python version and virtual environment. First, install uv if you don't have it:

```bash
# On Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (run in PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install Python 3.9 and create the virtual environment inside the project folder:

```bash
uv python install 3.9
uv venv --python 3.9 .venv
```

Activate the virtual environment:

```bash
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

Verify the correct version is active:
```bash
python --version
# Expected: Python 3.9.x
```

> **Important:** Make sure your virtual environment is active (you should see `(.venv)` at the start of your terminal prompt) before running any of the following steps. You will need to activate it again each time you open a new terminal.

---

### 2b. Check your CUDA drivers

Before installing PyTorch, confirm your NVIDIA drivers support CUDA 12.1:
```bash
nvidia-smi
```

Look for `CUDA Version: 12.x` (or higher) in the output. If the version shown is lower than 12.1, update your NVIDIA drivers from https://www.nvidia.com/drivers before continuing.

If `nvidia-smi` is not found, your NVIDIA drivers are not installed. Install them from the NVIDIA website for your GPU model.

---

### 2c. Install PyTorch with CUDA

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

This installs the CUDA 12.1 version of PyTorch. It is a large download (~2 GB) — this may take several minutes.

Verify that PyTorch can see your GPU:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce GTX 1650   (or whatever your GPU is)
```

If it prints `CUDA available: False`, do not proceed — PyTorch cannot use your GPU. Common causes:
- NVIDIA drivers not installed or too old (re-check `nvidia-smi`)
- Wrong PyTorch build (make sure you used the `--index-url` above, not the default pip index)

---

### 2d. Install the Ultralytics fork

This project uses a **modified version of Ultralytics** that contains all the custom APMLF-YOLO modules. You must install this local copy — do **not** install Ultralytics from pip, as that would give you the unmodified version.

From inside the project folder:
```bash
pip install -e ultralytics/
```

The `-e` flag installs it in **editable mode**, meaning Python reads the files directly from the `ultralytics/` folder. Any changes you make there take effect immediately without reinstalling.

Verify the custom modules are importable:
```bash
python -c "from ultralytics.nn.modules.cam_light import CAMLight; print('Install OK')"
# Expected: Install OK
```

If you see `ModuleNotFoundError`, make sure you ran the command from inside the `APMLF_YOLO Implementation/` folder.

---

## Step 3 — Download and Prepare the Dataset

### 3a. Download the DeepPCB dataset

The dataset used in the paper is the **DeepPCB dataset** published by Peking University researchers (cited as reference [14] in the paper). It contains PCB images annotated with 6 defect types.

**Download it from GitHub:**
```
https://github.com/tangsanli5201/DeepPCB
```

Click **Code → Download ZIP** and extract it, or clone with Git:
```bash
git clone https://github.com/tangsanli5201/DeepPCB.git
```

After downloading, you will have a folder called `DeepPCB/` containing PCB image pairs and annotation files.

**The 6 defect classes in the dataset:**

| Class ID | Name | Description |
|---|---|---|
| 0 | `missing_hole` | A hole that should exist is absent |
| 1 | `mouse_bite` | Irregular bite-shaped erosion on copper |
| 2 | `open_circuit` | A broken copper trace |
| 3 | `short` | Unintended connection between traces |
| 4 | `spur` | A small copper spike |
| 5 | `spurious_copper` | Copper remaining where it should have been etched |

---

### 3b. Understand the DeepPCB dataset structure and annotation format

**Folder structure inside PCBData/:**

```
PCBData/
├── trainval.txt              ← index file listing 1000 training pairs
├── test.txt                  ← index file listing 500 test pairs
└── groupXXXXX/
    ├── XXXXX/                ← images: XXXXXX.jpg (640×640)
    └── XXXXX_not/            ← annotations: XXXXXX.txt
```

Images and annotations are in **different subdirectories**. The provided `trainval.txt` and `test.txt` files give the correct pairing for each image — the convert script uses these directly.

**Annotation format** — each line in a `.txt` file describes one defect:

```
x1,y1,x2,y2,type
```

- `x1 y1` — top-left corner of the bounding box (pixels)
- `x2 y2` — bottom-right corner of the bounding box (pixels)
- `type` — integer defect code (see table below); may be comma or space separated

**Defect code mapping** (from DeepPCB README):

| DeepPCB code | DeepPCB name | Our class name | Our class ID |
|---|---|---|---|
| 1 | open | open_circuit | 2 |
| 2 | short | short | 3 |
| 3 | mousebite | mouse_bite | 1 |
| 4 | spur | spur | 4 |
| 5 | copper | spurious_copper | 5 |
| 6 | pin-hole | missing_hole | 0 |

---

### 3c. Convert to YOLO format

YOLO expects a `.txt` label file for each image, where every line is:

```
<class_id> <x_center> <y_center> <width> <height>
```

All five values are **normalized to [0, 1]** relative to the image dimensions. For example:
```
0 0.512 0.334 0.045 0.062
```
means: class 0 (missing_hole), bounding box centered at 51.2% across and 33.4% down, 4.5% of image width wide, 6.2% of image height tall.

**Conversion formula** (for an image of width `W` and height `H`):
```
class_id  = see mapping table in section 3b above (NOT defect_code - 1)
x_center  = (x1 + x2) / 2 / W
y_center  = (y1 + y2) / 2 / H
width     = (x2 - x1) / W
height    = (y2 - y1) / H
```

The class ID mapping is **not sequential** — for example, DeepPCB code 6 (pin-hole) maps to class 0 (missing_hole), and code 1 (open) maps to class 2 (open_circuit). The convert script handles this via a lookup table.

Create a new file called `convert_deeppcb.py` **outside** the project folder (e.g. in your home directory), paste the script below into it, and run it to convert the annotations:

```python
"""Convert DeepPCB annotations to YOLO format with horizontal flip augmentation.

DeepPCB dataset structure (github.com/tangsanli5201/DeepPCB):
  PCBData/
  ├── trainval.txt              ← 1000 image-annotation pairs
  ├── test.txt                  ← 500 image-annotation pairs
  └── groupXXXXX/
      ├── XXXXX/                ← images:      XXXXXX.jpg
      └── XXXXX_not/            ← annotations: XXXXXX.txt

Each line in trainval.txt / test.txt:
  groupXXXXX/XXXXX/XXXXXX.jpg  groupXXXXX/XXXXX_not/XXXXXX.txt

DeepPCB annotation format (per line, comma or space separated):
  x1 y1 x2 y2 type

DeepPCB defect codes → YOLO class IDs:
  1=open        → 2 (open_circuit)
  2=short       → 3 (short)
  3=mousebite   → 1 (mouse_bite)
  4=spur        → 4 (spur)
  5=copper      → 5 (spurious_copper)
  6=pin-hole    → 0 (missing_hole)

Usage:
    python convert_deeppcb.py \
        --deeppcb_dir /path/to/DeepPCB/PCBData \
        --output_dir /path/to/pcb_dataset

Replicates the paper's dataset preparation (Section 4.1): creates a
horizontally flipped copy of every image, then splits 8:1:1 AFTER
augmentation. The full DeepPCB dataset has 1500 images (vs the paper's
693); all available images are used since the exact paper subset is
not specified.

Requires: Pillow (installed automatically as a dependency of torchvision).
"""
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image, ImageOps

IMAGE_W = 640
IMAGE_H = 640

# DeepPCB integer code → YOLO class ID
# Source: DeepPCB README — "0=background(unused), 1=open, 2=short,
#         3=mousebite, 4=spur, 5=copper, 6=pin-hole"
# YOLO classes (pcb_defect.yaml): 0=missing_hole, 1=mouse_bite,
#   2=open_circuit, 3=short, 4=spur, 5=spurious_copper
DEFECT_CODE_TO_CLASS_ID = {
    1: 2,  # open      → open_circuit
    2: 3,  # short     → short
    3: 1,  # mousebite → mouse_bite
    4: 4,  # spur      → spur
    5: 5,  # copper    → spurious_copper
    6: 0,  # pin-hole  → missing_hole
}


def load_index(deeppcb, filename):
    """Read image-annotation pairs from a DeepPCB index file."""
    pairs = []
    index_path = deeppcb / filename
    if not index_path.exists():
        print(f"  WARNING: {filename} not found, skipping.")
        return pairs
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path = deeppcb / parts[0]
            ann_path = deeppcb / parts[1]
            if img_path.exists() and ann_path.exists():
                pairs.append((img_path, ann_path))
    return pairs


def parse_annotation(ann_path):
    """Parse a DeepPCB annotation file (handles comma or space separators)."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            # Handles both "x1 y1 x2 y2 type" and "x1,y1,x2,y2,type"
            parts = line.strip().replace(',', ' ').split()
            if len(parts) != 5:
                continue
            try:
                x1, y1, x2, y2, code = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except ValueError:
                continue
            class_id = DEFECT_CODE_TO_CLASS_ID.get(code)
            if class_id is None:
                continue  # skip code=0 (background) or unknown codes
            xc = (x1 + x2) / 2 / IMAGE_W
            yc = (y1 + y2) / 2 / IMAGE_H
            w  = (x2 - x1) / IMAGE_W
            h  = (y2 - y1) / IMAGE_H
            boxes.append((class_id, xc, yc, w, h))
    return boxes


def flip_boxes(boxes):
    """Horizontal flip: x_center → 1 - x_center. All other values unchanged."""
    return [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in boxes]


def write_label(dst_txt, boxes):
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in boxes]
    with open(dst_txt, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deeppcb_dir', required=True,
                        help='Path to DeepPCB/PCBData folder (must contain trainval.txt and test.txt)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write the converted dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    deeppcb = Path(args.deeppcb_dir)
    out = Path(args.output_dir)

    # Use the index files provided by DeepPCB for correct image-annotation pairing.
    # Images are in groupXXXXX/XXXXX/ and annotations in groupXXXXX/XXXXX_not/ —
    # they cannot be matched by filename alone.
    orig_pairs = load_index(deeppcb, 'trainval.txt')
    orig_pairs += load_index(deeppcb, 'test.txt')

    if not orig_pairs:
        print(f"\nERROR: No valid pairs found.")
        print(f"Make sure --deeppcb_dir points to the PCBData/ folder that")
        print(f"contains trainval.txt and test.txt.")
        return

    print(f"Found {len(orig_pairs)} original image-annotation pairs")

    # Build augmented list: original + horizontal flip for each image.
    # Augment FIRST, then split — matching the paper's order of operations.
    augmented = []
    skipped = 0
    for img_path, ann_path in orig_pairs:
        boxes = parse_annotation(ann_path)
        if not boxes:
            skipped += 1
            continue
        augmented.append(('orig', img_path, boxes,             img_path.stem))
        augmented.append(('flip', img_path, flip_boxes(boxes), img_path.stem + '_flip'))

    if skipped:
        print(f"Skipped {skipped} images with no parseable annotations")
    print(f"Augmented dataset size (orig + flip): {len(augmented)}")

    # Shuffle then split 8:1:1
    random.seed(args.seed)
    random.shuffle(augmented)
    n = len(augmented)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    splits = {
        'train': augmented[:n_train],
        'val':   augmented[n_train:n_train + n_val],
        'test':  augmented[n_train + n_val:],
    }

    for split, items in splits.items():
        img_dir = out / 'images' / split
        lbl_dir = out / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for kind, img_path, boxes, stem in items:
            dst_img = img_dir / (stem + '.jpg')
            dst_lbl = lbl_dir / (stem + '.txt')
            if kind == 'orig':
                shutil.copy(img_path, dst_img)
            else:
                ImageOps.mirror(Image.open(img_path)).save(dst_img)
            write_label(dst_lbl, boxes)
        print(f"  {split}: {len(items)} images")

    print(f"\nDone. Dataset written to: {out}")
    print("Update 'path' in data/pcb_defect.yaml to:", out.resolve())


if __name__ == '__main__':
    main()
```

Run the script:
```bash
python convert_deeppcb.py \
    --deeppcb_dir /path/to/DeepPCB/PCBData \
    --output_dir /path/to/pcb_dataset
```

> **Note:** If the DeepPCB images in your download are not 640×640, update `IMAGE_W` and `IMAGE_H` at the top of the script to match. You can check with any image viewer.

---

### 3d. Expected folder structure after conversion

```
pcb_dataset/
├── images/
│   ├── train/     ← ~2400 images (~80% of ~3000)
│   ├── val/       ← ~300 images (~10%)
│   └── test/      ← ~300 images (~10%)
└── labels/
    ├── train/     ← matching .txt label files (YOLO format)
    ├── val/
    └── test/
```

The script will print the exact counts when it runs. The full DeepPCB dataset has **1500 images** (1000 trainval + 500 test); after horizontal flip augmentation the total is ~3000, split 8:1:1. The paper reported using 693 images, but DeepPCB has always contained 1500 — the exact subset used by the paper's authors is not specified in the paper and cannot be recovered. Using all 1500 images is the correct approach and will produce results at least as good as the paper.

---

## Step 4 — Configure the Dataset Path

Open `data/pcb_defect.yaml` in any text editor. Find the `path` line near the top and replace it with the **full absolute path** to your `pcb_dataset/` folder (the one created by the conversion script in Step 3):

```yaml
# Change this line:
path: /path/to/pcb_dataset

# Example on Linux:
path: /home/yourname/datasets/pcb_dataset

# Example on Windows (use forward slashes):
path: C:/Users/yourname/datasets/pcb_dataset
```

Save the file. Do not change anything else in it — the class names and split names are already correct.

---

## Step 5 — Verify Everything Before Training

Run the test suite to confirm the whole setup is correct before committing to a long training run. From inside the project folder:

```bash
python -m pytest tests/ -q
```

Expected output:
```
51 passed, 14 warnings in X.XXs
```

If any tests fail, the error message will say which module has a problem. The most common cause is the editable install not being set up — run `pip install -e ultralytics/` again from inside the project folder.

Also check that the model builds and has the right number of parameters:

**On Linux/Mac:**
```bash
python -c "
import sys; sys.path.insert(0, 'ultralytics')
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel('ultralytics/ultralytics/cfg/models/v8/apmlf_yolo.yaml', nc=6, verbose=False)
total = sum(p.numel() for p in m.parameters())
print(f'Total parameters: {total:,}')
print(f'Target from paper: ~5,100,000')
"
```

**On Windows**, create a new file called `check_params.py` in the project folder, paste the code below, and run `python check_params.py`:
```python
import sys
sys.path.insert(0, 'ultralytics')
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel('ultralytics/ultralytics/cfg/models/v8/apmlf_yolo.yaml', nc=6, verbose=False)
total = sum(p.numel() for p in m.parameters())
print(f'Total parameters: {total:,}')
print(f'Target from paper: ~5,100,000')
```

Expected output:
```
Total parameters: 4,933,165
Target from paper: ~5,100,000
```

The 3.3% difference from the paper's 5.1M is expected and documented in `REPRODUCTION_NOTES.md`.

---

## Step 6 — Start Training

From inside the project folder:

```bash
python train.py --data data/pcb_defect.yaml --device 0
```

**What the arguments mean:**
- `--data data/pcb_defect.yaml` — the dataset config file you edited in Step 4
- `--device 0` — use the first GPU (GPU index 0). If you have multiple GPUs, use `--device 0,1`
- `--resume path/to/last.pt` — if training is interrupted, resume from the last saved checkpoint

Once training starts, you will see a progress bar printing loss values each epoch. You do not need to watch it — you can leave it running and check back later.

**All paper hyperparameters are already set** in `train.py` — do not change them:

| Setting | Value | Why |
|---|---|---|
| Optimizer | SGD | Paper Section 4 |
| Learning rate | 0.01 | Paper Section 4 |
| Momentum | 0.937 | Paper Section 4 |
| Weight decay | 0.0005 | Paper Section 4 |
| Batch size | 4 | Paper Section 4 |
| Epochs | 200 | Paper Section 4 |
| Image size | 640×640 | Paper Section 4 |
| Mosaic augmentation | **OFF** | Paper uses flip + crop only |
| Mixup augmentation | **OFF** | Paper uses flip + crop only |

**How long will training take?**

| GPU | Approximate time for 200 epochs |
|---|---|
| GTX 1650 (4 GB) — same as paper | ~15–20 hours |
| RTX 3080/3090 | ~6–8 hours |
| RTX 4090 | ~3–4 hours |

These are rough estimates. Actual time depends on your system.

---

## Step 7 — Monitor Training Progress

Training outputs are saved to `runs/apmlf_yolo/pcb_defect/` inside the project folder.

```
runs/apmlf_yolo/pcb_defect/
├── weights/
│   ├── best.pt        ← best model so far (highest mAP50 on validation set)
│   └── last.pt        ← most recent checkpoint (use this to resume if interrupted)
├── results.csv        ← per-epoch numbers for all metrics
├── results.png        ← training curves plotted as an image
└── val_batch*.jpg     ← sample images showing what the model detects on validation data
```

**Key numbers to watch in `results.csv`** (open it in Excel or any spreadsheet app):
- `metrics/mAP50(B)` — the main accuracy metric; should reach **99.1%** by epoch 200
- `train/box_loss` — localization loss; should steadily decrease over training
- `train/cls_loss` — classification loss (uses Slide Loss); should steadily decrease
- `val/box_loss` and `val/cls_loss` — validation losses; should decrease alongside training losses

A checkpoint is saved every 10 epochs, so if training is interrupted you can resume without losing much progress:
```bash
python train.py --data data/pcb_defect.yaml --device 0 --resume runs/apmlf_yolo/pcb_defect/weights/last.pt
```

---

## Step 8 — Evaluate the Final Model

After training completes, evaluate on the held-out test set to get your final numbers.

**On Linux/Mac:**
```bash
python -c "
import sys; sys.path.insert(0, 'ultralytics')
from ultralytics.models import YOLO
model = YOLO('runs/apmlf_yolo/pcb_defect/weights/best.pt')
results = model.val(data='data/pcb_defect.yaml', split='test')
print(f'Precision:  {results.box.mp * 100:.1f}%')
print(f'Recall:     {results.box.mr * 100:.1f}%')
print(f'mAP50:      {results.box.map50 * 100:.1f}%')
print(f'mAP50-95:   {results.box.map * 100:.1f}%')
"
```

**On Windows**, create a new file called `evaluate.py` in the project folder, paste the code below, and run `python evaluate.py`:
```python
import sys
sys.path.insert(0, 'ultralytics')
from ultralytics.models import YOLO
model = YOLO('runs/apmlf_yolo/pcb_defect/weights/best.pt')
results = model.val(data='data/pcb_defect.yaml', split='test')
print(f'Precision:  {results.box.mp * 100:.1f}%')
print(f'Recall:     {results.box.mr * 100:.1f}%')
print(f'mAP50:      {results.box.map50 * 100:.1f}%')
print(f'mAP50-95:   {results.box.map * 100:.1f}%')
```

### Expected Results (from paper Table 3)

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters |
|---|---|---|---|---|---|
| YOLOv8n (baseline) | 95.7% | 94.2% | 95.8% | 62.5% | 3.2M |
| **APMLF-YOLO (ours)** | **97.5%** | **98.3%** | **99.1%** | **68.9%** | **~5M** |

Our reproduction has 4.93M parameters (see `REPRODUCTION_NOTES.md` for why this differs from the paper's 5.1M by 3.3%).

---

## Troubleshooting

**"CUDA out of memory"**

Your GPU ran out of memory. Open `train.py` and change `'batch': 4` to `'batch': 2`. This uses less memory at the cost of slightly slower training.

---

**"ModuleNotFoundError: No module named 'ultralytics'"**

The editable install is not active. Make sure:
1. Your virtual environment is activated (you should see `(.venv)` at the start of your prompt)
2. You ran `pip install -e ultralytics/` from inside the `APMLF_YOLO Implementation/` folder

---

**`nvidia-smi` works but `torch.cuda.is_available()` returns False**

You installed the CPU-only version of PyTorch. Uninstall it and reinstall with the correct URL:
```bash
pip uninstall torch torchvision -y
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

---

**"Model not converging" — loss is not going down after many epochs**

Check that SGD is being used: look at the first few lines of training output — it should say `optimizer: SGD`. If it says `AdamW`, the `train.py` override is not being applied (make sure you are running `train.py` and not some other script).

---

**Loss becomes NaN (not a number) early in training**

This usually happens in the first 2–3 epochs when there are no positive detections yet. It should self-correct. If NaN persists past epoch 5:
- Check that your label files are in YOLO format (not the original DeepPCB format)
- Check that the `path` in `data/pcb_defect.yaml` points to the correct folder

---

**Training was interrupted — how do I resume?**

```bash
python train.py --data data/pcb_defect.yaml --device 0 --resume runs/apmlf_yolo/pcb_defect/weights/last.pt
```

---

**"51 passed" becomes fewer than 51 tests after reinstalling packages**

The editable install was overwritten. Fix it:
```bash
pip install -e ultralytics/
```
Run the tests again — all 51 should pass.

---

**Training resumes but mAP is stuck and not improving**

Delete the `runs/` folder and start training from scratch:
```bash
rm -rf runs/   # Linux/Mac
# or on Windows: delete the runs\ folder manually
```
Stale optimizer state from a previous training run with different settings can prevent the model from learning.
