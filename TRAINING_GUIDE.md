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

## Step 2 — Set Up the Python Environment

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

### 3a. Download the PKU PCB Defect Dataset

The dataset used in the paper is the **PKU PCB Defect Dataset** (also known as HRIPCB), published by the Open Lab on Human Robot Interaction at Peking University. It contains 693 RGB color PCB images annotated with 6 defect types in PASCAL VOC XML format.

**Download options (both give identical content):**

Option A — Kaggle (~2 GB ZIP, free account required):
```
https://www.kaggle.com/datasets/akhatova/pcb-defects
```
Download the `PCB_DATASET.zip` and extract it.

Option B — GitHub (no account needed):
```
https://github.com/Ironbrotherstyle/PCB-DATASET
```
Click **Code → Download ZIP** and extract it.

After extracting you will have a folder called `PCB_DATASET/`.

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

### 3b. Understand the dataset structure and annotation format

**Folder structure inside PCB_DATASET/:**

```
PCB_DATASET/
├── images/
│   ├── Missing_hole/      ← 115 RGB .jpg images (~3034×1586 px each)
│   ├── Mouse_bite/        ← 115 images
│   ├── Open_circuit/      ← 116 images
│   ├── Short/             ← 116 images
│   ├── Spur/              ← 115 images
│   └── Spurious_copper/   ← 116 images   (693 total)
├── Annotations/
│   ├── Missing_hole/      ← matching PASCAL VOC .xml files (same filenames)
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
├── PCB_USED/              ← 10 template PCB board images (not used for training)
├── rotation/              ← pre-rotated copies (NOT used — paper uses flip, not rotation)
└── rotate.py
```

Each image folder contains images named `{PCB_ID}_{class_name}_{seq}.jpg` (e.g. `01_missing_hole_03.jpg`). The annotation file for each image is in the matching `Annotations/{ClassName}/` subfolder with the same stem and `.xml` extension.

**Annotation format** — PASCAL VOC XML. Example:

```xml
<annotation>
  <size>
    <width>3034</width>
    <height>1586</height>
    <depth>3</depth>
  </size>
  <object>
    <name>missing_hole</name>
    <bndbox>
      <xmin>2459</xmin><ymin>1274</ymin>
      <xmax>2530</xmax><ymax>1329</ymax>
    </bndbox>
  </object>
  ...
</annotation>
```

Key points:
- `<name>` is the class string directly (e.g. `missing_hole`) — no integer code lookup needed
- `<width>/<height>` in `<size>` give the image dimensions used to normalize coordinates
- Each image contains 3–5 defects of the **same class** (single-class images)
- The `<path>` field contains an absolute path from the original author's machine — **ignore it**

**Class name → YOLO class ID mapping:**

| XML `<name>` | YOLO class ID |
|---|---|
| `missing_hole` | 0 |
| `mouse_bite` | 1 |
| `open_circuit` | 2 |
| `short` | 3 |
| `spur` | 4 |
| `spurious_copper` | 5 |

---

### 3c. Convert to YOLO format

YOLO expects a `.txt` label file for each image, where every line is:

```
<class_id> <x_center> <y_center> <width> <height>
```

All five values are **normalized to [0, 1]** relative to the image dimensions.

**Conversion formula** (for an image of width `W` and height `H`, read from the XML `<size>` element):
```
class_id  = see mapping table in section 3b above
x_center  = (xmin + xmax) / 2 / W
y_center  = (ymin + ymax) / 2 / H
width     = (xmax - xmin) / W
height    = (ymax - ymin) / H
```

Create a new file called `convert_pcb.py` **outside** the project folder (e.g. in your home directory), paste the script below into it, and run it:

```python
"""Convert PKU PCB Defect Dataset annotations to YOLO format with horizontal flip augmentation.

Dataset: HRIPCB / PKU PCB Defect Dataset
  Kaggle:  https://www.kaggle.com/datasets/akhatova/pcb-defects
  GitHub:  https://github.com/Ironbrotherstyle/PCB-DATASET

Folder structure (PCB_DATASET/):
  images/
    Missing_hole/      ← ~115 .jpg images per class (~3034×1586 px, RGB)
    Mouse_bite/
    Open_circuit/
    Short/
    Spur/
    Spurious_copper/
  Annotations/
    Missing_hole/      ← matching PASCAL VOC .xml files
    Mouse_bite/
    ...

Augmentation: horizontal flip of every image → 693 × 2 = 1386 images total.
The paper reports 1408 after "flipping and random cropping"; the extra 22 images
came from offline random cropping with parameters the authors did not publish.
1386 is the closest deterministically reproducible count. The translate=0.1
setting in train.py provides random-crop diversity online during training.

The dataset is split 8:1:1 AFTER augmentation, matching the paper's order.

Requires: Pillow (pip install pillow)
"""
import shutil
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageOps

# Class name (from VOC XML <name> tag) → YOLO class ID
# Matches data/pcb_defect.yaml
CLASS_NAME_TO_ID = {
    'missing_hole':    0,
    'mouse_bite':      1,
    'open_circuit':    2,
    'short':           3,
    'spur':            4,
    'spurious_copper': 5,
}

# Subfolder names inside images/ and Annotations/
CLASS_FOLDERS = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper',
]


def parse_voc_xml(xml_path):
    """Parse a PASCAL VOC XML annotation file.

    Returns (img_w, img_h, boxes) where boxes is a list of
    (class_id, x_center, y_center, width, height) normalized to [0, 1].
    """
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        class_id = CLASS_NAME_TO_ID.get(name)
        if class_id is None:
            continue
        bb = obj.find('bndbox')
        xmin = int(bb.find('xmin').text)
        ymin = int(bb.find('ymin').text)
        xmax = int(bb.find('xmax').text)
        ymax = int(bb.find('ymax').text)
        xc = (xmin + xmax) / 2 / img_w
        yc = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        boxes.append((class_id, xc, yc, w, h))
    return img_w, img_h, boxes


def flip_boxes(boxes):
    """Horizontal flip: x_center → 1 - x_center. All other values unchanged."""
    return [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in boxes]


def write_label(dst_txt, boxes):
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in boxes]
    with open(dst_txt, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True,
                        help='Path to PCB_DATASET folder (must contain images/ and Annotations/)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write the converted YOLO dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    out = Path(args.output_dir)

    # Collect all image-annotation pairs by walking images/{ClassName}/ folders
    orig_pairs = []
    for class_folder in CLASS_FOLDERS:
        img_dir = dataset / 'images' / class_folder
        ann_dir = dataset / 'Annotations' / class_folder
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue
        for img_path in sorted(img_dir.glob('*.jpg')):
            xml_path = ann_dir / (img_path.stem + '.xml')
            if xml_path.exists():
                orig_pairs.append((img_path, xml_path))
            else:
                print(f"  WARNING: no annotation for {img_path.name}, skipping.")

    if not orig_pairs:
        print("\nERROR: No valid image-annotation pairs found.")
        print("Make sure --dataset_dir points to the PCB_DATASET/ folder")
        print("that contains images/ and Annotations/ subfolders.")
        return

    print(f"Found {len(orig_pairs)} image-annotation pairs")

    # Build augmented list: original + horizontal flip for each image.
    # Augment FIRST, then split — matching the paper's order of operations.
    augmented = []
    skipped = 0
    for img_path, xml_path in orig_pairs:
        _, _, boxes = parse_voc_xml(xml_path)
        if not boxes:
            skipped += 1
            continue
        augmented.append(('orig', img_path, boxes,             img_path.stem))
        augmented.append(('flip', img_path, flip_boxes(boxes), img_path.stem + '_flip'))

    if skipped:
        print(f"Skipped {skipped} images with no parseable annotations")
    print(f"Augmented dataset size (orig + flip): {len(augmented)}")
    print(f"  (Paper reports 1408; we produce {len(augmented)} = 693×2.")
    print(f"   The 22-image gap is from random crops the authors did not publish.)")

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
python convert_pcb.py \
    --dataset_dir /path/to/PCB_DATASET \
    --output_dir /path/to/pcb_dataset
```

---

### 3d. Expected folder structure after conversion

```
pcb_dataset/
├── images/
│   ├── train/     ← 1109 images (80% of 1386)
│   ├── val/       ← 138 images (10%)
│   └── test/      ← 139 images (10%)
└── labels/
    ├── train/     ← matching .txt label files (YOLO format)
    ├── val/
    └── test/
```

The script will print the exact counts when it runs. Starting from 693 original images, horizontal flip augmentation produces **1386 images** total (693 × 2), split 8:1:1.

> **Note on the paper's 1408 count:** The paper reports 1408 images after "flipping and random cropping." Horizontal flip alone gives 1386. The extra 22 images came from offline random cropping with parameters the original authors did not publish and cannot be recovered from the public dataset. The 1.6% difference is not expected to affect training outcomes. The `translate=0.1` setting already in `train.py` provides equivalent random-crop diversity online during training.

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
