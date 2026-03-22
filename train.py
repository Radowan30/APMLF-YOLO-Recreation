"""APMLF-YOLO Training Script.

Paper hyperparameters (Section 4):
  Optimizer:    SGD (NOT Ultralytics default AdamW)
  LR:           0.01
  Momentum:     0.937
  Weight decay: 0.0005
  Batch size:   4
  Epochs:       200
  Image size:   640x640
  Augmentation: horizontal flip + random crop ONLY
                (mosaic=0, mixup=0 -- paper does NOT use these)

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics'))

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
            'model': 'ultralytics/ultralytics/cfg/models/v8/apmlf_yolo.yaml',
            'data': args.data,

            # Paper Section 4 hyperparameters -- DO NOT change without paper justification
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
            'mosaic': 0.0,           # DISABLED -- paper does not use mosaic
            'mixup': 0.0,            # DISABLED -- paper does not use mixup
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,        # slight translate ~= random crop effect
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
