# Robustness of YOLOv8 to Image Corruptions (COMP 4102)

## Setup (Linux / WSL2, Python 3.9+, NVIDIA GPU recommended)
    bash setup.sh
    source venv/bin/activate

This downloads COCO val2017 (~1 GB), builds a fixed 500-image subset (seed 0),
and writes subset.yaml.

## Running
    python eval_clean.py        # clean baseline for yolov8n and yolov8s
    (more scripts to be added)

## Data and models
- COCO val2017 (Lin et al., 2014)
- Pretrained YOLOv8 weights from Ultralytics (downloaded automatically)