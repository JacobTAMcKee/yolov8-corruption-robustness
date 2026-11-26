#!/usr/bin/env bash
# One-time setup: environment, COCO val2017 download, 500-image subset.
# Usage: bash setup.sh
set -e

# 1. Virtual environment and libraries
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download COCO val2017 images and YOLO-format labels
mkdir -p data
cd data
if [ ! -d val2017 ]; then
    wget -nc http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
fi
if [ ! -d coco/labels/val2017 ]; then
    wget -nc https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
    unzip -q coco2017labels.zip
fi
cd ..

# 3. Build the subset and subset.yaml
python make_subset.py

echo "Setup complete. Activate the environment with: source venv/bin/activate"