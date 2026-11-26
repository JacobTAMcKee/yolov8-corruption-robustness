import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

random.seed(0)  # fixed seed so the subset is reproducible
N_IMAGES = 500

src_images = Path("data/val2017")
src_labels = Path("data/coco/labels/val2017")
dst_root = Path("data/subset")
dst_images = dst_root / "images/val"
dst_labels = dst_root / "labels/val"
dst_images.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)

# Only keep images that have a label file (some COCO images have no objects)
all_images = sorted(p for p in src_images.glob("*.jpg")
                    if (src_labels / f"{p.stem}.txt").exists())
chosen = random.sample(all_images, N_IMAGES)

for img in chosen:
    shutil.copy(img, dst_images / img.name)
    shutil.copy(src_labels / f"{img.stem}.txt", dst_labels / f"{img.stem}.txt")
print(f"Copied {len(chosen)} images")

# Write the dataset config. Class names come from the pretrained model,
# which was trained on COCO, so they match the label indices.
names = YOLO("yolov8n.pt").names
config = {
    "path": str(dst_root.resolve()),  # absolute path, computed for whoever runs this
    "train": "images/val",            # unused, but Ultralytics requires the key
    "val": "images/val",
    "names": names,
}
with open("subset.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
print("Wrote subset.yaml")