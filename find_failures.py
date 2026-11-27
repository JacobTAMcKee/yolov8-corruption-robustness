import time
from pathlib import Path

import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
CONF = 0.25                                   # a detection counts if confidence >= 0.25
CASES = ["blur_21", "noise_50", "dark_0.2", "jpeg_5"]
N_LOST = 3                                    # images with the most lost detections, per case
CLEAN_DIR = Path("data/subset/images/val")
OUT = Path("results/failures")
OUT.mkdir(parents=True, exist_ok=True)

images = sorted(CLEAN_DIR.glob("*.jpg"))


def count_detections(paths):
    """Return the number of detections for each image, keeping nothing else in memory."""
    counts = []
    for r in model.predict([str(p) for p in paths], conf=CONF, imgsz=640,
                           device=0, batch=16, verbose=False, stream=True):
        counts.append(len(r.boxes))
    return counts


def draw(path, title):
    """Run the model on one image and return it with boxes drawn and a title banner above."""
    r = model.predict(str(path), conf=CONF, imgsz=640, device=0, verbose=False)[0]
    img = r.plot()
    # Add a 36-pixel black strip above the image instead of drawing over it
    img = cv2.copyMakeBorder(img, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(img, f"{title}: {len(r.boxes)} detections", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def save_pair(clean_path, corr_path, case, kind):
    pair = cv2.hconcat([draw(clean_path, "clean"), draw(corr_path, case)])
    cv2.imwrite(str(OUT / f"{case}_{kind}_{clean_path.stem}.jpg"), pair)


t0 = time.time()
clean_counts = count_detections(images)
print(f"clean pass done in {time.time() - t0:.0f}s")

for case in CASES:
    t0 = time.time()
    corr_dir = Path(f"data/corrupted/{case}/images/val")
    corr_paths = [corr_dir / p.name for p in images]
    corr_counts = count_detections(corr_paths)
    print(f"{case}: counted in {time.time() - t0:.0f}s")

    # change = detections lost (positive) or gained (negative) due to corruption
    changes = sorted(((c - k, i) for i, (c, k) in enumerate(zip(clean_counts, corr_counts))),
                     reverse=True)

    for drop, i in changes[:N_LOST]:                       # most detections lost
        save_pair(images[i], corr_paths[i], case, "lost")
        print(f"  lost    {images[i].name}: -{drop}")

    gain, i = changes[-1]                                  # most detections gained
    if gain < 0:
        save_pair(images[i], corr_paths[i], case, "gained")
        print(f"  gained  {images[i].name}: +{-gain}")