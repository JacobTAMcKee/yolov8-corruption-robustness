import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from corruptions import CORRUPTIONS

MODELS = ["yolov8n", "yolov8s"]
Path("results").mkdir(exist_ok=True)

# Build the list of (corruption, level, yaml) to evaluate. Clean baseline goes first.
jobs = [("clean", 0, "subset.yaml")]
for cname, (fn, levels) in CORRUPTIONS.items():
    for level in levels:
        jobs.append((cname, level, f"data/corrupted/{cname}_{level}/data.yaml"))

rows = []
for model_name in MODELS:
    model = YOLO(f"{model_name}.pt")
    for cname, level, data_yaml in jobs:
        m = model.val(data=data_yaml, imgsz=640, batch=16, device=0,
                      plots=False, verbose=False, project="runs/exp", exist_ok=True)
        rows.append({
            "model": model_name,
            "corruption": cname,
            "level": level,
            "map50": m.box.map50,
            "map50_95": m.box.map,
            "precision": m.box.mp,
            "recall": m.box.mr,
        })
        print(f"{model_name:8s} {cname:6s} {level:>5}  "
              f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")

pd.DataFrame(rows).to_csv("results/results.csv", index=False)
print("Saved results/results.csv")