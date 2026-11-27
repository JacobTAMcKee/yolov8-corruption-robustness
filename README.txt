# Robustness of YOLOv8 to Image Corruptions

COMP 4102 Final Project

We measure how much the accuracy of pretrained YOLOv8 object detectors (nano and
small) drops when COCO images are corrupted with Gaussian blur, Gaussian noise,
darkening, or JPEG compression (three severity levels each). Nothing is trained:
we evaluate the pretrained models on a fixed 500-image subset of COCO val2017
and look at failure cases. See the report for the method and results.

## Requirements

- Linux or WSL2 (Ubuntu), Python 3.9+ (we used Python 3.10)
- An NVIDIA GPU with working drivers. The scripts use `device=0`
  (we used an RTX 3070 with 8 GB)
- Internet access on the first run: COCO val2017 (about 1 GB) and the pretrained
  YOLOv8 weights are downloaded automatically
- Several GB of free disk space (PyTorch with CUDA libraries, the COCO images,
  and the 12 corrupted copies)
- `python3-venv`, `unzip`, and `wget`. On a fresh Ubuntu:
  `sudo apt update && sudo apt install -y python3-venv unzip wget`

Run all commands from the project folder.

## Run everything

```bash
bash run_all.sh
```

This sets up the environment if needed, downloads the data, and runs every step
in the table below. It took about [X] minutes on our machine, and each step
prints its own time. To skip the slowest step (the failure-case figures):

```bash
bash run_all.sh --skip-failures
```

Results are written to `results/`.

## What each step does

| Script | Purpose | Output |
|---|---|---|
| `setup.sh` | Creates `venv/`, installs `requirements.txt`, downloads COCO val2017 and labels, calls `make_subset.py` | `venv/`, `data/` |
| `make_subset.py` | Random 500-image subset (seed 0) and its dataset config | `data/subset/`, `subset.yaml` |
| `corruptions.py` | The four corruption functions and severity levels (imported by other scripts) | none |
| `make_corrupted.py` | Applies each corruption at each level to the subset | `data/corrupted/` (12 folders) |
| `check_corruptions.py` | Example image under every corruption and level | `results/corruption_examples.png` |
| `run_experiments.py` | Evaluates yolov8n and yolov8s on the clean set and the 12 corrupted sets | `results/results.csv` |
| `make_plots.py` | Relative drops and mAP-versus-severity plots | `results/summary.csv`, `results/map_vs_severity.png`, `results/relative_map_vs_severity.png` |
| `find_failures.py` | Clean vs. corrupted predictions for the images with the largest change in detections | `results/failures/*.jpg` |
| `eval_clean.py` | Optional quick check: clean-image mAP only (yolov8s should give about 0.457 mAP50-95) | console output |

## Running steps individually

```bash
bash setup.sh                  # once
source venv/bin/activate
python eval_clean.py           # quick sanity check
python make_corrupted.py
python run_experiments.py
python make_plots.py
```

## Reproducibility notes

- The subset uses a fixed seed (0), and noise is seeded from each image, so the
  corrupted sets can be rebuilt exactly.
- `requirements.txt` lists the libraries, and `requirements-lock.txt` records the
  exact versions we used.
- Corrupted images are saved as JPEG at quality 95 so that Ultralytics can read
  them from folders. This adds a small extra compression to every corrupted image.
- In `find_failures.py`, images are ranked in a batched pass and drawn in a
  single-image pass, so the counts printed in the figure titles can differ by a
  few detections from the ranking. The report quotes the numbers in the figures.
- `data/`, `venv/`, `runs/`, and model weights are not included in the submission.
  `setup.sh` rebuilds the data, and Ultralytics downloads the weights.

## Troubleshooting

- "No CUDA GPU found": check that `nvidia-smi` works. On WSL, the NVIDIA driver
  must be installed on the Windows side and you must be using WSL2. To run on a
  CPU, change `device=0` to `device="cpu"` in `run_experiments.py` and
  `find_failures.py` (this is much slower).
- `setup.sh` fails on `venv`, `unzip`, or `wget`: install them with the `apt`
  command above.
- All scripts can be rerun safely and overwrite their outputs.

## Data, models, and libraries

- COCO val2017 images and labels: Lin et al., 2014 (cocodataset.org). YOLO-format
  labels from the Ultralytics assets release.
- YOLOv8n and YOLOv8s pretrained weights and the validation routine: Ultralytics
  (github.com/ultralytics/ultralytics)
- OpenCV, NumPy, pandas, matplotlib