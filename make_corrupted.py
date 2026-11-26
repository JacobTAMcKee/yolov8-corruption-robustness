from pathlib import Path

import cv2
import yaml

from corruptions import CORRUPTIONS

SRC = Path("data/subset")
OUT = Path("data/corrupted")

src_images = sorted((SRC / "images/val").glob("*.jpg"))
names = yaml.safe_load(open("subset.yaml"))["names"]

for cname, (fn, levels) in CORRUPTIONS.items():
    for level in levels:
        root = OUT / f"{cname}_{level}"
        img_dir = root / "images/val"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Labels are identical to the clean ones, so link instead of copying
        lbl_dir = root / "labels/val"
        lbl_dir.parent.mkdir(parents=True, exist_ok=True)
        if not lbl_dir.exists():
            lbl_dir.symlink_to((SRC / "labels/val").resolve())

        for p in src_images:
            img = cv2.imread(str(p))
            # PNG would avoid extra JPEG loss, but JPEG quality 95 is close enough
            cv2.imwrite(str(img_dir / p.name), fn(img, level),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        cfg = {"path": str(root.resolve()), "train": "images/val",
               "val": "images/val", "names": names}
        with open(root / "data.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"built {cname}_{level}")