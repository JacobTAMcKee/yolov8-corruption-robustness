from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from corruptions import CORRUPTIONS

Path("results").mkdir(exist_ok=True)
img = cv2.imread(sorted(Path("data/subset/images/val").glob("*.jpg"))[0])

fig, axes = plt.subplots(4, 4, figsize=(14, 11))
for row, (cname, (fn, levels)) in enumerate(CORRUPTIONS.items()):
    axes[row][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row][0].set_title("clean")
    for col, level in enumerate(levels, start=1):
        axes[row][col].imshow(cv2.cvtColor(fn(img, level), cv2.COLOR_BGR2RGB))
        axes[row][col].set_title(f"{cname} {level}")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/corruption_examples.png", dpi=120)