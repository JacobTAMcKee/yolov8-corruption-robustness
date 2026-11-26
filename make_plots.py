import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")
MODELS = ["yolov8n", "yolov8s"]
CORRUPTIONS = ["blur", "noise", "dark", "jpeg"]
XLABEL = {"blur": "Gaussian kernel size", "noise": "noise sigma",
          "dark": "brightness factor", "jpeg": "JPEG quality"}

# Clean score per model, used to compute relative drops
clean = df[df.corruption == "clean"].set_index("model")["map50_95"]
df["rel_drop_pct"] = 100 * (1 - df["map50_95"] / df["model"].map(clean))

# Summary table for the report (rows are in severity order, so don't sort)
df.to_csv("results/summary.csv", index=False)
print(df[["model", "corruption", "level", "map50_95", "rel_drop_pct"]]
      .round(3).to_string(index=False))


def make_figure(relative, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, c in zip(axes.ravel(), CORRUPTIONS):
        for m in MODELS:
            sub = df[(df.model == m) & (df.corruption == c)]
            y = [clean[m]] + list(sub.map50_95)
            if relative:
                y = [100 * v / clean[m] for v in y]
            ax.plot(range(4), y, marker="o", label=m)
        levels = list(df[(df.model == MODELS[0]) & (df.corruption == c)].level)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["clean"] + [f"{l:g}" for l in levels])
        ax.set_xlabel(XLABEL[c] + " (increasing severity →)")
        ax.set_ylabel("% of clean mAP50-95" if relative else "mAP50-95")
        ax.set_title(c)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


make_figure(relative=False, fname="results/map_vs_severity.png")
make_figure(relative=True, fname="results/relative_map_vs_severity.png")