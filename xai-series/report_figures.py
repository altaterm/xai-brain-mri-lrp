"""Extra figures for the report template: dataset/loader evidence, case sheets and a duplicate check.

Reuses the code of 05_lrp_brain_mri.py by running only its setup, model-loading and XAI cells,
so the explanations come from exactly the same functions and the same trained checkpoint
(models/vgg16_brain_mri.pt from the Colab run). Run from the xai-series folder:

    ..\\.venv\\Scripts\\python.exe report_figures.py
"""
import datetime
import hashlib
import os
import subprocess
from pathlib import Path

SOURCE = Path("05_lrp_brain_mri.py").read_text(encoding="utf-8")
CELLS_TO_RUN = ("Imports", "Device and paths", "Plot style", "Load data", "Building the model",
                "Training strategy", "Train (skipped", "Layerwise relevance", "XAI method 2",
                "Helpers: brain mask")

namespace = {"IN_COLAB": False, "__name__": "case_figures"}
for cell in SOURCE.split("# %%")[1:]:
    if cell.strip().startswith(CELLS_TO_RUN):
        exec(compile("# %%" + cell, "05_lrp_brain_mri.py", "exec"), namespace)
globals().update(namespace)
assert MODEL_PATH.exists(), "Trained checkpoint missing: unzip it from the Colab output into models/"
model.eval()


def text_figure(lines, out_name, height_per_line=0.19):
    """Renders console-style text (produced live below) as a figure for the report."""
    fig = plt.figure(figsize=(9.6, 0.3 + height_per_line * len(lines)))
    fig.text(0.01, 0.98, "\n".join(lines), family="monospace", fontsize=9, va="top", color=INK,
             bbox=dict(facecolor="#f3f2ee", edgecolor=GRID, boxstyle="round,pad=0.6"))
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG_DIR / out_name)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


def count_images(folder):
    return sum(1 for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# Figure: dataset download and storage evidence.
RAW = Path("..") / "Brain-Tumor-Classification-DataSet"
cloned = datetime.datetime.fromtimestamp(os.stat(RAW / ".git").st_birthtime)
lines = [f"> git clone {git(RAW, 'remote', 'get-url', 'origin')}",
         f"  cloned on {cloned:%Y-%m-%d %H:%M} into {RAW.resolve().parent.name}/{RAW.name}/",
         f"> git log -1   ->  commit {git(RAW, 'rev-parse', 'HEAD')}",
         f"                   date   {git(RAW, 'log', '-1', '--format=%ci')}",
         "",
         "> image files per folder:   downloaded copy        ->  xai-series/data/brain_mri"]
for split in ("Training", "Testing"):
    for class_dir in sorted((RAW / split).iterdir()):
        local = Path("data/brain_mri") / split.lower() / class_dir.name
        lines.append(f"  {split + '/' + class_dir.name:30s} {count_images(class_dir):4d}  ->  "
                     f"{split.lower() + '/' + class_dir.name:30s} {count_images(local):4d}")
text_figure(lines, "13_dataset_download.png")

# Figure: the adapted loader reads both folders with the same class mapping.
lines = [f"TRAIN_ROOT = {TRAIN_ROOT!r}   ->  {len(train_full)} images",
         f"TEST_ROOT  = {TEST_ROOT!r}    ->  {len(test_dataset)} images",
         f"train class_to_idx = {train_full.class_to_idx}",
         f"test  class_to_idx = {test_dataset.class_to_idx}",
         f"mappings identical: {train_full.class_to_idx == test_dataset.class_to_idx}",
         "",
         "stratified split of the training folder (seed 42) and the untouched test folder:"]
lines += ["  " + line for line in counts.to_string().splitlines()]
text_figure(lines, "14_dataset_loading.png")


def class_probability(image, target):
    with torch.no_grad():
        return float(F.softmax(model(image.unsqueeze(0).to(device)), 1)[0, target])


def case_row(axes, image, gray, target, label):
    relevance, _, logit = apply_lrp_on_vgg16(image, target=target)
    cam, _, _ = grad_cam(image, target=target)
    axes[0].imshow(gray, cmap="gray")
    limit = np.percentile(np.abs(relevance), 99.5) + 1e-12
    axes[1].imshow(gray, cmap="gray")
    lrp_image = axes[1].imshow(relevance, cmap=LRP_CMAP, vmin=-limit, vmax=limit, alpha=0.75)
    axes[2].imshow(gray, cmap="gray")
    cam_image = axes[2].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.55)
    axes[0].set_ylabel(f"target: {NAMES[target]}\nlogit {logit:.2f}, p = {class_probability(image, target):.3f}",
                       color=INK)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    bar = plt.colorbar(lrp_image, ax=axes[1], fraction=0.046, pad=0.03, ticks=[-limit, 0, limit])
    bar.ax.set_yticklabels(["against", "0", "for"])
    bar = plt.colorbar(cam_image, ax=axes[2], fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0", "0.5", "max"])


def case_figure(file_name, true_class, targets, out_name):
    folder = Path(TEST_ROOT) / CLASSES[NAMES.index(true_class)]
    idx = next(i for i, (path, _) in enumerate(test_dataset.samples) if Path(path) == folder / file_name)
    image, label = test_dataset[idx]
    gray = to_gray_uint8(image)
    fig, axes = plt.subplots(len(targets), 3, figsize=(9.6, 3.1 * len(targets)), squeeze=False)
    for row, target in zip(axes, targets):
        case_row(row, image, gray, NAMES.index(target), NAMES[label])
    for ax, title in zip(axes[0], ["A  Original MRI", "B  LRP relevance (signed)", "C  Grad-CAM (conv5_3)"]):
        ax.set_title(title)
    fig.suptitle(f"testing/{CLASSES[label]}/{file_name}  |  dataset label: {NAMES[label]}", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG_DIR / out_name, "| test index", idx)


case_figure("image(95).jpg", "Meningioma", ["Meningioma"], "11_case_A_meningioma.png")
case_figure("image(18).jpg", "Meningioma", ["No tumour", "Meningioma"], "12_case_B_missed_meningioma.png")

# Exact-duplicate check between the training and testing folders (byte-identical files).
def digests(root):
    return {hashlib.md5(p.read_bytes()).hexdigest() for p in Path(root).rglob("*.jpg")}

train_hashes, test_hashes = digests(TRAIN_ROOT), digests(TEST_ROOT)
print(f"Unique training files {len(train_hashes)}, unique testing files {len(test_hashes)}, "
      f"byte-identical files in both: {len(train_hashes & test_hashes)}")

# How many test images have a byte-identical copy in training, and accuracy with / without them.
per_image = pd.read_csv(RESULTS_DIR / "per_image_results.csv")
test_paths = [Path(p) for p, _ in test_dataset.samples]
per_image["in_training"] = [hashlib.md5(test_paths[i].read_bytes()).hexdigest() in train_hashes
                            for i in per_image["index"]]
overlap = per_image[per_image.in_training]
rest = per_image[~per_image.in_training]
print(f"Test images with a copy in training: {len(overlap)} "
      f"(accuracy {overlap.correct.mean():.3f}); without: {len(rest)} (accuracy {rest.correct.mean():.3f})")
print(overlap.groupby("true").size().to_dict())
