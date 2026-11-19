# Assignment 2: XAI on MRI Data

**Explaining a brain-tumour MRI classifier with Layer-wise Relevance Propagation (LRP) and Grad-CAM**

Mohaned Altater (26068926), Explainable AI

A VGG16 network, fine-tuned on four MRI classes (glioma, meningioma, no tumour, pituitary)
following `05_lrp.py` from [deepfindr/xai-series](https://github.com/deepfindr/xai-series),
is explained with LRP and Grad-CAM. The explanations are then evaluated quantitatively
(deletion test, region relevance score, LRP vs Grad-CAM agreement) and used to explain the
diagnosis to a patient and the model to a doctor.

## Where to find each part of the assignment

| Task | Where |
|---|---|
| Task 1: download the dataset | `dataset_counts.csv`; notebook cell 2 (clones the dataset and prints the counts) |
| Task 2: dataset under `xai-series/data/brain_mri` | `xai-series/data/brain_mri/training/` (2,870 images) and `testing/` (394 images) |
| Task 3: LRP + second XAI method (Grad-CAM) | `xai-series/05_lrp_brain_mri.py` (code), `05_lrp_brain_mri_colab_v2.ipynb` (executed notebook with all outputs), `xai-series/figures/` |
| Task 4: doctor to patient, data scientist to doctor | Report, Sections 5 and 6 |
| Bonus: LRP vs Grad-CAM, false positives/negatives, explanation metrics | Report, Section 4; `xai-series/results/summary.json` |

**Report:** `Assignment2_XAI_Report_Mohaned_Altater.docx`

## Repository layout

```
├── Assignment2_XAI_Report_Mohaned_Altater.docx   report
├── 05_lrp_brain_mri_colab_v2.ipynb               executed notebook (Google Colab, T4 GPU)
├── dataset_counts.csv                            Task 1 evidence: images per split and class
├── requirements.txt
├── Assignment 2-4.md, Rubric Assignment 2.md     assignment brief and rubric
└── xai-series/                                   deepfindr/xai-series (cloned)
    ├── 05_lrp.py                                 original reference code (unchanged)
    ├── 05_lrp_brain_mri.py                       this assignment's code (# %% cells)
    ├── data/brain_mri/{training,testing}/        Task 2: the MRI dataset
    ├── figures/                                  all figures used in the report
    ├── results/                                  summary.json, per_image_results.csv, training_history.json
    └── 00-04_*.py, utils.py, data/*.csv          other files of the original repository (not used)
```

The notebook is generated from `05_lrp_brain_mri.py` with jupytext; both contain the same code.
Every change to the original `05_lrp.py` is marked with a `# CHANGED:` comment.

## Run on Google Colab (recommended)

1. Open <https://colab.research.google.com>, choose **File > Upload notebook** and select `05_lrp_brain_mri_colab_v2.ipynb`.
2. Check that **Runtime > Change runtime type** is set to **T4 GPU**.
3. Choose **Runtime > Run all**. The notebook clones the dataset and xai-series, fine-tunes all
   VGG16 layers (about 5 minutes), runs LRP, Grad-CAM and the metrics, and downloads
   `brain_xai_outputs.zip` with the figures, results and trained weights.

## Run locally (CPU)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd xai-series
..\.venv\Scripts\python.exe 05_lrp_brain_mri.py
```

Without a GPU the script fine-tunes only the last convolutional layer and the classifier
(full fine-tuning takes many hours on a CPU). If `xai-series/models/vgg16_brain_mri.pt`
exists (for example, unzipped from the Colab output), training is skipped and the trained model is loaded.

For a quick pipeline check on a small subset, set `$env:QUICK="1"` before running.

## Key results (test set, 394 images)

| Metric | Value |
|---|---|
| Validation / test accuracy | 94.8% / 75.1% |
| Tumour vs healthy: sensitivity / specificity | 84.8% / 100% |
| Deletion AUC, lower is better (LRP / Grad-CAM / random) | 0.378 / 0.391 / 0.453 |
| Share of positive evidence inside the head (LRP / Grad-CAM / head area) | 0.64 / 0.59 / 0.57 |

## Sources

- Dataset: Bhuvaji et al. (2020), [SartajBhuvaji/Brain-Tumor-Classification-DataSet](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet) (commit 3ea9f4e). Included here for coursework reproducibility; all credit to the original authors.
- Base code: [deepfindr/xai-series](https://github.com/deepfindr/xai-series), `05_lrp.py`.
