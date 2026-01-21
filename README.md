# Pokemon Prediction Model (Two-Part Project)

This repo contains two Jupyter notebooks for image-based Pokemon type prediction:
1) a multiclass image classifier, and
2) a multi-label model that predicts primary and secondary types.

## Project Overview
### Part 1: Multiclass Image Classification (CNN)
Notebook: `Pokemon_Final NN project - Multiclass.ipynb`
- End-to-end pipeline: data audit, EDA, train/val/test split, class imbalance handling.
- `tf.data` input pipeline and visual sanity checks.
- Baseline CNNs, model comparisons, and transfer learning (EfficientNetB0, MobileNetV3Large).
- Hyperparameter tuning, best-model selection, final test evaluation, and error analysis.
- Confusion matrix visualization and qualitative predictions.

### Part 2: Multi-Label Type Prediction
Notebook: `Pokemon_Final NN project - Mulilabel.ipynb`
- Goal: predict Type1 and Type2 from an image (two labels max).
- Data preprocessing and preparation.
- Model training with learning-curve visualization.
- Result inspection and tests on new images.

## Dataset
Both notebooks use the Kaggle dataset:
- `vishalsubbiah/pokemon-images-and-types`
- Images of Pokemon (gen 1-7) and CSV with primary/secondary types.

The dataset is downloaded inside the notebooks (via `kagglehub`) or configured by local paths.

## Repo Structure
- `Pokemon_Final NN project - Multiclass.ipynb`
- `Pokemon_Final NN project - Mulilabel.ipynb`
- `*.pdf` (project report)

## Setup
Recommended: Python 3.10+ with a virtual environment.

Install core dependencies:
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib pillow kagglehub
```

Optional: use a GPU-enabled TensorFlow build if you have CUDA installed.

## How To Run
1) Launch Jupyter:
```bash
jupyter notebook
```
2) Open and run the notebooks in order:
   - `Pokemon_Final NN project - Multiclass.ipynb`
   - `Pokemon_Final NN project - Mulilabel.ipynb`

Notes:
- The notebooks write intermediate CSVs and artifacts (splits, logs, models) in their output directories.
- If running outside Kaggle/Colab, update dataset paths to your local folders.

## Results
Each notebook includes its own evaluation outputs (metrics, plots, confusion matrix, and sample predictions).

