# AutoGluon-SHAP Submarine Landslide Susceptibility Model

Official Python implementation of the **AutoGluon-based stacking ensemble** and **SHAP interpretability framework** for submarine landslide susceptibility assessment in the Qiongdongnan Basin, South China Sea.

This repository provides the complete modeling pipeline described in the associated manuscript, including two-step feature selection (IV + correlation), AutoGluon stacking ensemble training, SHAP-based global importance and interaction analysis, and susceptibility mapping utilities.

## 1. Overview

Submarine landslides in deep-water canyon systems are controlled by multiple interacting factors whose nonlinear responses and conditional dependencies remain poorly quantified. This repository implements a data-driven workflow that combines automated machine learning with explainable AI to address this gap.

The framework consists of three main stages:

- **Feature screening:** Information Value (IV) filtering followed by Pearson correlation-based multicollinearity diagnosis.
- **Stacking ensemble modeling:** AutoGluon constructs a two-layer stacking ensemble using seven base learners (GBM, XGB, CAT, RF, XT, KNN, and a linear model) with automated hyperparameter optimization and multi-level stacking.
- **SHAP interpretability:** KernelExplainer is used to quantify global feature importance, single-factor nonlinear response patterns, and pairwise interaction effects.

### Key features

- Two-step feature selection combining IV thresholding and correlation filtering.
- AutoGluon automated stacking ensemble with configurable base learners and stacking depth.
- SHAP-based global importance ranking and approximate interaction analysis.
- Susceptibility probability mapping with natural-break classification support.
- Modular CLI interface for training, prediction, and interpretation.

## 2. Performance

The AutoGluon stacking ensemble achieved the following performance on the complete experimental dataset used in the study (test set, 30% hold-out):

| Metric | Value |
|---|---:|
| AUC | 0.9107 |
| Accuracy | 0.8363 |
| F1-score | 0.6401 |
| Precision | 0.5222 |
| Recall | 0.8268 |

The model outperformed nine benchmark classifiers (LR, GNB, RF, GBDT, XGB, LGBM, CAT, SVM, MLP). Five-fold cross-validation yielded a mean AUC of **0.913 ± 0.007**.

**Important:** These metrics were obtained using the complete experimental dataset described in the associated manuscript. The repository does not include the original survey data (see Section 8). The reported values cannot be reproduced using only the publicly released code without the full dataset.

## 3. Repository Structure

```text
submarine-landslide-susceptibility-model/
│
├── AutoGluon-SHAP Landslide Susceptibility Model.py
├── requirements.txt
├── LICENSE
└── README.md
```

The main script implements the complete workflow:

| Component | Description |
|---|---|
| `TwoStepFeatureSelector` | IV filtering + correlation-based feature selection |
| `train_autogluon` | AutoGluon stacking ensemble training and evaluation |
| `analyze_shap` | SHAP global importance and interaction approximation |
| `save_predictions` | Susceptibility probability export |
| `main` | CLI entry point with argument parsing |

## 4. Environment and Installation

### Requirements

- Python >= 3.8
- AutoGluon >= 1.0
- SHAP >= 0.44
- scikit-learn >= 1.0
- pandas, numpy, openpyxl, xlrd

### Installation

Clone the repository:

```bash
git clone https://github.com/JiangQilin911/submarine-landslide-susceptibility-model.git
cd submarine-landslide-susceptibility-model
```

Create and activate a virtual environment (optional but recommended):

```bash
conda create -n ag_shap_landslide python=3.10 -y
conda activate ag_shap_landslide
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install AutoGluon separately if not included in `requirements.txt`:

```bash
pip install autogluon
```

## 5. Data Availability and Preparation

### 5.1 Data availability

The original multibeam bathymetric data, 3D seismic volume, sediment core measurements, and earthquake catalog used in this study are subject to commercial licensing and survey-data restrictions and therefore cannot be publicly redistributed.

The source code is released to ensure methodological transparency and reproducibility of the modeling workflow. Users are expected to prepare their own input data following the format described below.

### 5.2 Input data format

The model accepts CSV or Excel files with the following structure:

- **First three columns:** spatial identifiers (e.g., longitude, latitude, grid ID) — preserved in the output.
- **Feature columns:** Chinese-named conditioning factors as listed below.
- **Target column:** binary label (default column name: `label`), where `1` = landslide, `0` = non-landslide.

The expected feature columns (Chinese names) are:

| Category | Feature (Chinese) | English equivalent |
|---|---|---|
| Geomorphology | 水深 | Water depth |
| | 坡度 | Slope gradient |
| | 坡向 | Aspect |
| | 曲率 | Curvature |
| | 地形起伏度 | Topographic relief |
| | 地面粗糙度 | Terrain roughness |
| | 距水道距离 | Distance to channels |
| Geological / seismic | 距断层距离 | Distance to faults |
| | 地震影响强度 | Peak ground acceleration (PGA) |
| | 相干性 | Seismic coherence |
| | 浅层气风险 | Shallow-gas risk |
| Sediment | 天然含水量 | Natural water content |
| | 塑性指数 | Plasticity index |
| | 湿容重 | Wet bulk unit weight |
| | 细粒含量 | Fine-grain content |

The script automatically maps Chinese column names to English equivalents for SHAP output.

### 5.3 Preprocessing notes

- Categorical/ordinal factors should be encoded as integer classes consistent with the reclassification scheme in the manuscript (Table 1).
- Missing values are not imputed by the script; ensure the input table is complete.
- For the Qiongdongnan Basin dataset, the positive-to-negative ratio was controlled during stratified sampling (2135 positive, 10000 negative). Users should consider class imbalance when preparing their own data.
- The two-step feature selector is fitted on the training split only and then applied to the test and full datasets to avoid data leakage.

## 6. Running the Code

The script uses `argparse` for command-line configuration.

```bash
python "AutoGluon-SHAP Landslide Susceptibility Model.py" \
    --input path/to/data.csv \
    --output path/to/output_dir \
    --target label \
    --iv_threshold 0.02 \
    --corr_threshold 0.8 \
    --seed 42
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Input CSV/Excel file path |
| `--output` | Yes | — | Output directory or CSV file path |
| `--target` | No | `label` | Target column name |
| `--iv_threshold` | No | `0.02` | IV threshold for feature removal |
| `--corr_threshold` | No | `0.8` | Absolute correlation threshold |
| `--seed` | No | `42` | Random seed |

### Outputs

The script produces the following files in the output directory:

| File | Description |
|---|---|
| `Susceptibility_Results.csv` | Original first three columns + predicted landslide probability |
| `SHAP_Feature_Importance.csv` | Mean absolute SHAP value per feature |
| `SHAP_Interaction_Intensities.csv` | Pairwise interaction intensity matrix (Top-5 features) |
| `autogluon_model/` | Trained AutoGluon predictor directory |

### Workflow summary

1. Data loading and train/test split (70:30, stratified).
2. Two-step feature selection on the training set.
3. AutoGluon stacking ensemble training with `best_quality` presets, `auto_stack=True`, and `num_stack_levels=2`.
4. Test-set evaluation and optimal threshold determination (Youden index).
5. SHAP KernelExplainer analysis on a stratified sample of the training set.
6. Full-dataset susceptibility prediction and export.

**Note:** The default `time_limit` for AutoGluon training is 1800 seconds (30 minutes). Adjust `AUTOGLUON_CONFIG["time_limit"]` in the script for faster or more thorough training.

## 7. Custom Dataset

Users can adapt the script to their own study area by satisfying the following conditions:

1. Input table contains the first three spatial identifier columns, followed by feature columns with the Chinese names listed in Section 5.2 (or modify `original_feature_names_cn` in the script).
2. The target column contains binary labels (`0`/`1`).
3. Feature values are numeric and consistent with the reclassification scheme (ordinal integers for categorical factors).
4. Train/test splitting is performed before feature selection and model training (already implemented in the script).
5. For SHAP analysis, the background sample size and `nsamples` parameter can be adjusted via `SHAP_CONFIG`.

To use English column names directly, modify the `original_feature_names_cn` list and `FEATURE_NAMES_MAP` dictionary in the script.

## 8. Reproducibility and Data Limitations

The source code is publicly available to support inspection, methodological verification, and adaptation to other deep-water continental margin settings.

However, the complete original survey data cannot be redistributed because of commercial licensing and confidentiality agreements with data providers. Consequently:

- The repository does not contain the full experimental dataset used in the manuscript.
- The reported AUC (0.9107) and other metrics cannot be reproduced without the original data.
- The feature selection thresholds, hyperparameter configurations, and SHAP settings are provided as-is and reflect the experimental setup described in the manuscript.

The code is intended to demonstrate the complete workflow and to facilitate application to comparable datasets in other deep-water canyon systems.

## 9. Citation

If you use this code in your research, please cite the associated manuscript:

> Jiang, Q., Liu, X., Tao, H., Zhang, H., Meng, X., Wang, Y. Nonlinear responses and conditional interactions of submarine landslide susceptibility in the Qiongdongnan Basin. *Engineering Geology*. [Publication details to be added after publication.]

## 10. License

This project is distributed under the license specified in the `LICENSE` file.

Please refer to that file for the terms and conditions governing the use, modification, and distribution of the source code.

The license for the source code does not grant permission to redistribute the original survey data or any derived data products. Users should independently verify applicable data-use restrictions before redistributing or reusing the data.

## 11. Contact

For questions about the model implementation or code, please open an issue in this GitHub repository:

https://github.com/JiangQilin911/submarine-landslide-susceptibility-model

For questions about the manuscript or data access, please contact the corresponding author:

- **Xiaolei Liu** — xiaolei@ouc.edu.cn
- **Qilin Jiang** — (GitHub: JiangQilin911)
