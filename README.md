# Student Stress Prediction — Model Card

Hybrid K-Means + per-cluster classification pipeline for predicting academic stress level from psychometric survey data.

---

## Dataset

| Field | Value |
|---|---|
| Name | StressLevelDataset (rxnach) |
| Source | Kaggle — psychometric survey of university students |
| Rows | 1,100 |
| Features | 20 psychometric scales (anxiety, depression, sleep quality, social support, etc.) |
| Primary target | `stress_level` — Low / Medium / High (real survey label) |
| Auxiliary target | `burnout` — **SYNTHETIC** (derived via formula from depression + anxiety + stress_load) |

> **Burnout caveat:** The burnout label is not real ground-truth. It is a formula-derived auxiliary target included only to demonstrate multi-target capability. Do not compare burnout metrics with stress-level metrics or treat them as independently validated results.

If the CSV is absent at `data/raw/StressLevelDataset.csv`, the loader generates a 2,000-row synthetic dataset and all output CSVs are tagged `data_source=synthetic`.

---

## Preprocessing

| Step | Where fitted | Applied to |
|---|---|---|
| Duplicate removal | Full dataset (stateless) | Full dataset |
| Label encoding (targets) | Full dataset (stateless) | Full dataset |
| Mean imputation | **Training set only** | Train + test |
| IQR outlier capping (1.5x) | **Training set only** | Train + test |
| MinMax scaling | **Training set only** | Train + test |

All statistics-fitting steps happen after the global train/test split to prevent data leakage.

---

## Feature Engineering

Five composite features derived from the raw psychometric scales:

| Feature | Formula |
|---|---|
| `psychological_burden` | `anxiety x 0.35 + depression x 0.35 - self_esteem x 0.30` |
| `academic_stress` | `study_load x 0.40 + peer_pressure x 0.30 - academic_performance x 0.30` |
| `wellbeing_index` | `sleep_quality + social_support + safety + basic_needs - headache - breathing_problem` |
| `environmental_stress` | `noise x 0.40 + bullying x 0.40 - living_conditions x 0.20` |
| `support_deficit` | `peer_pressure + future_career_concerns - social_support - teacher_student_relationship` |

---

## Architecture

    Raw CSV
      data_loader: schema normalisation
      preprocess_data: dedup + label encoding
      create_features: 5 composite features
      train_test_split (80/20 stratified, random_state=42)
           fit_transform_cleaning  [train fit]
           apply_encoding_and_scaling  [train fit]
           K-Means (K selected by silhouette, range 2-6)  [train fit]
                Cluster 0..K-1: per-cluster best classifier (RF / XGBoost / LR / Trivial)
                HybridModel: route test samples by cluster -> specialist prediction

---

## Evaluation Protocol

| Aspect | Choice | Rationale |
|---|---|---|
| Train/test split | 80/20 stratified | Single global split; all models on same held-out set |
| Cross-validation | 5x3 repeated stratified K-fold (15 folds) | More stable than single 5-fold |
| Significance test | McNemar's test with Yates correction | Paired test for same-sample comparison |
| Primary metric | Macro F1 | Equal class weight; robust to mild imbalance |
| Secondary metrics | Accuracy, balanced accuracy, per-class P/R/F1 | Full picture |

---

## Models Compared

| Model | Description |
|---|---|
| **Hybrid** | KMeans routing + per-cluster specialist (RF / XGBoost / LR / TrivialPredictor) |
| **Baseline RF** | Random Forest (100 trees) on full feature set |
| **XGBoost_tuned** | XGBoost with RandomizedSearchCV (n_iter=15, cv=3) over depth / lr / subsample |
| **LogisticReg** | Logistic Regression (max_iter=2000) on full feature set — linear baseline |

---

## Known Limitations

- **Burnout is synthetic** — burnout metrics reflect formula recovery, not clinical validity.
- **Small dataset** — 1,100 rows limits statistical power; McNemar may not reach significance even when a real difference exists.
- **Cluster purity effect** — two of four K-Means clusters are pure (single stress class). Hybrid gains on those clusters come from trivial routing, not learned boundaries. See `ablation_cluster_breakdown.csv`.
- **Synthetic fallback** — when the real CSV is absent, results are from generated data and are not representative.
- **No temporal validation** — all data is cross-sectional; no longitudinal predictions.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/results/model_comparison.csv` | Accuracy / balanced acc / macro F1 / CV F1 for all 4 models |
| `outputs/results/cluster_diagnostics.csv` | Per-cluster composition, routing, internal performance |
| `outputs/results/ablation_cluster_breakdown.csv` | Per-cluster accuracy: hybrid vs every baseline |
| `outputs/plots/` | All PNG visualisations (EDA, clustering, confusion matrices, importance) |
| `outputs/models/` | Saved .pkl files for all trained models |

---

## Reproducing Results

    pip install -r requirements.txt

    # Place the dataset at:  data/raw/StressLevelDataset.csv
    # (omit to use synthetic data automatically)

    python main.py

    jupyter notebook notebooks/eda.ipynb
