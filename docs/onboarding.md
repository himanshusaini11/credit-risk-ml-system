# Onboarding — Credit Risk ML System

## Prerequisites

- Python 3.11
- `uv` (package manager) — install via `curl -Lsf https://astral.sh/uv/install.sh | sh`
- Docker (optional, for containerized serving)
- Git

---

## Setup From Scratch

### 1. Clone the repository
```bash
git clone https://github.com/himanshusaini11/credit-risk-ml-system.git
cd credit-risk-ml-system
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Get the dataset

Download the Bank Loan Defaulter Hackathon dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/ankitkalauni/bank-loan-defaulter-prediction-hackathon
- Files needed: `train.csv`, `test.csv`
- Place them in `data/` directory

```bash
mkdir -p data/
# Copy downloaded CSVs here
```

### 4. Configure the pipeline
```bash
# Review and adjust if needed
cat configs/default.yaml
```

Key settings to verify:
- `data.path`: points to your `train.csv`
- `model.type`: set to `random_forest` (stable) or `logistic_regression`
  (`lightgbm` is pending integration — do not set it yet)
- `split.train_fraction / val_fraction / test_fraction`: default 0.70/0.15/0.15

---

## Running the Pipeline

### Full training run
```bash
python -m creditrisk.cli train --config configs/default.yaml
```

This will:
1. Load and validate data
2. Split into train/val/test (stratified, seeded)
3. Fit preprocessor on train only
4. Train the configured model
5. Optionally calibrate on val split
6. Select threshold on val predictions
7. Evaluate on test split
8. Save all artifacts to `artifacts/models/{timestamp}-{git_hash}/`

### Evaluate without retraining
```bash
python -m creditrisk.cli evaluate --config configs/default.yaml
```

### Batch predictions on new data
```bash
python -m creditrisk.cli predict-batch \
  --config configs/default.yaml \
  --input data/test.csv \
  --output predictions.csv
```

---

## Interpreting Results

### metrics.json (inside artifact version directory)
```json
{
  "val_roc_auc": 0.65,
  "val_pr_auc": 0.28,
  "test_roc_auc": 0.64,
  "test_pr_auc": 0.27,
  "threshold": 0.12,
  "threshold_strategy": "min_expected_cost"
}
```

**What these mean:**
- `test_roc_auc` ≥ 0.65 → model is passing the production gate
- `test_roc_auc` ≈ 0.52 → near-random; likely a preprocessing or leakage issue
- `test_roc_auc` > 0.70 → suspect leakage; run `/debug-leakage` in Claude Code
- `val_roc_auc` >> `test_roc_auc` (>0.05 gap) → overfitting on val; review threshold selection

**Class imbalance context:**
The dataset is ~9.25% positive (default). PR-AUC is more informative than accuracy here.
A model predicting all 0s gets 90.75% accuracy — that is not a good model.

---

## Starting the API

```bash
uv run uvicorn creditrisk.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test endpoints:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"record": {"Loan Amount": 15000, "Interest Rate": 12.5, "Grade": "C", ...}}'
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Note: `CREDITRISK_DISABLE_SHAP=1` is set automatically in `test_api.py` to avoid SHAP
timeout during CI. This is a known workaround — do not remove it.

```bash
# With coverage report
uv run pytest tests/ --cov=src/creditrisk --cov-report=term-missing
```

---

## Using Claude Code Commands

From inside Claude Code with this project open:

| Command            | Purpose                                           |
|--------------------|---------------------------------------------------|
| `/review`          | Pre-commit leakage and AUC review checklist       |
| `/retrain`         | Step-by-step clean retrain procedure              |
| `/debug-leakage`   | Detect data leakage in preprocessing code         |

---

## Common Issues

**`ValueError: could not convert string to float`**
The model received a categorical value it wasn't trained on. If using `LabelEncoder`
(legacy), replace it with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.

**AUC stuck at ~0.52**
Likely causes:
1. Data leakage fix not yet applied — `LabelEncoder` or `pd.get_dummies` on combined data
2. Model using default threshold 0.5 on imbalanced data — use `min_expected_cost` strategy
3. Wrong features in pipeline — verify drop list in `configs/default.yaml`

**SHAP timeout in tests**
Expected — set `CREDITRISK_DISABLE_SHAP=1` for test runs. SHAP is slow on large backgrounds.