# Credit Risk ML System

Production-grade credit risk classification system achieving ROC-AUC 0.8498 on Give Me Some Credit (150,000 rows) using LightGBM with leakage-safe preprocessing and FastAPI serving.

---

## Key Results

- **test_roc_auc: 0.8498** | val_roc_auc: 0.8574
- **Dataset:** Give Me Some Credit — 150,000 rows, 6.68% positive rate, 10 raw numeric features
- **Pipeline:** 8 engineered features added before fitting; all transformers fit on training split only
- **Serving:** 4 endpoints — `/health`, `/model-info`, `/predict`, `/explain`

---

## Architecture

Raw CSV → `load.py` (drop index col) → `schema.py` (dtype inference) → `split.py` (stratified 70/15/15) → `preprocess.py` (FE + impute + scale, fit on train only) → `model/train.py` (LightGBM, `scale_pos_weight` from `y_train`) → `evaluate.py` (threshold on val, final metrics on test) → `registry.py` (versioned artifacts) → `api/main.py` (FastAPI).

See [`docs/architecture.md`](docs/architecture.md) for the full annotated diagram and model comparison table.

---

## Tech Stack

| Component        | Technology                                              |
|------------------|---------------------------------------------------------|
| Model            | LightGBM (gradient boosting, `scale_pos_weight` for class imbalance) |
| Serving          | FastAPI + uvicorn                                       |
| Preprocessing    | scikit-learn Pipeline — SimpleImputer → StandardScaler  |
| Explainability   | SHAP (permutation importance fallback)                  |
| Containerization | Docker (Python 3.11-slim)                               |
| Testing          | pytest                                                  |
| CI               | GitHub Actions — ruff + black + pytest on push/PR       |

---

## Quickstart

```bash
# Install
uv sync

# Dataset
# Download cs-training.csv from https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place at data/cs-training.csv

# Train
PYTHONPATH=src .venv/bin/python -m creditrisk.cli --config configs/default.yaml train

# Evaluate latest artifact
PYTHONPATH=src .venv/bin/python -m creditrisk.cli --config configs/default.yaml evaluate

# Serve
.venv/bin/python -m uvicorn creditrisk.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

---

## Project Structure

```
credit-risk-ml-system/
├── CLAUDE.md                        # Project memory — conventions, status, dev commands
├── configs/
│   └── default.yaml                 # Central config: data path, model params, thresholds
├── src/creditrisk/
│   ├── config.py                    # Pydantic config loader (validated at startup)
│   ├── cli.py                       # CLI: data-summary | train | evaluate | predict-batch
│   ├── api/
│   │   ├── main.py                  # FastAPI app: /health /predict /explain /model-info
│   │   ├── model_loader.py          # Loads versioned model bundle into app.state
│   │   └── schemas.py               # Pydantic request/response models
│   ├── data/
│   │   ├── load.py                  # CSV ingestion; drops Unnamed: 0 index column
│   │   ├── schema.py                # Schema inference (dtype-based) and validation
│   │   ├── preprocess.py            # Pipeline: build_feature_engineering → ColumnTransformer
│   │   └── split.py                 # Stratified/time-based train/val/test split
│   ├── model/
│   │   ├── train.py                 # End-to-end training orchestration
│   │   ├── evaluate.py              # Metrics, threshold selection (f1/recall@precision/cost)
│   │   ├── calibrate.py             # CalibratedClassifierCV wrapper
│   │   └── registry.py              # Versioned artifact persistence: {timestamp}-{git_hash}/
│   ├── explain/
│   │   └── shap_explain.py          # SHAP → permutation_importance → coefficients fallback
│   └── monitoring/
│       ├── drift.py                 # PSI drift detection (implemented, not yet wired)
│       └── logging.py               # Structured logging setup (implemented, not yet wired)
├── tests/
│   ├── test_api.py
│   ├── test_batch_predict.py
│   ├── test_data_contract.py
│   ├── test_split.py
│   └── test_train_evaluate.py
├── docs/
│   ├── architecture.md              # Full data flow diagram, model comparison table
│   └── onboarding.md               # Setup guide, pipeline walkthrough
├── artifacts/                       # Gitignored — versioned model artifacts live here
├── Dockerfile
├── pyproject.toml
├── ruff.toml
└── requirements.txt
```

---

## Feature Engineering

All transformations in `build_feature_engineering()` are pure (no fitted state) and run before the `ColumnTransformer`.

**Data quality fixes applied in-place:**
- `RevolvingUtilizationOfUnsecuredLines` — clipped to `[0, 1]` (raw values reach 50,000+)
- `DebtRatio` — clipped to `[0, 10]`
- `age` — clipped to `[18, 100]` (one record has age = 0)
- `MonthlyIncome` — clipped to `[0, 99th percentile]` to neutralize extreme outliers
- `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate` — clipped to `[0, 10]`; value 98 is a sentinel/data quality flag, not a real count

**Missingness indicators (captured before imputation):**
- `income_missing` — 1 if `MonthlyIncome` was null (19.82% of rows)
- `dependents_missing` — 1 if `NumberOfDependents` was null (2.62% of rows)

**Engineered features:**
- `total_delinquencies` — sum of all three delinquency count columns (after sentinel capping)
- `delinquency_severity` — weighted sum: 30-59 day × 1, 60-89 day × 2, 90+ day × 3
- `utilization_x_delinquency` — revolving utilization × (total delinquencies + 1)
- `income_to_debt` — `MonthlyIncome / (DebtRatio × MonthlyIncome + 1)`; safe against zero denominators
- `credit_line_utilization` — open credit lines / (real estate loans + 1)
- `age_bin` — ordinal bin: 18–30 → 0, 30–45 → 1, 45–60 → 2, 60–100 → 3

---

## Key Engineering Decisions

**Leakage and why it matters.** Earlier exploration used a Kaggle notebook that applied label encoding and frequency encoding to the combined train+test DataFrame before splitting. This leaks test distribution into training, inflating SVC AUC to 0.68 on a dataset whose honest ceiling is ~0.53. The production pipeline splits first, then fits all transformers strictly on the training partition. `remainder="drop"` in the `ColumnTransformer` ensures no unanticipated columns reach the model.

**Why Give Me Some Credit.** The Bank Loan Hackathon dataset has very low mutual information between its features and the default target in a leakage-free setting (honest AUC ~0.53). Give Me Some Credit is a well-labelled benchmark from a Kaggle competition with an established public leaderboard, 150,000 rows, and features that genuinely predict serious delinquency. It allows comparison against known baselines.

**LightGBM with runtime `scale_pos_weight`.** The GMS Credit dataset has a 6.68% positive rate. `scale_pos_weight` is computed from `y_train` at runtime as `(negatives) / (positives)` rather than read from config, so it automatically reflects the actual class distribution in whatever training split is used. LightGBM with histogram-based splits handles the mix of integer counts and continuous ratios without requiring separate encoding pipelines.

**Train-only fit discipline.** `build_preprocessor()` returns a `Pipeline` whose first step (`fe`) is a stateless `FunctionTransformer` and whose second step (`ct`) is a `ColumnTransformer` fitted exclusively on `X_train`. `SimpleImputer` computes medians from training data only; `StandardScaler` computes mean and variance from training data only. Validation and test splits are transformed using those training statistics, never re-fitted.

---

## Development

Full development commands, conventions, and runbook are in [`CLAUDE.md`](CLAUDE.md).

**Slash commands** (Claude Code):

| Command           | Description                                              |
|-------------------|----------------------------------------------------------|
| `/review`         | Leakage and AUC review checklist                         |
| `/retrain`        | Step-by-step clean retrain procedure                     |
| `/debug-leakage`  | Checklist for detecting preprocessing leakage            |
