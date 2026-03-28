# Architecture — Credit Risk ML System

## Data Flow (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│  raw CSV (Give Me Some Credit, 150,000 rows × 10 numeric cols)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/load.py                                                   │
│  - pd.read_csv                                                  │
│  - basic shape / dtype check                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/schema.py                                                 │
│  - Schema inference                                             │
│  - Required column validation (raises SchemaError if missing)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/split.py                                                  │
│  - Stratified split on SeriousDlqin2yrs (70 / 15 / 15)        │
│  - Seeded for reproducibility                                   │
│  - Fractions validated to sum to 1.0 ± 1e-6                    │
│  → X_train, X_val, X_test, y_train, y_val, y_test              │
└───────┬─────────────────────────────────────┬───────────────────┘
        │                                     │
        ▼                                     ▼
┌───────────────────┐               ┌─────────────────────────┐
│  X_train, y_train │               │  X_val, X_test          │
│  (fit goes here)  │               │  (transform only)       │
└────────┬──────────┘               └───────────┬─────────────┘
         │                                      │
         ▼                                      │
┌─────────────────────────────────────────────────────────────────┐
│  data/preprocess.py — Pipeline (fe → ct)                        │
│  Fit on X_train only:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ fe step (pure, no fit):                                 │   │
│  │   Winsorization: util→[0,1], DebtRatio→[0,10],         │   │
│  │     age→[18,100], MonthlyIncome→[0, 99th pct]          │   │
│  │   Sentinel cap: delinquency cols → [0, 10]             │   │
│  │   Missingness flags: income_missing, dependents_missing │   │
│  │   Engineered: total_delinquencies, delinquency_severity,│   │
│  │     utilization_x_delinquency, income_to_debt,          │   │
│  │     credit_line_utilization, age_bin                    │   │
│  │ ct step (fitted):                                       │   │
│  │   All cols → SimpleImputer(median) → StandardScaler    │   │
│  └─────────────────────────────────────────────────────────┘   │
│  .transform(X_val), .transform(X_test) ◄────────────────────── │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  model/train.py — _build_estimator()                           │
│  Supported: logistic_regression | random_forest | lightgbm     │
│  - LightGBM: scale_pos_weight set from class distribution      │
│  - Optional: ImbPipeline with SMOTE                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  model/calibrate.py                                             │
│  - CalibratedClassifierCV on val split (optional)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  model/evaluate.py — select_threshold()                        │
│  Strategies: f1 | max_recall_at_precision | min_expected_cost  │
│  - Threshold selected on val predictions only                  │
│  - Final metrics computed on test split (once)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  model/registry.py                                              │
│  Artifacts saved: model.joblib, preprocess.joblib,             │
│  metrics.json, schema.json, config.yaml, background.joblib,    │
│  ROC/PR curve PNGs                                             │
│  Versioned as: artifacts/models/{timestamp}-{git_hash}/        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  api/main.py — FastAPI                                          │
│  Endpoints:                                                     │
│  GET  /health       → {"status": "ok"}                         │
│  GET  /model-info   → version, threshold, metrics, importances │
│  POST /predict      → prob_default, decision, threshold        │
│  POST /explain      → /predict + SHAP explanation array        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Comparison Table

Results marked **[LEAKY]** used train+test combined before encoding — values are unreliable.

### Bank Loan Hackathon dataset (67,463 rows, ~9.25% positive rate)

| Model                        | Resampling   | ROC-AUC | Notes                                    | Trust      |
|------------------------------|--------------|---------|------------------------------------------|------------|
| Random Forest (baseline)     | None         | 0.516   | Near-random; weak signal extraction     | ✅ Honest  |
| Random Forest                | SMOTE        | 0.508   | SMOTE did not help; possible overfit    | ✅ Honest  |
| Random Forest                | NearMiss     | 0.503   | High recall, very low precision         | ✅ Honest  |
| Random Forest (class_weight) | None         | 0.522 (test) / 0.532 (val) | Production artifact (20251220184226-nogit) | ✅ Honest  |
| Decision Tree                | None         | 0.507   | Overfitting likely                      | ✅ Honest  |
| XGBoost (scale_pos_weight)   | None         | 0.507   | Not tuned; needs depth/lr tuning        | ✅ Honest  |
| XGBoost (grid-tuned)         | None         | 0.533   | Best honest XGBoost result              | ✅ Honest  |
| SVC + SMOTE                  | SMOTE        | 0.682   | Inflated — joint encoding leakage       | ❌ Leaky   |

**Note:** Bank Loan Hackathon features have very low mutual information with target in
a leakage-free split. Honest AUC ceiling ~0.53. Project migrated to GMS Credit.

### Give Me Some Credit dataset (150,000 rows, 6.68% positive rate)

| Model                                      | Resampling   | ROC-AUC | Notes                                                                                   | Trust      |
|--------------------------------------------|--------------|---------|-----------------------------------------------------------------------------------------|------------|
| LightGBM (GMS Credit, leakage-free)        | class_weight | **0.8498** | Sentinel capping, Winsorization, missingness indicators, 8 engineered features      | ✅ Honest  |

**Production target:** ≥ 0.85 on held-out test split. **Achieved: 0.8498.**

---

## Known Architecture Gaps (from audit)

| Gap                          | Severity | Status   |
|------------------------------|----------|----------|
| No API authentication        | 🔴 High  | Pending  |
| No decision/audit logging    | 🔴 High  | Pending  |
| No TLS                       | 🔴 High  | Pending  |
| PSI drift detection unwired  | 🟠 Med   | Pending  |
| /health doesn't check bundle | 🟠 Med   | Pending  |
| No request logging middleware| 🟠 Med   | Pending  |
| Single uvicorn worker        | 🟡 Low   | Pending  |
| Loose dependency pinning     | 🟠 Med   | Pending  |