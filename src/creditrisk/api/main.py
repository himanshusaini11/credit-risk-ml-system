"""FastAPI service for credit risk inference."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException

from creditrisk.api.model_loader import load_model_bundle
from creditrisk.api.schemas import ExplainResponse, ModelInfoResponse, PredictRequest, PredictResponse
from creditrisk.explain.shap_explain import explain_record

CONFIG_PATH = os.getenv("CREDITRISK_CONFIG", "configs/default.yaml")

app = FastAPI(title="Credit Risk Decision Service", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    bundle = load_model_bundle(CONFIG_PATH)
    app.state.model_bundle = bundle


def _get_bundle() -> Dict[str, Any]:
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded")
    return bundle


def _validate_record(record: Dict[str, Any], schema: Dict[str, Any]) -> pd.DataFrame:
    required = set(schema.get("feature_columns", []))
    missing = [col for col in required if col not in record]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    df = pd.DataFrame([record])
    return df


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    bundle = _get_bundle()
    metrics = bundle["metrics"]
    schema = bundle["schema"]
    return ModelInfoResponse(
        model_version=bundle["model_version"],
        threshold=float(metrics["threshold"]),
        metrics=metrics,
        global_explain=schema.get("global_explain"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    bundle = _get_bundle()
    schema = bundle["schema"]
    df = _validate_record(payload.record, schema)
    prob = float(bundle["model"].predict_proba(df)[0, 1])
    threshold = float(bundle["metrics"]["threshold"])
    decision = "REJECT" if prob >= threshold else "APPROVE"
    return PredictResponse(
        prob_default=prob,
        decision=decision,
        threshold=threshold,
        model_version=bundle["model_version"],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: PredictRequest) -> ExplainResponse:
    bundle = _get_bundle()
    schema = bundle["schema"]
    df = _validate_record(payload.record, schema)
    feature_names = schema.get("feature_names", schema.get("feature_columns", []))
    explanation = explain_record(
        bundle["model"],
        bundle["preprocessor"],
        df[schema["feature_columns"]],
        feature_names,
        background=bundle.get("background"),
        top_k=5,
    )
    prob = float(bundle["model"].predict_proba(df)[0, 1])
    threshold = float(bundle["metrics"]["threshold"])
    decision = "REJECT" if prob >= threshold else "APPROVE"
    return ExplainResponse(
        prob_default=prob,
        decision=decision,
        threshold=threshold,
        model_version=bundle["model_version"],
        method=explanation.get("method", "unknown"),
        explanation=[
            {"feature": item["feature"], "importance": item["importance"]}
            for item in explanation.get("contributions", [])
        ],
        note=explanation.get("note"),
    )
