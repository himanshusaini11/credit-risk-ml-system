"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    record: Dict[str, Any]


class FeatureContribution(BaseModel):
    feature: str
    importance: float


class PredictResponse(BaseModel):
    prob_default: float
    decision: Literal["APPROVE", "REJECT"]
    threshold: float
    model_version: str


class ExplainResponse(PredictResponse):
    method: str
    explanation: List[FeatureContribution]
    note: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    metrics: Dict[str, Any]
    global_explain: Optional[Dict[str, Any]] = None
