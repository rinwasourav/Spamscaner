import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_MODEL_PATH = "models/signaltrust_xgboost.pkl"
FEATURE_COLUMNS = ["calls_per_hour", "avg_call_duration", "contact_degree", "spam_report_count"]

# Global model state
loaded_model: Optional[Any] = None

def load_model(path: str = DEFAULT_MODEL_PATH):
    global loaded_model
    if os.path.exists(path):
        try:
            loaded_model = joblib.load(path)
            print(f"[+] Loaded ML model from '{path}'")
        except Exception as e:
            print(f"[!] Error loading model artifact: {e}")
            loaded_model = None
    else:
        print(f"[!] Model artifact '{path}' not found. API will use heuristic fallback until trained.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="SignalTrust Threat Scoring API",
    description="Real-time machine learning inference API for anti-spam and call fraud detection.",
    version="1.0.0",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    calls_per_hour: float = Field(..., ge=0, description="Frequency of outgoing calls in the past hour")
    avg_call_duration: float = Field(..., ge=0, description="Average duration of recent calls in seconds")
    contact_degree: int = Field(..., ge=0, description="Number of mutual/shared contacts")
    spam_report_count: int = Field(..., ge=0, description="Total spam flags received in last 24h")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "calls_per_hour": 45.0,
                "avg_call_duration": 4.2,
                "contact_degree": 0,
                "spam_report_count": 12
            }
        }
    )

class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Probability of call being spam/fraud (0.0 to 1.0)")
    is_threat: bool = Field(..., description="True if risk score exceeds threat threshold")
    threat_level: str = Field(..., description="Classification: LOW, MEDIUM, or HIGH risk")
    threshold_used: float = Field(0.70, description="Decision threshold applied")
    model_version: str = Field("xgboost-v1", description="Model engine version")

def heuristic_predict(req: PredictRequest) -> float:
    """Fallback heuristic risk scoring if model artifact is unavailable."""
    score = 0.1
    if req.calls_per_hour > 20:
        score += 0.35
    if req.avg_call_duration < 15:
        score += 0.25
    if req.contact_degree == 0:
        score += 0.15
    if req.spam_report_count > 3:
        score += 0.25
    return min(float(score), 1.0)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": loaded_model is not None,
        "model_path": DEFAULT_MODEL_PATH
    }

@app.post("/predict", response_model=PredictResponse)
def predict_threat(data: PredictRequest) -> PredictResponse:
    threshold = 0.70
    
    if loaded_model is not None:
        try:
            df = pd.DataFrame([{
                "calls_per_hour": data.calls_per_hour,
                "avg_call_duration": data.avg_call_duration,
                "contact_degree": data.contact_degree,
                "spam_report_count": data.spam_report_count
            }])[FEATURE_COLUMNS]
            
            risk_score = float(loaded_model.predict_proba(df)[0][1])
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model inference failed: {str(e)}"
            )
    else:
        risk_score = heuristic_predict(data)

    is_threat = risk_score >= threshold

    if risk_score >= 0.85:
        threat_level = "HIGH"
    elif risk_score >= 0.50:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    return PredictResponse(
        risk_score=round(risk_score, 4),
        is_threat=is_threat,
        threat_level=threat_level,
        threshold_used=threshold,
        model_version="xgboost-v1" if loaded_model is not None else "heuristic-fallback"
    )
