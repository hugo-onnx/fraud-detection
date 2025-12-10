from pydantic import BaseModel, Field, model_validator

class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name to value mapping")

    @model_validator(mode="after")
    def validate_features(self):
        for k, v in self.features.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Feature '{k}' must be numeric, got {type(v).__name__}")
        return self

class DriftReportRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)

class PredictionResponse(BaseModel):
    fraud_probability: float
    model_version: str