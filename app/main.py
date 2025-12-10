from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import router
from app.services.ml_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

app.include_router(router)