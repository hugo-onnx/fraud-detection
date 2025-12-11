from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from contextlib import asynccontextmanager

from app.api.routes import router
from app.services.ml_service import model_service
from app.gradio_app import gr_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

app.include_router(router)

app = mount_gradio_app(app, gr_app, path="/gradio")