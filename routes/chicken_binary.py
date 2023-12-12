import base64
from typing import Annotated
from fastapi import APIRouter, Depends, Form, Request, File, UploadFile
from starlette import status
from starlette.responses import RedirectResponse
import models
from cnnClassifier.pipeline.predict import PredictionPipeline
from database import SessionLocal
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from .auth import get_current_user
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io


from cnnClassifier.utils.common import decodeImage

router = APIRouter(
    prefix="/chicken-binary",
    tags=["chicken-binary"],
    responses={404: {"description": "Not found"}}
)
templates = Jinja2Templates(directory="templates")


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.post("/", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    contents = await image.read()  # Read the contents of the uploaded file

    decodeImage(contents, clApp.filename)
    result = clApp.classifier.predict()
    encoded_image = base64.b64encode(contents).decode('utf-8')
    # print(result)
    return templates.TemplateResponse("home.html", {"request": request, 'result': result['result'], 'image_data': encoded_image})
