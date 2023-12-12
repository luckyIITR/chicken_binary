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
import os
import asyncio
import subprocess

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
    user = await get_current_user(request)
    return templates.TemplateResponse("home.html", {"request": request, 'user': user})


@router.post("/", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    user = await get_current_user(request)

    contents = await image.read()  # Read the contents of the uploaded file

    decodeImage(contents, clApp.filename)
    result = clApp.classifier.predict()
    encoded_image = base64.b64encode(contents).decode('utf-8')
    # print(result)
    return templates.TemplateResponse("home.html",
                                      {"request": request, 'user': user,'result': result['result'], 'image_data': encoded_image})


# Keep track of the task status, you might use a database or in-memory storage
task_status = "pending"
is_process_running = False


async def execute_long_running_task():
    global task_status, is_process_running

    # Check if a process is already running
    if is_process_running:
        return

    is_process_running = True
    task_status = "running"

    process = await asyncio.create_subprocess_exec(
        "python", "main.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()

    task_status = "completed"
    is_process_running = False


@router.get('/train', response_class=HTMLResponse)
async def train_model(request: Request):
    user = await get_current_user(request)
    if user is None:
        return RedirectResponse(url="/auth", status_code=status.HTTP_302_FOUND)
    msg = ''
    if task_status == 'running':
        msg = 'Training Process already Running'
    asyncio.create_task(execute_long_running_task())
    if msg == '':
        msg = "Training Started"
    return templates.TemplateResponse('status.html', {"request": request, "msg": msg, 'user': user})


@router.get('/admin', response_class=HTMLResponse)
async def train_model(request: Request):
    user = await get_current_user(request)
    if user is None:
        return RedirectResponse(url="/auth", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse('admin.html', {"request": request, 'user': user})
