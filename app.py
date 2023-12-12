from fastapi import FastAPI, Request
from starlette import status
import models
from database import engine
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from routes import auth, chicken_binary
from cnnClassifier import logger

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.client.host} - \"{request.method} {request.url.path} {request.scope['http_version']}\"")
    response = await call_next(request)
    return response

app.include_router(auth.router)
app.include_router(chicken_binary.router)


@app.get("/", response_class=HTMLResponse)
async def authentication_page(request: Request):
    return RedirectResponse(url="/chicken-binary", status_code=status.HTTP_302_FOUND)