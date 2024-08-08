from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    return templates.TemplateResponse(
        "welcome.html", {"request": request, "title": "Welcome"}
    )


@app.get("/train", response_class=HTMLResponse)
async def train_pipeline(request: Request):
    # Dummy training logic
    training_result = (
        "Training completed successfully!"  # Replace this with actual training logic
    )
    return templates.TemplateResponse(
        "train.html",
        {"request": request, "title": "Training", "training_result": training_result},
    )


@app.get("/test", response_class=HTMLResponse)
async def test_pipeline(request: Request):
    # Dummy prediction logic
    prediction_result = (
        "Prediction made: Class A"  # Replace this with actual prediction logic
    )
    return templates.TemplateResponse(
        "test.html",
        {
            "request": request,
            "title": "Testing",
            "prediction_result": prediction_result,
        },
    )
