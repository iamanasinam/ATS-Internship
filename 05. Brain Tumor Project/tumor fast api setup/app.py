from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# from fastapi import HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`",
)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load the entire model (architecture + weights)
model = torch.load("./ResNet_best_model.pth", map_location=torch.device("cpu"))
model.eval()

# Transform to apply to input images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = "Tumor Detected" if predicted.item() == 1 else "No Tumor Detected"

    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction": prediction}
    )


# @app.post("/predict/")
# async def predict(request: Request, file: UploadFile = File(...)):
#     # Check if the uploaded file is an image
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400, detail="Invalid file type. Please upload an image file."
#         )

#     # Read the image file
#     try:
#         image = Image.open(io.BytesIO(await file.read()))
#     except Exception:
#         raise HTTPException(
#             status_code=400,
#             detail="Error reading the image. Please upload a valid image file.",
#         )

#     # Preprocess the image
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#         prediction = "Tumor Detected" if predicted.item() == 1 else "No Tumor Detected"

#     return templates.TemplateResponse(
#         "index.html", {"request": request, "prediction": prediction}
#     )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
