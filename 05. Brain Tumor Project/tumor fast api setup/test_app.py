from fastapi.testclient import TestClient
from main import app  # Replace 'main' with the name of your FastAPI app file

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Stroke Detection API"}


def test_predict_tumor():
    files = {"file": open("path_to_sample_image.jpg", "rb")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 200
    assert (
        "prediction" in response.json()
    )  # Adjust this based on your response structure
