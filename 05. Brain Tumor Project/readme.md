# Brain Tumor Detection Using Deep Learning

This project involves the development of a machine learning model for the detection of brain tumors using MRI images. The project uses PyTorch for model development, including training custom and pre-trained models. A FastAPI application is also included to serve the model and provide predictions via a web interface.

## Project Structure

- `brain_tumor_dataset/`: Contains the MRI images in two folders:
  - `yes/`: Images labeled as having a tumor.
  - `no/`: Images labeled as not having a tumor.
- `app.py`: FastAPI application file to serve the model and predict brain tumor presence.
- `static/`: Directory containing static files like CSS for styling the web interface.
- `templates/`: Directory containing the HTML template for the web interface.
- `ANN_BEST_MODEL.pth`, `VGG_BEST_MODEL.pth`, `ResNet_BEST_MODEL.pth`: Saved model files for the best performing models.

## Dependencies

- Python 3.8 or later
- PyTorch
- Torchvision
- Pillow
- Scikit-learn
- FastAPI
- Uvicorn

You can install the dependencies using `pip`:

```bash
pip install torch torchvision pillow scikit-learn fastapi uvicorn
