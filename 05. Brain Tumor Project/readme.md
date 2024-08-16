# Brain Tumor Detection Using Deep Learning

This project involves the development of a machine-learning model for the detection of brain tumours using MRI images. The project uses PyTorch for model development, including training custom and pre-trained models. A FastAPI application is also included to serve the model and provide predictions via a web interface.

## Project Structure

- `brain_tumor_dataset/`: Contains the MRI images in two folders:
  - `yes/`: Images labelled as having a tumour.
  - `no/`: Images labelled as not having a tumour.
- `app.py`: FastAPI application file to serve the model and predict brain tumour presence.
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
```

## Dataset
The dataset is split into two classes:
- **yes**: Images with brain tumors.
- **no**: Images without brain tumors.

Images are loaded from the respective directories and preprocessed before being fed into the models.

## Models
The following models are trained and evaluated:
- **Custom ANN (Artificial Neural Network)**: A simple feedforward neural network.
- **VGG-Net**: A pre-trained VGG-16 model with fine-tuning on the last layer.
- **ResNet**: A pre-trained ResNet-50 model with fine-tuning on the last layer.

The best-performing model is saved and used in the FastAPI application for predictions.

## Training
The models are trained using the following settings:
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50
- **Batch Size**: 35

The dataset is split into 80% for training and 20% for testing.

## Evaluation
The models are evaluated on the test set using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **Confusion Matrix**

The best model is selected based on accuracy and is saved for deployment.

## FastAPI Application
The FastAPI application allows users to upload MRI images and receive a prediction on whether a tumor is present or not.

### Running the Application
Start the FastAPI server:
```bash
uvicorn app:app --reload
```
## Prediction
Upload an MRI image through the web interface, and the model will predict whether the image indicates the presence of a tumor.

## Deployment
The application can be deployed on platforms like Azure for broader access. The deployment guide is in progress.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Dataset**: MRI images used for training and testing.
- **PyTorch**: For model development and training.
- **FastAPI**: This is used to create the web interface for predictions.

## Contact
For any inquiries or contributions, don't hesitate to contact Anas Inam at iamanasinam@gmail.com.
