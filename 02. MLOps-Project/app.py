# app.py

from dotenv import load_dotenv
import subprocess
import pandas as pd
from heart_disease_prediction.components import estimator
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
from heart_disease_prediction.components.model_trainer import (
    read_transformed_data,
    model_evaluation,
)


# Load environment variables from the .env file
load_dotenv()

# Data Ingestion
data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()

# Extract the test.csv path and schema.yaml path
test_csv_path = diArtifacts.test_file_path
schema_path = "./config/schema.yaml"

# Validate the test data
subprocess.run(
    [
        "python",
        "heart_disease_prediction/components/data_validation.py",
        test_csv_path,
        schema_path,
    ]
)

# Data Transformation
data_transformation = DataTransformation(diArtifacts, schema_path)
transformation_artifacts = data_transformation.initiate_data_transformation()

print("Data transformation completed. Artifacts:", transformation_artifacts)

# Read transformed data and evaluate the model
read_transformed_data()
expected_score = 0.85  # Define the expected accuracy score
model_evaluation(expected_score)

print("Model evaluation completed.")

# Paths to your files
model_path = "./artifact/best_model.pkl"
preprocessing_path = "./artifact/preprocessing.pkl"
test_data_path = diArtifacts.test_file_path


# Function to get user input and predict
def get_user_input_and_predict():
    print("Please enter the following details:")
    user_data = {
        "age": input("Enter age (e.g., 45): "),
        "hypertension": input("Enter hypertension (0 for No, 1 for Yes): "),
        "heart_disease": input("Enter heart disease (0 for No, 1 for Yes): "),
        "avg_glucose_level": input("Enter average glucose level (e.g., 85.96): "),
        "bmi": input("Enter BMI (e.g., 22.3): "),
        "gender": input("Enter gender (Male/Female): "),
        "ever_married": input("Ever married (Yes/No): "),
        "work_type": input(
            "Work type (Private/Self-employed/Govt_job/Children/Never_worked): "
        ),
        "Residence_type": input("Residence type (Urban/Rural): "),
        "smoking_status": input(
            "Smoking status (formerly smoked/never smoked/smokes/Unknown): "
        ),
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Convert appropriate columns to numeric
    numeric_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
    ]
    for column in numeric_columns:
        user_df[column] = pd.to_numeric(user_df[column])

    # Predict using the estimator
    prediction = estimator.predict(model_path, preprocessing_path, user_df)
    print(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")


# Execute prediction
get_user_input_and_predict()


# Enter age (e.g., 45): 50
# Enter hypertension (0 for No, 1 for Yes): 1
# Enter heart disease (0 for No, 1 for Yes): 0
# Enter average glucose level (e.g., 85.96): 105.5
# Enter BMI (e.g., 22.3): 28.7
# Enter gender (Male/Female): Male
# Ever married (Yes/No): Yes
# Work type (Private/Self-employed/Govt_job/Children/Never_worked): Private
# Residence type (Urban/Rural): Urban
# Smoking status (formerly smoked/never smoked/smokes/Unknown): never smoked
