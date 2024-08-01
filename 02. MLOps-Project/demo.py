from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from dotenv import load_dotenv
import os
import subprocess

# Load environment variables from the .env file

load_dotenv()

data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()
# print(diArtifacts)


# Extract the test.csv path and schema.yaml path
test_csv_path = diArtifacts.test_file_path
schema_path = "./config/schema.yaml"

# print("Going to print the new test.csv data")

# Validate the test data
subprocess.run(
    [
        "python",
        "heart_disease_prediction/components/data_validation.py",
        test_csv_path,
        schema_path,
    ]
)
