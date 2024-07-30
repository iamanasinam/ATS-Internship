from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from dotenv import load_dotenv

# Load environment variables from the .env file

load_dotenv()

data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()
print(diArtifacts)
