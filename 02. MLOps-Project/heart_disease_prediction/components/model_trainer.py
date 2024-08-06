# heart_disease_prediction/component/model_trainer.py
import pandas as pd


def load_transformed_data(file_path):
    transformed_x_train = pd.read_csv(file_path)
    return transformed_x_train
