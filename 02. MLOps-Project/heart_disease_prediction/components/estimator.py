# estimator.py

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def load_and_evaluate(model_path, preprocessing_path, test_data_path):
    # Load the model and preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(preprocessing_path)

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Extract true labels before transformation
    y_test = test_data["stroke"].values  # Adjust based on your dataset

    # Apply preprocessing to the test data
    transformed_data = preprocessing_pipeline.transform(test_data)

    # Make predictions
    predictions = model.predict(transformed_data)

    # Ensure y_test is reshaped to match predictions if necessary
    y_test = y_test.reshape(len(y_test), 1)
    predictions = predictions.reshape(len(predictions), 1)

    # Concatenate horizontally
    result = np.concatenate((predictions, y_test), axis=1)
    print("Predictions and True Labels:")
    print(result)

    # Compute and print the confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
