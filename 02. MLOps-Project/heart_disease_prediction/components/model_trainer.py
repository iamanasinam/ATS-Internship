import pandas as pd
import yaml
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle


def read_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "../../config/schema.yaml")
    try:
        with open(schema_path, "r") as file:
            schema = yaml.safe_load(file)
        return schema
    except FileNotFoundError:
        print(f"Schema file not found at {schema_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading schema file: {e}")
        return None


def load_transformed_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Data file not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None


def process_data():
    train_file_path = "./artifact/transformed_train/train_transformed.csv"
    test_file_path = "./artifact/transformed_test/test_transformed.csv"

    transformed_train = load_transformed_data(train_file_path)
    transformed_test = load_transformed_data(test_file_path)

    return transformed_train, transformed_test


def read_transformed_data():
    transformed_train, transformed_test = process_data()
    schema = read_schema()
    if transformed_train is not None:
        print("Training data:")
        print(transformed_train.shape)
    if transformed_test is not None:
        print("Test data:")
        print(transformed_test.shape)
    if schema is not None:
        print("Schema file contents:")
        print(schema)


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, output_dict=True)

    evaluation_report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }
    return evaluation_report, model


def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def save_evaluation_report_as_yaml(report, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        yaml.dump(report, file, default_flow_style=False)
    print(f"Evaluation report saved at {file_path}")


def model_evaluation(expected_score):
    transformed_train, transformed_test = process_data()

    if transformed_train is not None and transformed_test is not None:
        X_train = transformed_train.drop("stroke", axis=1)
        y_train = transformed_train["stroke"]
        X_test = transformed_test.drop("stroke", axis=1)
        y_test = transformed_test["stroke"]

        # Train and evaluate KNN
        knn_model = KNeighborsClassifier()
        knn_report, knn_trained_model = train_and_evaluate_model(
            knn_model, X_train, y_train, X_test, y_test
        )

        # Train and evaluate SVM
        svm_model = SVC()
        svm_report, svm_trained_model = train_and_evaluate_model(
            svm_model, X_train, y_train, X_test, y_test
        )

        # Compare models
        if knn_report["accuracy"] > svm_report["accuracy"]:
            best_model_report = knn_report
            best_model = knn_trained_model
            best_model_name = "KNN"
        else:
            best_model_report = svm_report
            best_model = svm_trained_model
            best_model_name = "SVM"

        print(f"{best_model_name} Model Evaluation Report:")
        for key, value in best_model_report.items():
            print(f"{key}: {value}")

        # Check if the best model meets the expected score
        if best_model_report["accuracy"] >= expected_score:
            print(
                f"Best model ({best_model_name}) meets the expected score. Saving model..."
            )
            model_path = "./artifact/best_model.pkl"
            save_model(best_model, model_path)

            # Save evaluation report as YAML
            yaml_report_path = "./artifact/model_evaluation_report.yaml"
            save_evaluation_report_as_yaml(best_model_report, yaml_report_path)
            print(f"Evaluation report saved at {yaml_report_path}")
        else:
            print(
                f"Best model ({best_model_name}) does not meet the expected score. Model discarded."
            )
