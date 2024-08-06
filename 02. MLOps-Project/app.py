from heart_disease_prediction.components.model_trainer import load_transformed_data


def main():
    file_path = "./artifact/transformed_test/test_transformed.csv"
    transformed_x_train = load_transformed_data(file_path)
    print(transformed_x_train)


if __name__ == "__main__":
    main()
