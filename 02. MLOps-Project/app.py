from heart_disease_prediction.components.model_trainer import (
    process_data,
    read_transformed_data,
    model_evaluation,
)


def main():

    read_transformed_data()
    expected_score = 0.85  # Define the expected accuracy score
    model_evaluation(expected_score)

    # transformed the data
    # read_transformed_data()

    # read_transformed_data()
    # model_evaluation()
    # Process the data
    # transformed_train, transformed_test = process_data()
    # # process_data()
    # test = print_data()
    # print(test)

    # Print the data (or handle it as needed)
    # print("Transformed Train Data:")
    # print(transformed_train.shape)

    # print("\nTransformed Test Data:")
    # print(transformed_test.shape)


if __name__ == "__main__":
    main()
