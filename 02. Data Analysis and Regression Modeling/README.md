# Data Analysis and Regression Modeling

## Overview

This project involves performing data analysis and regression modeling using Python. The primary tasks include:

1. **Simple Linear Regression:** Implementing and visualizing a simple linear regression model.
2. **Multiple Linear Regression with PyTorch:** Using PyTorch for implementing a multiple linear regression model.

## Project Structure

The project consists of the following key components:

1. **Data Loading and Preparation**
2. **Simple Linear Regression**
3. **Multiple Linear Regression with PyTorch**

## Dependencies

To run this project, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `sklearn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib torch scikit-learn

```

## Usage

### 1. Data Loading and Preparation
This section loads the dataset from a CSV file and prepares it for analysis.

- **Data File Path:** Ensure the data file `data_mod.csv` is located in the `/content/sample_data/` directory or update the path accordingly.

### 2. Simple Linear Regression
In this part, the following steps are performed:

- **Extract Features and Target:** Extract `sqft_living` as the feature and `price` as the target variable.
- **Normalize Data:** Normalize both the feature and target variable.
- **Plot Data and Initial Regression Line:** Visualize the normalized data and initial regression line.
- **Calculate Cost Function:** Compute the initial cost function for the regression model.
- **Implement Gradient Descent:** Perform gradient descent to optimize the regression coefficients.
- **Plot Cost Function Over Iterations:** Visualize how the cost function decreases over iterations.

### 3. Multiple Linear Regression with PyTorch
In this section, a multiple linear regression model is implemented using PyTorch:

- **Load and Normalize Data:** Load the dataset and normalize features and target variables.
- **Convert to PyTorch Tensors:** Convert the data into PyTorch tensors for training.
- **Initialize Weights and Bias:** Initialize the weights and bias for the regression model.
- **Define Hypothesis and Cost Functions:** Define functions for calculating predictions and cost.
- **Train Model with Gradient Descent:** Train the model using gradient descent and print the loss at regular intervals.
- **Print Final Results:** Output the final regression coefficients and cost function value.

## Code
The code is structured as follows:

1. **Import Libraries:** Import necessary libraries for data manipulation, visualization, and machine learning.
2. **Loading and Checking Data:** Load the dataset and perform initial checks.
3. **Simple Linear Regression:** Perform and visualize simple linear regression.
4. **Multiple Linear Regression with PyTorch:** Implement and train a multiple linear regression model using PyTorch.

## Example
To run the code, ensure you have the dataset in the correct location and run the notebook or Python script. The notebook provides detailed comments and visualizations to help understand each step of the process.


## Additional Information

- **Author:** Anas
- **Contact:** iamanasinam@example.com

Feel free to reach out if you have any questions or issues with the implementation.