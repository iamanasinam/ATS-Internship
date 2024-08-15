# K-Nearest Neighbors (KNN) Image Classification

This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm to classify images using a provided dataset. The project is implemented in Python using Jupyter Notebook and demonstrates how to apply 1-NN, 3-NN, and 5-NN classification to find the closest matches for a given test image.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Additional Information](#additional-information)

## Project Overview

The goal of this project is to classify a test image by comparing it with a dataset of images using the K-Nearest Neighbors algorithm. The KNN algorithm works by finding the 'k' closest images to the test image and assigning a label based on majority voting. This project implements 1-NN, 3-NN, and 5-NN to showcase the differences in performance and accuracy.

## Dataset

The dataset used in this project is stored in a `.mat` file (`data.mat`). The images are stored in a 4D array with dimensions `[32, 32, 3, -1]`, where each image is of size 32x32 pixels with 3 color channels (RGB).

Before running the notebook, ensure that you have the `data.mat` file in the appropriate directory.

## Requirements

To run this project, you'll need the following Python libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `opencv-python`

You can install the required packages using pip:

```bash
pip install numpy scipy matplotlib Pillow opencv-python
```


## Usage

 - Clone the repository:


    ```bash
        git clone https://github.com/iamanasinam/ATS-Internship
    ```
- Open `03. Celebrity Look-Alike Classifier`:
    ```bash
    cd 03. Celebrity Look-Alike Classifier
    ```

-  Open the Jupyter Notebook `KNN_Image_Classification.ipynb`.

-  Run the notebook cells sequentially to perform 1-NN, 3-NN, and 5-NN classification on your test image.

## Results

After running the notebook, the closest images from the dataset will be displayed for each of the KNN variations (1-NN, 3-NN, and 5-NN). The notebook also prints the most common label among the nearest neighbors.

**Example output:**

- **1-NN:** Displays the closest image from the dataset.
- **3-NN:** Displays the three closest images and the most common label among them.
- **5-NN:** Displays the five closest images and the most common label among them.


## Additional Information

- **Author:** Anas
- **Repository:** KNN Image Classification
- **Contact:** iamanasinam@example.com

Feel free to reach out if you have any questions or issues with the implementation.
