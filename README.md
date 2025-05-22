Faces vs. Non-Faces Classification
Project Overview
This project implements a face vs. non-face image classification system using a K-Nearest Neighbors (KNN) classifier with dimensionality reduction techniques, specifically Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). The goal is to evaluate the performance of KNN with PCA and LDA on a dataset of face and non-face images, analyzing how accuracy varies with different proportions of non-face images and train-test split ratios.
The project is implemented in a Jupyter Notebook (Faces_VS_Non-Faces.ipynb) using Python and common machine learning libraries.
Features

Image Preprocessing: Loads and processes face and non-face images, converting non-face images to grayscale and flattening all images into feature vectors.
Dimensionality Reduction: Applies PCA and LDA to reduce the dimensionality of image data for efficient classification.
Classification: Uses a KNN classifier (k=1) to distinguish between face (label=1) and non-face (label=0) images.
Evaluation: 
Plots classification accuracy as a function of the number of non-face images in the training set.
Compares accuracy with different train-test splits (70-30 vs. 50-50) using PCA.


Visualization: Generates plots to visualize the impact of non-face data on classification performance.

Dataset
The dataset consists of two directories:

Faces: Contains 400 face images, each resized to 92x112 pixels.
Non-Faces: Contains 550 non-face images, converted to grayscale and resized to 92x112 pixels.
Total Features: Each image is flattened into a 10,304-dimensional vector (92 * 112).
Labels: Faces are labeled as 1, non-faces as 0.

Note: The dataset is assumed to be stored in a Google Drive directory (/content/drive/MyDrive/dataset/) with subdirectories faces and nonfaces. Update the path in the notebook if your dataset is stored elsewhere.
Requirements
To run the project, you need the following dependencies:

Python 3.6+
Jupyter Notebook
Libraries:
numpy
matplotlib
Pillow (PIL)
scikit-learn
google-colab (for Google Drive integration, optional if running locally)



Install the dependencies using:
pip install numpy matplotlib pillow scikit-learn google-colab

Setup

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Prepare the Dataset:

Place the faces and nonfaces directories in the appropriate path (e.g., /content/drive/MyDrive/dataset/ if using Google Colab, or a local directory).
Ensure the dataset structure matches:dataset/
├── faces/
│   ├── dir1/
│   ├── dir2/
│   └── ...
└── nonfaces/
    ├── dir1/
    ├── dir2/
    └── ...




Install Dependencies:Run the pip command above to install required libraries.

Open the Notebook:Launch Jupyter Notebook:
jupyter notebook Faces_VS_Non-Faces.ipynb

Alternatively, open the notebook in Google Colab and mount your Google Drive if the dataset is stored there.


Usage

Run the Notebook:
Execute all cells in `Faces_VS_Non-Faces



