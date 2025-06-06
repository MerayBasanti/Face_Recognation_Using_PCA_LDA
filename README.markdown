# Faces vs. Non-Faces Classification

## Project Overview
This project implements a face vs. non-face image classification system using a K-Nearest Neighbors (KNN) classifier with dimensionality reduction techniques, specifically Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). The goal is to evaluate the performance of KNN with PCA and LDA on a dataset of face and non-face images, analyzing how accuracy varies with different proportions of non-face images and train-test split ratios.

The project is implemented in a Jupyter Notebook (`Faces_VS_Non-Faces.ipynb`) using Python and common machine learning libraries.

## Features
- **Image Preprocessing**: Loads and processes face and non-face images, converting non-face images to grayscale and flattening all images into feature vectors.
- **Dimensionality Reduction**: Applies PCA and LDA to reduce the dimensionality of image data for efficient classification.
- **Classification**: Uses a KNN classifier (k=1) to distinguish between face (label=1) and non-face (label=0) images.
- **Evaluation**: 
  - Plots classification accuracy as a function of the number of non-face images in the training set.
  - Compares accuracy with different train-test splits (70-30 vs. 50-50) using PCA.
- **Visualization**: Generates plots to visualize the impact of non-face data on classification performance.

## Dataset
The dataset consists of two directories:
- **Faces**: Contains 400 face images, each resized to 92x112 pixels.
- **Non-Faces**: Contains 550 non-face images, converted to grayscale and resized to 92x112 pixels.
- **Total Features**: Each image is flattened into a 10,304-dimensional vector (92 * 112).
- **Labels**: Faces are labeled as 1, non-faces as 0.

**Note**: The dataset is assumed to be stored in a Google Drive directory (`/content/drive/MyDrive/dataset/`) with subdirectories `faces` and `nonfaces`. Update the path in the notebook if your dataset is stored elsewhere.

## Requirements
To run the project, you need the following dependencies:
- Python 3.6+
- Jupyter Notebook
- Libraries:
  - `numpy`
  - `matplotlib`
  - `Pillow` (PIL)
  - `scikit-learn`
  - `google-colab` (for Google Drive integration, optional if running locally)

Install the dependencies using:
```bash
pip install numpy matplotlib pillow scikit-learn google-colab
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare the Dataset**:
   - Place the `faces` and `nonfaces` directories in the appropriate path (e.g., `/content/drive/MyDrive/dataset/` if using Google Colab, or a local directory).
   - Ensure the dataset structure matches:
     ```
     dataset/
     ├── faces/
     │   ├── dir1/
     │   ├── dir2/
     │   └── ...
     └── nonfaces/
         ├── dir1/
         ├── dir2/
         └── ...
     ```

3. **Install Dependencies**:
   Run the pip command above to install required libraries.

4. **Open the Notebook**:
   Launch Jupyter Notebook:
   ```bash
   jupyter notebook Faces_VS_Non-Faces.ipynb
   ```
   Alternatively, open the notebook in Google Colab and mount your Google Drive if the dataset is stored there.

## Usage
1. **Run the Notebook**:
   - Execute all cells in `Faces_VS_Non-Faces.ipynb` sequentially.
   - Ensure the dataset path in the `load_images` function matches your setup.
   - The notebook will:
     - Load and preprocess the images.
     - Shuffle the data.
     - Generate a plot showing accuracy vs. the number of non-face images in the training set (using PCA).
     - Compare accuracy for 70-30 and 50-50 train-test splits with PCA.

2. **Key Outputs**:
   - **Accuracy Plot**: Visualizes how classification accuracy changes with varying non-face data proportions.
   - **Console Output**: Displays the number of PCA components used and the accuracy for different train-test splits.
     - Example: 
       ```
       Using 47 components to retain 85.0% of the variance
       Using 40 components to retain 85.0% of the variance
       Accuracy of KNN classifier with k=1 after PCA with alpha=0.7: 0.95
       Accuracy of KNN classifier with k=1 after PCA with alpha=0.5: 0.925
       ```

3. **Modifying Parameters**:
   - Adjust the `step` parameter in `acc_vs_non_faces_in_training` to change the number of steps for varying non-face proportions.
   - Uncomment the LDA section in `acc_vs_non_faces_in_training` to test LDA instead of PCA.
   - Modify the `alpha` parameter (e.g., 0.85) in PCA to retain a different percentage of variance.

## Results
- **PCA Performance**:
  - With a 70-30 train-test split, the KNN classifier (k=1) achieves an accuracy of 95% after PCA with 47 components (retaining 85% variance).
  - With a 50-50 train-test split, the accuracy is 92.5% with 40 components.
- **Accuracy Plot**: The plot (generated by `acc_vs_non_faces_in_training`) shows how accuracy varies with the number of non-face images in the training set.

## Notes
- The notebook assumes access to Google Drive for dataset loading. If running locally, modify the `load_images` function to use local file paths.
- The LDA implementation is commented out in the provided notebook. To test LDA, uncomment the relevant line in the `acc_vs_non_faces_in_training` function.
- The dataset is not included in this repository due to size constraints. Ensure you have the dataset available before running the notebook.

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Suggestions for optimizing the code, adding new features, or improving documentation are welcome!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on this repository or contact me at [meraybasanti@gmail.com].
