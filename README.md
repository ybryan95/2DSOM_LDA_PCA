# 2DSOM_LDA_PCA

This repository contains MATLAB scripts implementing 2D Self-Organizing Maps (SOM), Linear Discriminant Analysis (LDA), and Principal Component Analysis (PCA). The scripts are designed to work with different types of data distributions and demonstrate the application of these techniques in data visualization and classification.

## Files

1. `h3p1.m`: This script implements a 2D SOM for data visualization. It generates data from uniform, Gaussian, and exponential distributions, and trains a SOM to represent this data. The script also includes a dynamic learning rate and neighborhood size, and visualizes the SOM at each iteration.

2. `h3p2_1.m`: This script demonstrates the use of LDA for data classification. It generates two classes of data, computes the between-class and within-class scatter matrices, and finds the optimal projection vector for class separation. The script also includes a test case for classifying a new data point.

3. `h3p2_2.m`: This script applies PCA and LDA to the Wisconsin Breast Cancer dataset. It first normalizes the data, computes the covariance matrix, and finds the principal components. The data is then projected onto the first two principal components for visualization. The script also applies LDA to the projected data for classification.

## Usage

To run the scripts, open them in MATLAB and press the run button. Make sure the Wisconsin Breast Cancer dataset (`breast_cancer.txt`) is in the same directory as `h3p2_2.m` for it to run correctly.

## Dependencies

These scripts require MATLAB to run. No additional toolboxes are needed.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

