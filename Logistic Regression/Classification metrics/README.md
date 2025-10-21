## README

### Project Title

**Breast Cancer Classification with Logistic Regression**

### Overview

This project demonstrates the process of training and evaluating a logistic regression model on the Breast Cancer dataset using several classification performance metrics. It provides a concise example of how to implement a complete classification pipeline in Python using scikit-learn.

### Objectives

* Load and explore the Breast Cancer dataset
* Train a Logistic Regression model
* Evaluate the model using multiple classification metrics
* Visualize the ROC curve

### Contents

The notebook includes the following main sections:

1. **Importing Libraries** – Required packages such as pandas, scikit-learn, and matplotlib.
2. **Loading Dataset** – Uses the built-in Breast Cancer dataset from scikit-learn.
3. **Data Preparation** – Creates a DataFrame, splits data into training and testing sets, and applies normalization using `StandardScaler`.
4. **Model Training** – Trains a logistic regression model with an increased iteration limit (`max_iter=8000`).
5. **Prediction and Evaluation** – Computes the following metrics:

   * Accuracy
   * Recall
   * Precision (for both classes)
   * F1 Score
   * Confusion Matrix
   * Classification Report
   * ROC Curve
6. **Visualization** – Plots the ROC curve to assess the classifier's performance visually.

### Dependencies

* Python 3.x
* pandas
* scikit-learn
* matplotlib

To install dependencies:

```bash
pip install pandas scikit-learn matplotlib
```

### How to Run

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook Classification_metrics.ipynb
   ```
2. Run all cells in order.
3. Review the printed metrics and ROC plot to interpret the model’s performance.

### Results

The logistic regression model achieves strong performance on the Breast Cancer dataset, demonstrating the effectiveness of simple linear models for binary classification tasks.

### License

This project is free to use.
