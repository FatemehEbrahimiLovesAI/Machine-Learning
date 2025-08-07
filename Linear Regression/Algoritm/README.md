##  README - Linear Regression from Scratch (Grid Search)

###  Project Overview

This project demonstrates a simple implementation of Linear Regression from scratch using NumPy.
The goal is to fit a line to a dataset (`student_loan_train.csv`) by optimizing the weight (`w1`) using brute-force grid search and calculating the Mean Squared Error (MSE) as the loss function.

### How It Works

1. Model definition:
   The model uses the formula y_hat = w1 * x + w0.

2. Loss function:
   Mean Squared Error is used to evaluate the performance of the model.

3. Grid Search Optimization:

   * w1 is searched over a range of values using np.linspace().
   * For each w1, the corresponding MSE is calculated.
   * The w1 with the lowest MSE is selected as the best.

4. Visualization:

   * The original data points are shown using scatter().
   * The best-fit line is plotted using plot().

---

### Technologies Used

* Python 3
* NumPy
* Pandas
* Matplotlib

---

### Output

The script prints the best value of w1 and shows a plot with:

* Blue dots: Original data
* Red line: Fitted linear regression line
