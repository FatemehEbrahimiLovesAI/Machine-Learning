# Persian Digit Classification Project

This project implements a pipeline for loading, preprocessing, and
training models on the **Hoda Persian handwritten digit dataset**.\
The repository includes dataset preparation code and a Jupyter notebook
for model development.

------------------------------------------------------------------------

## Project Structure

    .
    ├── Data_hoda_full.mat  # The mat Dataset
    ├── dataset.py          # Data loading, preprocessing, resizing, flattening
    ├── main.ipynb          # Model training workflow (notebook)
    └── README.md           # Documentation

------------------------------------------------------------------------

## Dataset

The project uses the **Hoda Persian Digit Dataset**, loaded from a
`.mat` file.

### Preprocessing steps (`dataset.py`):

-   Load dataset from MAT file\
-   Split into train and test subsets\
-   Resize images to **10×10** using OpenCV\
-   Flatten images into **100‑dimensional vectors**\
-   Return NumPy arrays ready for ML/DL models

### Example Function

``` python
x_train, x_test, y_train, y_test = load_dataset(train_value=1000, test_value=200)
```

------------------------------------------------------------------------

## How to Use

### 1. Install Dependencies

``` bash
pip install numpy scipy opencv-python
```

### 2. Call the Loader

``` python
from dataset import load_dataset
x_train, x_test, y_train, y_test = load_dataset()
```

### 3. Open the Notebook

Use `main.ipynb` to train and evaluate machine learning or deep learning
models.

------------------------------------------------------------------------

## Model Training

The notebook includes:

-   Dataset loading\
-   Normalization\
-   Model definition\
-   Training and evaluation\
-   Visualization of results

------------------------------------------------------------------------

## Notes

-   Make sure to update the dataset path inside `dataset.py`:

        C:\Users\pc\Documents\university file\Deep Learning\PersianDigit\Data_hoda_full.mat

-   You can replace it with a relative path for portability.

------------------------------------------------------------------------

## License

This project is for educational and research purposes. 
