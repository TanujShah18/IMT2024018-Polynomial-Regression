# Polynomial Regression Assignment

This repository contains the implementation for the polynomial regression assignment. The two parts use different datasets, so the model and polynomial degree are selected separately for each part.

## Final Models

| Part | Polynomial Degree | Regression Model | Alpha | 5-Fold CV MSE |
|---|---:|---|---:|---:|
| Part 1 | 5 | Lasso (L1) | 0.0085 | 0.350415 |
| Part 2 | 10 | Ridge (L2) | 0.99 | 0.272780 |

The models were selected using 5-fold cross-validation with shuffling and `random_state=42`. Polynomial feature generation and standardization are kept inside the sklearn pipeline to avoid data leakage.

## Repository Structure

```text
IMT2024018-Polynomial-Regression/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── IMT2024018_train_var1.csv
│   ├── IMT2024018_test_var1.csv
│   ├── IMT2024018_train_var2.csv
│   ├── IMT2024018_test_var2.csv
│   └── README.md
│
├── src/
│   ├── train_part1.py
│   ├── train_part2.py
│   ├── inference_part1.py
│   └── inference_part2.py
│
│
└── results/
    ├── IMT2024018_pred_var1.csv
    └── IMT2024018_pred_var2.csv
```

## Training and Validation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

From the repository root, run:

```bash
python src/train_part1.py
python src/train_part2.py
```

The training scripts compare Linear Regression, Ridge Regression and Lasso Regression over polynomial degrees 1 to 10 using 5-fold cross-validation. The validation scores are based on mean squared error. The selected models are then fitted on the complete training data and saved in `models/`.

The preprocessing steps are inside the sklearn pipeline, so polynomial expansion and scaling are fitted separately within each cross-validation fold.

## Inference

After training, run:

```bash
python src/inference_part1.py
python src/inference_part2.py
```

The inference scripts load the saved models and generate predictions for the corresponding test datasets.

The final prediction files contain only one column:

```text
y
```

The prediction files are already included in `results/`.

## Results

### Part 1

The selected model is a degree-5 polynomial regression with Lasso regularization (`alpha = 0.0085`). Its 5-fold cross-validation MSE is approximately `0.350415`.

### Part 2

The selected model is a degree-10 polynomial regression with Ridge regularization (`alpha = 0.99`). Its 5-fold cross-validation MSE is approximately `0.272780`.

The test datasets were not used for model selection. They were used only after the final model had been selected and fitted on the complete training data.
