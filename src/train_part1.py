import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

train = pd.read_csv(BASE_DIR / "data/IMT2024018_train_var1.csv")

X = train.drop("y", axis=1)
y = train["y"]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for degree in range(1, 11):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", LinearRegression())
    ])

    score = -cross_val_score(
        model, X, y, cv=cv,
        scoring="neg_mean_squared_error"
    ).mean()

    results.append(["Linear", degree, np.nan, score])

ridge_alphas = [0.5, 1, 2, 5, 10, 20, 50, 100]

for degree in range(1, 11):
    for alpha in ridge_alphas:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=alpha))
        ])

        score = -cross_val_score(
            model, X, y, cv=cv,
            scoring="neg_mean_squared_error"
        ).mean()

        results.append(["Ridge", degree, alpha, score])

lasso_alphas = [0.004, 0.006, 0.008, 0.0085, 0.010, 0.015, 0.020]

for degree in range(1, 11):
    for alpha in lasso_alphas:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("reg", Lasso(alpha=alpha, max_iter=200000))
        ])

        score = -cross_val_score(
            model, X, y, cv=cv,
            scoring="neg_mean_squared_error"
        ).mean()

        results.append(["Lasso", degree, alpha, score])

results = pd.DataFrame(
    results,
    columns=["model", "degree", "alpha", "cv_mse"]
)

results = results.sort_values("cv_mse")
results.to_csv(BASE_DIR / "results/part1_model_comparison.csv", index=False)

final_model = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("scale", StandardScaler()),
    ("reg", Lasso(alpha=0.0085, max_iter=200000))
])

final_model.fit(X, y)

joblib.dump(final_model, BASE_DIR / "models/part1_model.pkl")

print("Best model: degree 5 Lasso")
print("Alpha:", 0.0085)
print("CV MSE:", results.iloc[0]["cv_mse"])
print("Model saved.")
