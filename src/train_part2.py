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

train = pd.read_csv(BASE_DIR / "data/IMT2024018_train_var2.csv")

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

ridge_alphas = [0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.5, 2.0, 3.0]

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

lasso_alphas = [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003]

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
results.to_csv(BASE_DIR / "results/part2_model_comparison.csv", index=False)

final_model = Pipeline([
    ("poly", PolynomialFeatures(degree=10, include_bias=False)),
    ("scale", StandardScaler()),
    ("reg", Ridge(alpha=0.99))
])

final_model.fit(X, y)

joblib.dump(final_model, BASE_DIR / "models/part2_model.pkl")

print("Best model: degree 10 Ridge")
print("Alpha:", 0.99)
print("CV MSE:", results.iloc[0]["cv_mse"])
print("Model saved.")
