import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

model = joblib.load(BASE_DIR / "models/part1_model.pkl")

test = pd.read_csv(BASE_DIR / "data/IMT2024018_test_var1.csv")
X_test = test.drop("y", axis=1, errors="ignore")

prediction = model.predict(X_test)

pd.DataFrame({"y": prediction}).to_csv(
    BASE_DIR / "results/IMT2024018_pred_var1.csv",
    index=False
)

print("Part 1 predictions saved.")
