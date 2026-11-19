"""Train and compare Ridge and Lasso regression on California Housing."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTPUT_DIR, "housing_model.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "housing_model_results.csv")

def prepare_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy().drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.20, random_state=42)

def train():
    X_train, X_test, y_train, y_test = prepare_data()
    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=20000))
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "MSE": mse, "RMSE": np.sqrt(mse), "R2 Score": r2_score(y_test, pred)})
    results = pd.DataFrame(rows).sort_values("R2 Score", ascending=False)
    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_PATH)
    results.to_csv(RESULTS_PATH, index=False)
    print("Training complete.")
    print(results.to_string(index=False))
    print("Best model:", best_name)
    print("Saved:", MODEL_PATH)
    return best_model, results

if __name__ == "__main__":
    train()
