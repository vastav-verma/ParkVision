"""
Model Training Pipeline
========================
Trains and evaluates multiple regression models for parking demand prediction.
Saves the best model and encoders for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

CATEGORICAL_COLS = ["event_type", "venue", "city", "registration_mode", "weather"]
FEATURE_COLS = [
    "event_type", "venue", "city", "month", "day_of_week", "is_weekend",
    "is_holiday", "expected_attendees", "duration_hours", "registration_mode",
    "is_paid_event", "num_sessions", "has_vip_guests", "alumni_batch_size",
    "weather", "is_rainy", "temperature_celsius", "total_parking_capacity",
    "public_transport_score", "distance_from_city_center_km", "carpooling_promoted"
]
TARGET_COL = "parking_demand"


def load_or_generate_data(csv_path: str = "data/parking_dataset.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from utils.data_generator import generate_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_dataset(2000)
        df.to_csv(csv_path, index=False)
        return df


def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    X = df[FEATURE_COLS]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, encoders


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "R2": round(r2, 4), "MAPE(%)": round(mape, 2)}


def train_all_models():
    print("📂 Loading data...")
    df = load_or_generate_data()
    X, y, encoders = preprocess(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=15,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                        max_depth=5, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.08,
                                          max_depth=6, random_state=42, verbosity=0)

    results = []
    trained_models = {}
    print("\n🏋️  Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate(name, model, X_test, y_test)
        results.append(res)
        trained_models[name] = model
        print(f"  ✅ {name:25s}  R²={res['R2']:.4f}  MAE={res['MAE']:.1f}  RMSE={res['RMSE']:.1f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    # Best model by R²
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n🏆 Best Model: {best_name} (R²={results_df.iloc[0]['R2']})")

    # Feature importance (for tree models)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        feature_importance = fi

    # Save everything
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "best_model": best_model,
        "best_model_name": best_name,
        "all_models": trained_models,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "results_df": results_df,
        "feature_importance": feature_importance,
        "categorical_cols": CATEGORICAL_COLS,
    }
    with open("models/parking_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("💾 Artifacts saved → models/parking_model.pkl")
    return artifacts


if __name__ == "__main__":
    train_all_models()
