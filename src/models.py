# src/models.py

from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from .config import ARTIFACTS_DIR

def build_original_models() -> Dict[str, Any]:
    """
    Build a dict of original ML models with reasonable defaults.
    These are the models that learn y directly from X (no knowledge).
    """
    models = {
        "KNN": KNeighborsRegressor(
            n_neighbors=3,
            weights="distance"
        ),
        "SVR": SVR(
            kernel="rbf",
            C=100.0,
            gamma="scale",
            epsilon=0.1
        ),
        "DT": DecisionTreeRegressor(
            max_depth=20,
            random_state=0
        ),
        "ANN": MLPRegressor(
            hidden_layer_sizes=(80, 80),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=500,
            random_state=0
        ),
        "RF": RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            random_state=0,
            n_jobs=-1
        ),
        "XGB": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=0
        )
    }
    return models

# -----------------------------
# (Optional) Optuna-based tuning for one model type (example: XGB)
# -----------------------------

import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def objective_xgb(trial: optuna.Trial, X, y) -> float:
    """
    Optuna objective function to tune XGBRegressor on RMSE (5-fold CV).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=0,
            **params
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)

    return float(np.mean(rmses))

def tune_xgb_with_optuna(
    X,
    y,
    n_trials: int = 50,
    study_name: str = "xgb_keml_tuning"
) -> XGBRegressor:
    """
    Run Optuna tuning for XGBRegressor and return the best-fitted model.

    Saves the Optuna study and best model under artifacts/.
    """
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize"
    )
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=n_trials)

    best_params = study.best_params
    # Refit best model on full training data
    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=0,
        **best_params
    )
    best_model.fit(X, y)

    # Save study and model
    optuna_path = ARTIFACTS_DIR / "optuna" / f"{study_name}_study.pkl"
    model_path = ARTIFACTS_DIR / "models" / f"{study_name}_best_xgb.joblib"

    joblib.dump(study, optuna_path)
    joblib.dump(best_model, model_path)

    return best_model
