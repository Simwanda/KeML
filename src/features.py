# src/features.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression, f_regression

from xgboost import XGBRegressor

def standardize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardization: (x - mean) / std, fitted on training set only.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    return X_train_scaled, X_test_scaled, scaler

def compute_combined_feature_weights(
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """
    Approximation of the combined weight values (CWV) used in the paper:

    - Train SVR and XGB on (X, y) and use their absolute coefficients / feature importances.
    - Also compute mutual_info_regression and f_regression scores.
    - Normalize within each method and then average across methods.

    Returns a DataFrame with per-feature weights and method breakdown.
    """
    feature_names = X.columns.tolist()
    X_val = X.values
    y_val = y.values

    # 1) SVR (RBF kernel) -> use permutation importance via abs dual coefs? Instead we
    # approximate by fitting linear SVR for feature weights. For simplicity, we use RBF
    # but then feature importance ~ 0; so here use linear kernel.
    svr = SVR(kernel="linear")
    svr.fit(X_val, y_val)
    svr_importance = np.abs(svr.coef_).ravel()

    # 2) XGBoost feature importances (gain-based)
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0
    )
    xgb.fit(X_val, y_val)
    xgb_importance = xgb.feature_importances_

    # 3) mutual information
    mi = mutual_info_regression(X_val, y_val, random_state=0)

    # 4) f_regression (F-statistics)
    f_vals, _ = f_regression(X_val, y_val)
    f_vals = np.nan_to_num(f_vals, nan=0.0)

    # Normalize each method
    def normalize(v):
        s = v.sum()
        return v / s if s > 0 else np.zeros_like(v)

    m_svr = normalize(svr_importance)
    m_xgb = normalize(xgb_importance)
    m_mi  = normalize(mi)
    m_f   = normalize(f_vals)

    # Mean of normalized weights
    m_avg = (m_svr + m_xgb + m_mi + m_f) / 4.0

    # Combined Weight Value (CWV) by dividing by max
    cwv = m_avg / np.max(m_avg)

    df_weights = pd.DataFrame({
        "feature": feature_names,
        "weight_SVR": m_svr,
        "weight_XGB": m_xgb,
        "weight_MI":  m_mi,
        "weight_F":   m_f,
        "mean_weight": m_avg,
        "CWV": cwv
    }).sort_values("CWV", ascending=False).reset_index(drop=True)

    return df_weights

def drop_highly_correlated_features(
    X: pd.DataFrame,
    cwv_df: pd.DataFrame,
    corr_threshold: float = 0.75
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Example helper: drop highly correlated features guided by CWV.
    It keeps the more important feature (higher CWV) and drops
    the less important one if |corr| > threshold.

    Returns reduced X and list of retained features.
    """
    corr = X.corr().abs()
    feature_order = cwv_df["feature"].tolist()

    # Start with all features available
    to_keep = set(feature_order)
    to_drop = set()

    for i, fi in enumerate(feature_order):
        if fi in to_drop:
            continue
        for fj in feature_order[i+1:]:
            if fj in to_drop:
                continue
            if corr.loc[fi, fj] > corr_threshold:
                # Drop the lower-weight feature
                wi = cwv_df.loc[cwv_df["feature"] == fi, "CWV"].values[0]
                wj = cwv_df.loc[cwv_df["feature"] == fj, "CWV"].values[0]
                if wi >= wj:
                    to_drop.add(fj)
                else:
                    to_drop.add(fi)
                    break

    retained = list(to_keep - to_drop)
    retained = [f for f in feature_order if f in retained]  # keep CWV order

    X_reduced = X[retained].copy()
    return X_reduced, retained
