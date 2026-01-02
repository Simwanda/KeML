# src/data_utils.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

from .config import DB_CSV_PATH, RANDOM_STATE, TEST_SIZE

# -----------------------
# Helpers for EC4-style quantities (simplified)
# -----------------------

def steel_area_rect_hollow(H: float, B: float, t: float) -> float:
    """
    Approx. steel tube area for a rectangular hollow section:
    As = H*B - (H - 2t)*(B - 2t)
    Units consistent with input.
    """
    return H * B - (H - 2.0 * t) * (B - 2.0 * t)

def concrete_area_rect(H: float, B: float, t: float) -> float:
    """
    Concrete core area inside the steel tube:
    Ac = (H - 2t)*(B - 2t)
    """
    return (H - 2.0 * t) * (B - 2.0 * t)

def compute_secondary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute secondary features as described in Table 4 of the paper.
    Assumes primary columns: H, B, t, L, e, fy, fc_prime.
    """
    out = df.copy()

    H = out["H"].values
    B = out["B"].values
    t = out["t"].values
    L = out["L"].values
    e = out["e"].values
    fy = out["fy"].values
    fc = out["fc_prime"].values

    # Areas and plastic axial resistance
    As = steel_area_rect_hollow(H, B, t)
    Ac = concrete_area_rect(H, B, t)

    Asfy = As * fy
    Acfc = Ac * fc
    Npl  = Asfy + Acfc  # plastic resistance

    # Slenderness of composite column (λ) and e/H, H/t, δ, α
    lam = 2.0 * (3.0 ** 0.5) * L / H      # λ = 2*sqrt(3)*L/H
    e_over_H = e / H
    H_over_t = H / t
    delta = Asfy / Npl
    alpha = As / Ac

    # Material stiffness (approx)
    Es = 2.0e5  # MPa, adjust if you have exact values
    Ec = 4700.0 * np.sqrt(fc)  # EC2-style, MPa

    # Effective flexural stiffness (simplified)
    # Treat section as rectangular beam with second moment ≈ BH^3/12 (outer, inner)
    I_steel = (B * H**3 - (B - 2.0*t) * (H - 2.0*t)**3) / 12.0
    I_conc  = ((B - 2.0*t) * (H - 2.0*t)**3) / 12.0

    EIeff_s = Es * I_steel
    EIeff_c = Ec * I_conc

    # Elastic buckling resistance
    Ncr = (np.pi**2) * (EIeff_s + 0.6 * EIeff_c) / (L**2)

    # Reduction factor χ (EC4-style slenderness; here we approximate)
    # λ_bar = sqrt(Npl / Ncr), with some lower bound to avoid zero-division
    lam_bar = np.sqrt(np.maximum(Npl / np.maximum(Ncr, 1e-6), 1e-8))
    # Basic EC3/EC4 style approximation: chi = 1 / (phi + sqrt(phi^2 - lam_bar^2))
    # where phi ~ 0.5 * (1 + alpha*(lam_bar - 0.2) + lam_bar**2)
    # Here we use alpha_chi = 0.21 as a typical value.
    alpha_chi = 0.21
    phi = 0.5 * (1.0 + alpha_chi * (lam_bar - 0.2) + lam_bar**2)
    chi = 1.0 / (phi + np.sqrt(np.maximum(phi**2 - lam_bar**2, 1e-8)))

    # Add to DataFrame
    out["Asfy"]   = Asfy
    out["Acfc"]   = Acfc
    out["Npl"]    = Npl
    out["lambda"] = lam
    out["e_over_H"] = e_over_H
    out["H_over_t"] = H_over_t
    out["delta"]    = delta
    out["alpha"]    = alpha
    out["EIeff_s"]  = EIeff_s
    out["EIeff_c"]  = EIeff_c
    out["Ncr"]      = Ncr
    out["chi"]      = chi

    return out

def load_and_prepare_database(
    csv_path: Optional[str] = None,
    use_empirical_column: bool = True,
    empirical_col_name: str = "N_empirical"
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load the database, compute secondary features, and prepare:
      - X: features DataFrame
      - y: experimental ultimate strength
      - y_emp: empirical baseline prediction (Han, Naser, EC4, etc.)
      - feature_names: list of feature columns actually used

    If 'use_empirical_column' is False OR empirical_col_name not in df,
    baseline will default to Npl (plastic axial resistance).
    """
    path = csv_path or DB_CSV_PATH
    df_raw = pd.read_csv(path)

    # Compute secondary features
    df = compute_secondary_features(df_raw)

    # Target
    y = df["Nu_exp"].astype(float)

    # Knowledge term (empirical eq.)
    if use_empirical_column and empirical_col_name in df.columns:
        y_emp = df[empirical_col_name].astype(float)
    else:
        # fallback: use Npl as baseline knowledge
        y_emp = df["Npl"].astype(float)

    # Select 15 input features as in the paper (removing chi, H/t, delta, Npl)
    # You can adjust this list to match exactly your version of the database.
    primary_feats = ["H", "B", "t", "L", "e", "fy", "fc_prime"]
    secondary_feats = [
        "e_over_H", "lambda", "Asfy", "Acfc",
        "alpha", "EIeff_s", "EIeff_c", "Ncr",
    ]
    # Add Npl, H_over_t, delta, chi if you want the full 19 and then drop as in the paper.
    # For now we keep 15 features above; modify if needed.
    feature_names = primary_feats + secondary_feats

    X = df[feature_names].astype(float)

    return X, y, y_emp, feature_names

def train_test_split_rcfst(
    X: pd.DataFrame,
    y: pd.Series,
    y_emp: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Train-test split that keeps empirical baseline aligned.
    """
    X_train, X_test, y_train, y_test, y_emp_train, y_emp_test = train_test_split(
        X, y, y_emp,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test, y_emp_train, y_emp_test
