"""Loss Given Default (LGD) and Exposure at Default (EAD).

Definitions used in this project
--------------------------------
EAD (at default)  : outstanding principal when the loan charges off,
                    ``funded_amnt - total_rec_prncp``.
Recovery          : post charge-off collections, ``recoveries``.
LGD               : ``1 - recoveries / EAD``, clipped to [0, 1].

Only defaulted loans carry a realised LGD, which is the classic "LGD is only
observed on defaults" sample-selection problem discussed in the report.

Expected loss is then ``EL = PD x LGD x EAD``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .utils import save_fig


def compute_lgd_ead(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ead``, ``recovery`` (amount and rate) and ``lgd`` columns.

    Rows without a resolved default are returned with NaN in those columns.
    """
    out = df.copy()
    funded = pd.to_numeric(out["funded_amnt"], errors="coerce")
    rec_prncp = pd.to_numeric(out["total_rec_prncp"], errors="coerce").fillna(0.0)
    recoveries = pd.to_numeric(out["recoveries"], errors="coerce").fillna(0.0)

    ead = (funded - rec_prncp).clip(lower=0.0)
    out["ead"] = np.where(out["default_flag"] == 1, ead, np.nan)
    out["recovery_amount"] = np.where(out["default_flag"] == 1, recoveries, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lgd = 1.0 - recoveries / ead.replace(0.0, np.nan)
    out["lgd"] = np.where(out["default_flag"] == 1, lgd, np.nan)
    out["lgd"] = out["lgd"].clip(0.0, 1.0)
    out["ead_rate"] = np.where(out["default_flag"] == 1, ead / funded.replace(0.0, np.nan), np.nan)
    return out


def lgd_eda_table(lgd_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of realised LGD, overall and by grade."""
    d = lgd_df.dropna(subset=["lgd"])
    overall = d["lgd"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).rename("lgd").to_frame().T
    overall.insert(0, "group", "portfolio")
    by_grade = (
        d.groupby("grade")["lgd"]
        .agg(n="size", mean="mean", median="median", std="std")
        .reset_index()
        .rename(columns={"grade": "group"})
    )
    by_grade["group"] = by_grade["group"].astype(str)
    out = pd.concat([overall[["group", "count", "mean", "std", "min", "50%", "max"]]
                     .rename(columns={"count": "n", "50%": "median"}),
                     by_grade[["group", "n", "mean", "std", "median"]]], ignore_index=True)
    return out.round(4)


def prepare_lgd_matrix(lgd_df: pd.DataFrame, num_features, cat_features) -> Tuple[pd.DataFrame, pd.Series]:
    """Feature matrix (origination variables) and target for the LGD models."""
    d = lgd_df.dropna(subset=["lgd"]).copy()
    X = d[list(num_features) + list(cat_features)].copy()
    for c in num_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_features:
        X[c] = X[c].astype(str)
    y = d["lgd"].astype(float)
    return X, y


def fit_lgd_gamma(X_train: pd.DataFrame, y_train: pd.Series, num_features, cat_features) -> Tuple[object, pd.DataFrame]:
    """Gamma GLM with a log link (the 'Beta/Gamma-type' LGD model).

    LGD is squeezed away from the 0/1 boundary and the mean response is modeled
    with a Gamma family, which respects the positive, right-skewed support.
    """
    n = len(y_train)
    y_squeezed = (y_train * (n - 1) + 0.5) / n
    design = _one_hot_design(X_train, num_features, cat_features)
    model = sm.GLM(y_squeezed, design, family=sm.families.Gamma(link=sm.families.links.Log()))
    res = model.fit(maxiter=200)
    coef = pd.DataFrame({"variable": list(res.params.index), "coefficient": list(res.params.values)})
    return res, coef


def _one_hot_design(X: pd.DataFrame, num_features, cat_features) -> pd.DataFrame:
    num = X[list(num_features)].astype(float)
    num = num.fillna(num.median())
    dummies = pd.get_dummies(X[list(cat_features)].astype(str), drop_first=True, dtype=float)
    design = pd.concat([num, dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    return design


def predict_lgd_gamma(res, X: pd.DataFrame, num_features, cat_features) -> np.ndarray:
    """Predict LGD with the fitted Gamma GLM (re-using the training design)."""
    design = _one_hot_design(X, num_features, cat_features)
    design = design.reindex(columns=res.params.index, fill_value=0.0)
    return np.clip(res.predict(design), 0.0, 1.0)


def fit_lgd_tree(X_train: pd.DataFrame, y_train: pd.Series, num_features, cat_features, seed: int = 42):
    """Gradient-boosting regressor for LGD (challenger to the Gamma GLM)."""
    from lightgbm import LGBMRegressor

    pre = _lgd_preprocessor(num_features, cat_features)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "reg",
                LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=100,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=seed,
                    verbose=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train.to_numpy())
    return pipe


def _lgd_preprocessor(num_features, cat_features) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(num_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02, sparse_output=False), list(cat_features)),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Two-part LGD model (block 4)
# ---------------------------------------------------------------------------
# Realised LGD piles up at 0 (full recovery) and 1 (no recovery), which a single
# regression cannot represent. The two-part model splits the problem:
#   Part 1  P(boundary | x)  : logistic model for the event LGD == 1 (no recovery),
#                              i.e. recoveries == 0.
#   Part 2  E[LGD | x, LGD<1]: severity model on the strictly-interior LGD values,
#                              fitted as a fractional (quasi-Beta) GLM on the
#                              positive part, which respects the (0, 1) support.
# The combined estimate is E[LGD] = P(LGD=1) * 1 + (1 - P(LGD=1)) * E[LGD | LGD<1].
BOUNDARY_TOL = 1e-9


def lgd_boundary_flags(y: pd.Series) -> pd.Series:
    """Boolean flag: the loan realised the no-recovery boundary ``LGD == 1``."""
    return (pd.to_numeric(y, errors="coerce") >= 1.0 - BOUNDARY_TOL).astype(int)


def fit_lgd_two_part(
    X_train: pd.DataFrame, y_train: pd.Series, num_features, cat_features, seed: int = 42
) -> Dict[str, Any]:
    """Fit the two-part LGD model (boundary logistic + interior fractional GLM).

    Returns a dict holding both fitted components plus the boundary rate, so the
    two parts can be scored and reported separately.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=float)
    flag = (y >= 1.0 - BOUNDARY_TOL).astype(int)
    boundary_rate = float(flag.mean())

    # Part 1: logistic boundary model on the raw features.
    pre = _lgd_preprocessor(num_features, cat_features)
    clf = Pipeline(
        [
            ("prep", pre),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=seed)),
        ]
    )
    clf.fit(X_train, flag)

    # Part 2: fractional (quasi-Beta) GLM on the strictly-interior LGD values.
    interior = flag == 0
    sev_res = None
    if interior.sum() >= 50:
        X_int = X_train.loc[interior]
        y_int = y[interior]
        n = len(y_int)
        # Squeeze into (0, 1) so the logit link is finite, then fit a Binomial
        # (fractional-response / quasi-Beta) GLM: the standard two-part severity
        # choice when the response is a continuous proportion.
        y_frac = (y_int * (n - 1) + 0.5) / n
        design = _one_hot_design(X_int, num_features, cat_features)
        model = sm.GLM(y_frac, design, family=sm.families.Binomial())
        sev_res = model.fit(maxiter=200)
    return {"boundary": clf, "severity": sev_res, "boundary_rate": boundary_rate}


def predict_lgd_two_part(model: Dict[str, Any], X: pd.DataFrame, num_features, cat_features) -> np.ndarray:
    """Combined E[LGD] from the two-part model, clipped to ``[0, 1]``."""
    p_boundary = model["boundary"].predict_proba(X)[:, 1]
    sev_res = model["severity"]
    if sev_res is None:  # pragma: no cover - defensive
        interior = np.zeros(len(X))
    else:
        design = _one_hot_design(X, num_features, cat_features)
        design = design.reindex(columns=sev_res.params.index, fill_value=0.0)
        interior = np.asarray(sev_res.predict(design), dtype=float)
    combined = p_boundary * 1.0 + (1.0 - p_boundary) * interior
    return np.clip(combined, 0.0, 1.0)


def lgd_by_grade_calibration(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, grade_col: str = "grade"
) -> pd.DataFrame:
    """Mean predicted vs mean observed LGD by grade (the realistic use case).

    Loan-level LGD is noisy; the portfolio/grade aggregation is what a bank
    actually consumes, so this table is the primary LGD calibration check.
    """
    tab = pd.DataFrame(
        {
            "grade": df[grade_col].astype(str).to_numpy(),
            "observed": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    out = (
        tab.groupby("grade")
        .agg(n=("observed", "size"), mean_observed=("observed", "mean"), mean_predicted=("predicted", "mean"))
        .reset_index()
    )
    out["gap"] = out["mean_predicted"] - out["mean_observed"]
    overall = pd.DataFrame(
        [
            {
                "grade": "portfolio",
                "n": int(len(tab)),
                "mean_observed": round(float(tab["observed"].mean()), 4),
                "mean_predicted": round(float(tab["predicted"].mean()), 4),
                "gap": round(float(tab["predicted"].mean() - tab["observed"].mean()), 4),
            }
        ]
    )
    return pd.concat([out.round(4), overall], ignore_index=True)


def lgd_distribution_plot(lgd_df: pd.DataFrame, name: str = "lgd_distribution.png"):
    """Histogram of realised LGD with the portfolio mean marked."""
    d = lgd_df.dropna(subset=["lgd"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d["lgd"], bins=50, color="steelblue", alpha=0.85)
    ax.axvline(d["lgd"].mean(), color="crimson", ls="--", label=f"mean = {d['lgd'].mean():.3f}")
    ax.set_xlabel("LGD = 1 - recoveries / EAD")
    ax.set_ylabel("Number of defaulted loans")
    ax.set_title("Realised LGD distribution (defaulted loans)")
    ax.legend()
    return save_fig(fig, name)


def lgd_scatter_plot(y_true, pred_gamma, pred_tree, name: str = "lgd_model_comparison.png"):
    """Predicted vs realised LGD for the Gamma GLM and the tree model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pred, label in zip(axes, [pred_gamma, pred_tree], ["Gamma GLM", "LightGBM"]):
        ax.scatter(y_true, pred, s=4, alpha=0.15, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Realised LGD")
        ax.set_ylabel("Predicted LGD")
        ax.set_title(f"{label} (MAE={mean_absolute_error(y_true, pred):.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    return save_fig(fig, name)


def lgd_distribution_plot_by_grade(lgd_df: pd.DataFrame, name: str = "lgd_by_grade.png"):
    """Box plot of realised LGD by lender grade."""
    d = lgd_df.dropna(subset=["lgd"])
    grades = sorted(d["grade"].astype(str).unique())
    data = [d.loc[d["grade"].astype(str) == g, "lgd"].values for g in grades]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=grades, showfliers=False)
    ax.set_xlabel("Lender grade (A = lowest risk)")
    ax.set_ylabel("Realised LGD")
    ax.set_title("Realised LGD by lender grade")
    return save_fig(fig, name)


def expected_loss_table(df: pd.DataFrame, pd_hat: np.ndarray, lgd_hat: np.ndarray) -> pd.DataFrame:
    """Per-loan expected loss, ready for aggregation."""
    ead_col = "funded_amnt" if "funded_amnt" in df.columns else "loan_amnt"
    out = df[["grade", "loan_amnt", "default_flag"]].copy()
    out["pd"] = np.asarray(pd_hat, dtype=float)
    out["lgd"] = np.asarray(lgd_hat, dtype=float)
    out["ead"] = pd.to_numeric(df[ead_col], errors="coerce").to_numpy()
    out["el"] = out["pd"].to_numpy() * out["lgd"].to_numpy() * out["ead"].to_numpy()
    out["grade"] = out["grade"].astype(str)
    return out


def aggregate_el(el_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Expected loss aggregated by grade and for the whole portfolio."""
    by_grade = (
        el_df.groupby("grade")
        .agg(
            n_loans=("el", "size"),
            mean_pd=("pd", "mean"),
            mean_lgd=("lgd", "mean"),
            ead=("ead", "sum"),
            expected_loss=("el", "sum"),
            realised_defaults=("default_flag", "sum"),
        )
        .reset_index()
    )
    by_grade["el_rate"] = by_grade["expected_loss"] / by_grade["ead"]
    portfolio = pd.DataFrame(
        [
            {
                "n_loans": int(len(el_df)),
                "mean_pd": float(el_df["pd"].mean()),
                "mean_lgd": float(el_df["lgd"].mean()),
                "ead": float(el_df["ead"].sum()),
                "expected_loss": float(el_df["el"].sum()),
                "el_rate": float(el_df["el"].sum() / el_df["ead"].sum()),
                "realised_defaults": int(el_df["default_flag"].sum()),
            }
        ]
    )
    return {"by_grade": by_grade.round(4), "portfolio": portfolio.round(4)}
