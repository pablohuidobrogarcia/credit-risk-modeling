"""Probability-of-default models: WoE logistic scorecard plus challengers.

Models implemented
------------------
* ``logit``                 : statsmodels Logit on the WoE matrix (the
                              interpretable scorecard; coefficients + odds
                              ratios).
* ``probit``                : statsmodels Probit (GLM link comparison).
* ``logit_l1`` / ``logit_l2``: regularised logistic regression (Lasso/Ridge)
                              with ``C`` tuned by 5-fold cross-validation.
* ``decision_tree``         : CART with cost-complexity pruning.
* ``random_forest``         : bagged trees.
* ``lightgbm``              : gradient boosting.

Linear models consume the WoE matrix (interpretable scorecard). Tree models
consume the raw features through a preprocessing pipeline, which is the usual
industry setup and gives a genuinely different hypothesis class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .woe_iv import WoEEncoder, transform_frame


# ---------------------------------------------------------------------------
# Uniform container
# ---------------------------------------------------------------------------
@dataclass
class FittedModel:
    """A trained PD model with a uniform ``predict_proba`` interface."""

    name: str
    kind: str  # "woe" | "raw"
    estimator: Any
    features: List[str]
    encoders: Optional[Dict[str, WoEEncoder]] = None
    cutoff: float = 0.5
    cv_auc: Optional[float] = None
    cv_auc_std: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted probability of default for every row of ``X``."""
        if self.kind == "woe":
            woe = transform_frame(X, self.encoders)  # type: ignore[arg-type]
            design = sm.add_constant(woe, has_constant="add")
            design = design.reindex(columns=self.features, fill_value=1.0)
            return np.asarray(self.estimator.predict(design), dtype=float)
        return self.estimator.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Linear / GLM models on the WoE scale
# ---------------------------------------------------------------------------
def _design_matrix(woe: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in woe.columns if woe[c].std() > 1e-12]
    return sm.add_constant(woe[keep], has_constant="add")


def _coefficient_table(res: Any, link: str) -> pd.DataFrame:
    """Coefficients with standard errors, p-values, confidence intervals and OR."""
    ci = np.asarray(res.conf_int(), dtype=float)
    names = list(getattr(res.model, "exog_names", None) or res.params.index)
    tab = pd.DataFrame(
        {
            "variable": names,
            "coefficient": np.asarray(res.params, dtype=float),
            "std_err": np.asarray(res.bse, dtype=float),
            "z": np.asarray(res.tvalues, dtype=float),
            "p_value": np.asarray(res.pvalues, dtype=float),
            "ci_lower": ci[:, 0],
            "ci_upper": ci[:, 1],
        }
    )
    tab["odds_ratio"] = np.exp(tab["coefficient"])
    tab["or_ci_lower"] = np.exp(tab["ci_lower"])
    tab["or_ci_upper"] = np.exp(tab["ci_upper"])
    tab["link"] = link
    return tab


def fit_statsmodels_woe(
    name: str,
    link: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    encoders: Dict[str, WoEEncoder],
) -> Tuple[FittedModel, pd.DataFrame]:
    """Fit Logit/Probit on the WoE scale and return (model, coefficient table)."""
    woe = transform_frame(X_train, encoders)
    design = _design_matrix(woe)
    y = np.asarray(y_train, dtype=float)
    model_cls = sm.Logit if link == "logit" else sm.Probit
    res = model_cls(y, design).fit(disp=False, maxiter=200)
    fitted = FittedModel(
        name=name,
        kind="woe",
        estimator=res,
        features=design.columns.tolist(),
        encoders=encoders,
        params={"link": link, "pseudo_r2": float(res.prsquared), "llf": float(res.llf)},
    )
    return fitted, _coefficient_table(res, link)


def fit_regularised_logit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    encoders: Dict[str, WoEEncoder],
    penalty: str = "l2",
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
    cv_sample_size: int = 150_000,
) -> Tuple[FittedModel, pd.DataFrame, pd.DataFrame]:
    """L1/L2 logistic regression on the WoE scale, ``C`` chosen by CV AUC.

    The cross-validation grid runs on a random subsample when the training
    window is large (``cv_sample_size``); the final model is refit on the full
    training sample.
    """
    woe = transform_frame(X_train, encoders)
    y_all = np.asarray(y_train, dtype=int)
    solver = "saga" if penalty == "l1" else "lbfgs"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if len(woe) > cv_sample_size:
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(len(woe), size=cv_sample_size, replace=False))
        woe_cv, y_cv = woe.iloc[sub], y_all[sub]
    else:
        woe_cv, y_cv = woe, y_all

    grid_rows = []
    best_auc, best_C = -np.inf, float(C_grid[0])
    for C in C_grid:
        clf = LogisticRegression(penalty=penalty, C=float(C), solver=solver, max_iter=5000, random_state=seed)
        scores = cross_val_score(clf, woe_cv.to_numpy(), y_cv, cv=cv, scoring="roc_auc")
        mean_auc = float(np.mean(scores))
        grid_rows.append({"penalty": penalty, "C": float(C), "cv_auc": mean_auc, "cv_auc_std": float(np.std(scores))})
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, float(C)

    clf = LogisticRegression(penalty=penalty, C=best_C, solver=solver, max_iter=5000, random_state=seed)
    clf.fit(woe.to_numpy(), y_all)
    columns = ["const", *woe.columns.tolist()]
    wrapper = _RegularisedEstimator(clf, woe.columns.tolist())

    fitted = FittedModel(
        name=f"logit_{penalty}",
        kind="woe",
        estimator=wrapper,
        features=columns,
        encoders=encoders,
        cv_auc=best_auc,
        params={"penalty": penalty, "C": best_C},
    )
    coef = pd.DataFrame(
        {
            "variable": ["const", *(woe.columns.tolist())],
            "coefficient": [float(clf.intercept_[0]), *[float(c) for c in clf.coef_[0]]],
        }
    )
    coef["odds_ratio"] = np.exp(coef["coefficient"])
    coef["link"] = f"logit-{penalty}"
    return fitted, coef, pd.DataFrame(grid_rows)


class _RegularisedEstimator:
    """Adapt a fitted sklearn LogisticRegression to the statsmodels-style API."""

    def __init__(self, clf: LogisticRegression, woe_columns: List[str]):
        self.clf = clf
        self.woe_columns = woe_columns

    def predict(self, design: pd.DataFrame) -> np.ndarray:
        X = design.reindex(columns=self.woe_columns, fill_value=0.0)
        return self.clf.predict_proba(X.to_numpy(dtype=float))[:, 1]


# ---------------------------------------------------------------------------
# Tree / boosting challengers on raw features
# ---------------------------------------------------------------------------
def make_tree_pipeline(model_name: str, num: Sequence[str], cat: Sequence[str], seed: int = 42) -> Pipeline:
    """Preprocessing + estimator pipeline for the tree-based challengers."""
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), list(num)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                list(cat),
            ),
        ]
    )
    if model_name == "decision_tree":
        clf: Any = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=500, ccp_alpha=0.0005, random_state=seed
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
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
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("prep", pre), ("clf", clf)])


def cross_validate_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 42, n_splits: int = 5
) -> Tuple[float, float]:
    """Mean and std of the cross-validated ROC-AUC of a pipeline."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, np.asarray(y, dtype=int), cv=cv, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores))


def fit_tree_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num: Sequence[str],
    cat: Sequence[str],
    seed: int = 42,
    run_cv: bool = True,
) -> FittedModel:
    """Fit a tree/boosting challenger, optionally with cross-validated AUC."""
    pipeline = make_tree_pipeline(model_name, num, cat, seed)
    cv_auc = cv_std = None
    if run_cv:
        cv_auc, cv_std = cross_validate_pipeline(clone(pipeline), X_train, y_train, seed=seed)
    pipeline.fit(X_train, np.asarray(y_train, dtype=int))
    return FittedModel(
        name=model_name,
        kind="raw",
        estimator=pipeline,
        features=list(num) + list(cat),
        cv_auc=cv_auc,
        cv_auc_std=cv_std,
        params={"numeric": list(num), "categorical": list(cat)},
    )
