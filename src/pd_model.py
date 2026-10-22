"""Probability-of-default models: WoE logistic scorecard plus challengers.

Models implemented
------------------
* ``logit``                 : statsmodels Logit on the WoE matrix (the
                              interpretable scorecard; coefficients + odds
                              ratios).
* ``probit``                : statsmodels Probit (GLM link comparison).
* ``logit_l1`` / ``logit_l2``: regularized logistic regression (Lasso/Ridge)
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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import ClassifierTags

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


def fit_regularized_logit(
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
    wrapper = _RegularizedEstimator(clf, woe.columns.tolist())

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


class _RegularizedEstimator:
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
class MonotoneLGBMClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM wrapper that resolves monotone constraints at fit time.

    The estimator sits at the end of a :class:`~sklearn.pipeline.Pipeline`, so it
    receives the *already transformed* matrix. The number of columns is only
    known once the ``ColumnTransformer`` has fitted its one-hot encoder, hence
    the constraints cannot be baked into ``LGBMClassifier`` up front. This
    wrapper builds the correct-length constraint vector on ``fit`` (negatives/
    zeros for the one-hot columns, which are unconstrained).

    Business-sensible direction (+1 = higher value -> higher PD):
    higher income / longer credit history lower PD; higher DTI, utilization and
    delinquency-related counts raise it.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        numeric: Optional[Sequence[str]] = None,
        monotone: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ) -> None:
        self.params = params
        self.numeric = numeric
        self.monotone = monotone
        self.seed = seed
        self.model_: Optional[Any] = None
        self._widened: Optional[Dict[str, int]] = None

    def __sklearn_tags__(self):
        """Declare this estimator as a classifier (sklearn >= 1.6 tags API)."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "MonotoneLGBMClassifier":
        from lightgbm import LGBMClassifier

        n_features = int(X.shape[1])
        numeric = list(self.numeric or [])
        mono_map = dict(self.monotone or {})
        base = [int(mono_map.get(c, 0)) for c in numeric]
        if len(base) < n_features:
            base = base + [0] * (n_features - len(base))
        base = base[:n_features]
        params = dict(self.params or {})
        if any(base):
            params["monotone_constraints"] = base
            params["monotone_constraints_method"] = "advanced"
        params.setdefault("random_state", self.seed)
        self._widened = {c: int(mono_map.get(c, 0)) for c in numeric}
        self.model_ = LGBMClassifier(**params)
        self.model_.fit(X, y, **fit_params)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        assert self.model_ is not None, "fit() must be called before predict()"
        return self.model_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        assert self.model_ is not None, "fit() must be called before predict()"
        return self.model_.predict_proba(X)


def make_tree_pipeline(
    model_name: str,
    num: Sequence[str],
    cat: Sequence[str],
    seed: int = 42,
    lgbm_params: Optional[Dict[str, Any]] = None,
    monotone_constraints: Optional[Dict[str, int]] = None,
) -> Pipeline:
    """Preprocessing + estimator pipeline for the tree-based challengers.

    Parameters
    ----------
    lgbm_params:
        Optional LightGBM hyper-parameter overrides (used by the temporal tuner).
    monotone_constraints:
        Optional mapping ``feature -> +1/-1/0`` for business-sensible monotone
        constraints. Only the *numeric* features (in ``num`` order) receive a
        constraint; unknown names are ignored.
    """
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(num)),
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
        params: Dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 100,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": seed,
            "verbose": -1,
        }
        if lgbm_params:
            params.update(lgbm_params)
        # Monotone constraints are applied *after* the ColumnTransformer, so the
        # constraint vector has to match the transformed width (numerics first,
        # then the one-hot expansion of the categoricals - which is only known
        # once the encoder is fitted). ``MonotoneLGBMClassifier`` resolves that
        # length at fit time; passing a fixed-length list to LGBMClassifier
        # inside a Pipeline fails because sklearn forwards it before the
        # transform runs.
        clf = MonotoneLGBMClassifier(
            params=params,
            numeric=list(num),
            monotone={k: int(v) for k, v in (monotone_constraints or {}).items()},
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
    lgbm_params: Optional[Dict[str, Any]] = None,
    monotone_constraints: Optional[Dict[str, int]] = None,
) -> FittedModel:
    """Fit a tree/boosting challenger, optionally with cross-validated AUC."""
    pipeline = make_tree_pipeline(
        model_name, num, cat, seed, lgbm_params=lgbm_params, monotone_constraints=monotone_constraints
    )
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
        params={
            "numeric": list(num),
            "categorical": list(cat),
            "lgbm_params": dict(lgbm_params or {}),
            "monotone_constraints": dict(monotone_constraints or {}),
        },
    )


def tune_lightgbm_temporal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    years: pd.Series,
    num: Sequence[str],
    cat: Sequence[str],
    tune_years: Sequence[int],
    valid_years: Sequence[int],
    monotone_constraints: Optional[Dict[str, int]] = None,
    n_trials: int = 12,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune LightGBM by *temporal* validation inside the mature train window.

    A small random search over business-reasonable hyper-parameters is scored by
    AUC on the held-out *later* slice of the training window (e.g. fit on
    2010-2012, validate on 2013). The primary test window (2014-2015) is never
    touched, so the golden rule holds: no tuning on the test data.

    Returns
    -------
    (best_params, trial_table)
        ``best_params`` is the LightGBM override dict for the winning trial;
        ``trial_table`` records every trial's AUC for the report.
    """
    from lightgbm import LGBMClassifier

    tune_years = list(tune_years)
    valid_years = list(valid_years)
    m_fit = years.isin(tune_years).to_numpy()
    m_val = years.isin(valid_years).to_numpy()
    if m_fit.sum() == 0 or m_val.sum() == 0:  # pragma: no cover - defensive
        raise ValueError("Temporal tuning needs non-empty fit and validation slices")

    Xf, yf = X_train.loc[m_fit], np.asarray(y_train)[m_fit]
    Xv, yv = X_train.loc[m_val], np.asarray(y_train)[m_val]

    pre = ColumnTransformer(
        [
            # ``keep_empty_features`` matters here: several bureau variables only
            # exist from 2012 onward, so they are *entirely* missing inside the
            # 2010-2011 slices. Without it sklearn drops the column and warns;
            # with it the column is carried as a constant, which is the honest
            # treatment of a field that did not exist in those vintages.
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(num)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                list(cat),
            ),
        ]
    )
    Xf_t = pre.fit_transform(Xf)
    Xv_t = pre.transform(Xv)

    mono = None
    if monotone_constraints:
        base = [int(monotone_constraints.get(c, 0)) for c in num]
        if any(base):
            # ColumnTransformer output: numerics first, then one-hot categoricals.
            n_cat = Xf_t.shape[1] - len(list(num))
            mono = base + [0] * n_cat

    rng = np.random.default_rng(seed)
    grid = {
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [50, 100, 200, 400],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "n_estimators": [300, 500, 800, 1200],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 20.0],
        "max_depth": [-1, 5, 8],
    }
    rows: List[Dict[str, Any]] = []
    best_auc, best_params = -np.inf, {}
    for _t in range(int(n_trials)):
        params = {
            "num_leaves": int(rng.choice(grid["num_leaves"])),
            "min_child_samples": int(rng.choice(grid["min_child_samples"])),
            "learning_rate": float(rng.choice(grid["learning_rate"])),
            "n_estimators": int(rng.choice(grid["n_estimators"])),
            "colsample_bytree": float(rng.choice(grid["colsample_bytree"])),
            "subsample": float(rng.choice(grid["subsample"])),
            "subsample_freq": 1,
            "reg_lambda": float(rng.choice(grid["reg_lambda"])),
            "max_depth": int(rng.choice(grid["max_depth"])),
            "n_jobs": -1,
            "random_state": seed,
            "verbose": -1,
        }
        if mono is not None:
            params["monotone_constraints"] = mono
            params["monotone_constraints_method"] = "advanced"
        clf = LGBMClassifier(**params)
        clf.fit(Xf_t, yf)
        p = clf.predict_proba(Xv_t)[:, 1]
        auc_val = float(roc_auc_score(yv, p))
        rows.append({**{k: params[k] for k in grid}, "valid_auc": round(auc_val, 4)})
        if auc_val > best_auc:
            best_auc, best_params = auc_val, {k: params[k] for k in grid}
    table = pd.DataFrame(rows).sort_values("valid_auc", ascending=False).reset_index(drop=True)
    return best_params, table
