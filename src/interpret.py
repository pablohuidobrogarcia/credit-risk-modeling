"""Model interpretability: SHAP for the tree model vs logistic coefficients."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pd_model import FittedModel
from .utils import save_fig


def _transformed_feature_names(pipeline) -> List[str]:
    """Feature names produced by the preprocessing pipeline."""
    pre = pipeline.named_steps["prep"]
    names: List[str] = []
    for name, transformer, columns in pre.transformers_:
        if name == "num":
            names.extend(list(columns))
        elif name == "cat":
            try:
                names.extend(list(transformer.get_feature_names_out(columns)))
            except Exception:  # pragma: no cover
                names.extend([f"{c}_cat" for c in columns])
    return names


def shap_summary(
    model: FittedModel,
    X: pd.DataFrame,
    sample_size: int = 5000,
    seed: int = 42,
    name: str = "shap_summary.png",
) -> pd.DataFrame:
    """SHAP values for a tree/boosting model (TreeExplainer).

    Returns the mean absolute SHAP value per engineered feature, aggregated back
    from the one-hot columns to the original variable.
    """
    import shap

    if model.kind != "raw":
        raise ValueError("SHAP summary is implemented for the raw-feature tree models.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Xs = X.iloc[np.sort(idx)]

    pipeline = model.estimator
    pre = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    # The monotone-constrained LightGBM lives inside a thin wrapper (it needs to
    # resolve the constraint vector after the ColumnTransformer widens the
    # categoricals). TreeExplainer only understands the underlying booster, so
    # unwrap to it when present.
    clf = getattr(clf, "model_", clf)
    Xt = pre.transform(Xs)
    feature_names = _transformed_feature_names(pipeline)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):  # older shap / classifier output
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, features, classes)
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, max_display=20)
    save_fig(fig, name)

    imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    # Aggregate one-hot dummies back to their source variable.
    num = list(model.params.get("numeric", []))
    cat = list(model.params.get("categorical", []))
    imp["variable"] = [
        _source_variable(f, num, cat) for f in imp["feature"].astype(str)
    ]
    agg = imp.groupby("variable", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)
    return agg.reset_index(drop=True)


def _source_variable(feature_name: str, numeric: List[str], categorical: List[str]) -> str:
    """Map an engineered column name back to its originating variable."""
    if feature_name in numeric or feature_name in categorical:
        return feature_name
    for col in sorted(categorical, key=len, reverse=True):
        if feature_name.startswith(f"{col}_"):
            return col
    for col in sorted(numeric, key=len, reverse=True):
        if feature_name.startswith(f"{col}_"):
            return col
    return feature_name


def logistic_vs_shap(shap_importance: pd.DataFrame, coef_table: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side ranking of SHAP importance and |standardised logit coefficient|."""
    coef = coef_table.copy()
    coef = coef[coef["variable"] != "const"][["variable", "coefficient", "odds_ratio"]]
    coef["abs_coef"] = coef["coefficient"].abs()
    merged = shap_importance.merge(coef, on="variable", how="outer")
    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False)
    merged["coef_rank"] = merged["abs_coef"].rank(ascending=False)
    return merged.sort_values("shap_rank").reset_index(drop=True)
