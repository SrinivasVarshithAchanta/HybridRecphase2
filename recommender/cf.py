from __future__ import annotations

import inspect
import warnings

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.cpu.als import AlternatingLeastSquares as CPUALS


ALS_FACTORS = 128
ALS_ITERATIONS = 30
ALS_REGULARIZATION = 0.05
ALS_RANDOM_STATE = 42

CONFIDENCE_MODES = ("weighted_confidence", "raw_ratings", "log_confidence", "centered_positive", "positive_ge4")


def implicit_api_info():
    doc = inspect.getdoc(CPUALS.fit) or ""
    return {
        "implicit_version": implicit.__version__,
        "fit_parameter": "user_items",
        "doc_says_rows_are_users": "rows of the matrix are the users" in doc,
    }


def confidence_values(ratings: np.ndarray, mode: str) -> np.ndarray:
    """Map raw 1-5 ratings to implicit-ALS confidences.

    weighted_confidence  1 + rating - 1  (keeps every rating; 1 -> 1 ... 5 -> 5)
    raw_ratings          rating unchanged (1..5)
    log_confidence       1 + log(rating) (compresses the 5-star tail)
    centered_positive    max(0, rating - 3) + 1 (0.5 above neutral and up)
    positive_ge4         legacy arm: ratings >= 4 only, value 1 + (r - 4)
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    if mode == "weighted_confidence":
        return (1.0 + np.maximum(ratings - 1.0, 0.0)).astype(np.float32)
    if mode == "raw_ratings":
        return ratings.astype(np.float32)
    if mode == "log_confidence":
        return (1.0 + np.log(np.maximum(ratings, 1.0))).astype(np.float32)
    if mode == "centered_positive":
        return (np.maximum(ratings - 3.0, 0.0) + 1.0).astype(np.float32)
    if mode == "positive_ge4":
        return (1.0 + np.maximum(ratings - 4.0, 0.0)).astype(np.float32)
    raise ValueError(mode)


def build_user_item_matrix(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    mode: str = "weighted_confidence",
):
    """Build the users x items CSR confidence matrix."""
    if mode == "positive_ge4":
        used = train_df.loc[train_df["overall"] >= 4.0]
    else:
        used = train_df
    values = confidence_values(used["overall"].to_numpy(dtype=np.float64), mode)
    matrix = sp.coo_matrix(
        (values, (used["user_idx"].to_numpy(), used["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    ).tocsr()
    return matrix, int(used.shape[0])


def fit_als_with(
    matrix: sp.csr_matrix,
    n_users: int,
    n_items: int,
    factors: int = ALS_FACTORS,
    iterations: int = ALS_ITERATIONS,
    regularization: float = ALS_REGULARIZATION,
    random_state: int = ALS_RANDOM_STATE,
    use_gpu: bool | None = None,
):
    """Fit implicit ALS with orientation assertions for implicit 0.5+."""
    if matrix.shape != (n_users, n_items):
        raise AssertionError(
            f"Expected users×items {(n_users, n_items)}, got {matrix.shape}"
        )
    api = implicit_api_info()
    if not api["doc_says_rows_are_users"]:
        raise RuntimeError(
            f"Unexpected implicit API {api}. Re-check orientation before fitting."
        )
    # ALS device honesty: implicit's CUDA extension is a separate package
    # (implicit[gpu]) that is NOT installed here, so ALS runs on CPU even when
    # torch sees the RTX GPU. Torch-based encoders (text, CNN) do use CUDA.
    if use_gpu is None:
        try:
            import implicit.gpu

            has_gpu_ext = bool(implicit.gpu.HAS_CUDA if hasattr(implicit.gpu, "HAS_CUDA") else False)
        except Exception:
            has_gpu_ext = False
        use_gpu = False  # keep CPU until implicit[gpu] is verifiably installed
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
        use_gpu=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # implicit 0.7.x: rows = users, columns = items. Do not transpose.
        model.fit(matrix, show_progress=False)
        warn_msgs = [str(w.message) for w in caught]
    uf = np.asarray(model.user_factors)
    itf = np.asarray(model.item_factors)
    if uf.shape[0] != n_users:
        raise AssertionError(
            f"user_factors rows {uf.shape[0]} != n_users {n_users}. "
            "Matrix orientation is wrong."
        )
    if itf.shape[0] != n_items:
        raise AssertionError(
            f"item_factors rows {itf.shape[0]} != n_items {n_items}."
        )
    if uf.shape[1] != itf.shape[1]:
        raise AssertionError("ALS factor dimensions do not match.")
    return {
        "model": model,
        "user_factors": uf.astype(np.float32),
        "item_factors": itf.astype(np.float32),
        "warnings": warn_msgs,
        "api": api,
        "shape_ok": True,
        "config": {
            "factors": factors,
            "iterations": iterations,
            "regularization": regularization,
            "random_state": random_state,
            "orientation": "users x items, no transpose",
        },
    }


def fit_als(matrix: sp.csr_matrix, n_users: int, n_items: int):
    """Backward-compatible default fit."""
    return fit_als_with(matrix, n_users, n_items)


def als_score_matrix(user_factors: np.ndarray, item_factors: np.ndarray) -> np.ndarray:
    """Dense (n_users, n_items) CF score matrix; float32 keeps memory sane.

    192k users x 63k items x 4 bytes is ~46 GB, so for the full dataset the
    orchestrator chunks users instead of materializing this whole matrix.
    """
    return np.asarray(user_factors, dtype=np.float32) @ np.asarray(item_factors, dtype=np.float32).T


def cf_scores(user_idx: int, user_factors: np.ndarray, item_factors: np.ndarray):
    if user_idx < 0 or user_idx >= user_factors.shape[0]:
        return None
    return user_factors[user_idx] @ item_factors.T


# --- ALS hyperparameter / confidence sweep ---------------------------------

ALS_SWEEP_FACTORS = (64, 128, 256)
ALS_SWEEP_REG = (0.01, 0.05, 0.1)
ALS_SWEEP_ITERS = (20, 30, 50)


def als_sweep_configs(factors=None, regularizations=None, iterations=None):
    """All (factors, reg, iterations) combos requested by the review panel."""
    factors = factors or ALS_SWEEP_FACTORS
    regularizations = regularizations or ALS_SWEEP_REG
    iterations = iterations or ALS_SWEEP_ITERS
    return [
        {"factors": f, "regularization": r, "iterations": it}
        for f in factors
        for r in regularizations
        for it in iterations
    ]
