"""Score fusion: bi-modal (legacy) and tri-modal (CF / text / image)."""
from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def zscore_batch(mat: np.ndarray, axis: int = -1) -> np.ndarray:
    """Per-row z-score for (n_users, n_items) matrices; 1-D input is z-scored whole."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        return zscore(mat)
    mean = np.mean(mat, axis=axis, keepdims=True)
    std = np.std(mat, axis=axis, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (mat - mean) / std


def score_stats(arr) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "n": 0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def clamp_weight(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def tri_modal_scores(
    cf,
    text,
    image,
    alpha_cf: float = 0.2,
    beta_text: float = 0.6,
    gamma_image: float = 0.2,
    normalize: bool = True,
):
    """Weighted tri-modal fusion; z-scored per user when normalize=True.

    Weights are used as given (they need not sum to 1). When a modality is
    missing (None), its weight is redistributed across the remaining ones so
    fusion stays comparable.
    """
    present = [(s, w) for s, w in ((cf, alpha_cf), (text, beta_text), (image, gamma_image)) if s is not None]
    if not present:
        return None
    if len(present) < 3:
        total_w = sum(w for _, w in present) or 1.0
        present = [(s, w / total_w) for s, w in present]
    if normalize:
        weighted = [w * zscore_batch(np.asarray(s, dtype=np.float32)) for s, w in present]
    else:
        weighted = [w * np.asarray(s, dtype=np.float64) for s, w in present]
    out = weighted[0]
    for term in weighted[1:]:
        out = out + term
    return out


def hybrid_scores(cf, semantic, alpha, normalize=True):
    """Legacy bi-modal fusion (kept for alpha sweeps and backward compat)."""
    if cf is None and semantic is None:
        return None
    if cf is None:
        return np.asarray(semantic, dtype=np.float64)
    if semantic is None:
        return np.asarray(cf, dtype=np.float64)
    if normalize:
        return alpha * zscore(cf) + (1.0 - alpha) * zscore(semantic)
    return alpha * np.asarray(cf, dtype=np.float64) + (1.0 - alpha) * np.asarray(
        semantic, dtype=np.float64
    )


def tri_modal_dynamic_weights(
    history_count: int,
    cool_user_threshold: int,
    alpha_cf: float,
    beta_text: float,
    gamma_image: float,
) -> tuple[float, float, float]:
    """Dynamic fusion: cool users lean on content (text + image), warm users on CF.

    Blend factor rises linearly from 0 (fully content) at zero history to 1
    (fully requested weights) at cool_user_threshold.
    """
    blend = 1.0
    if cool_user_threshold > 0:
        blend = min(1.0, max(0.0, history_count / cool_user_threshold))
    content = (0.0, 0.7, 0.3)
    target = (alpha_cf, beta_text, gamma_image)
    return tuple(blend * t + (1.0 - blend) * c for t, c in zip(target, content))


def topk_from_scores(scores, item_ids, seen, k=10):
    """Legacy per-user top-k (kept for tests / serving code paths)."""
    order = np.argsort(-np.asarray(scores))
    recs = []
    rec_idx = []
    for idx in order:
        item = item_ids[int(idx)]
        if item in seen:
            continue
        recs.append(item)
        rec_idx.append(int(idx))
        if len(recs) == k:
            break
    return recs, rec_idx


def contribution_explanation(alpha, cf_z, sem_z, image_z=None):
    """Human-readable contribution breakdown for one recommended item."""
    cf_part = alpha * cf_z
    sem_part = (1.0 - alpha) * sem_z
    img_part = 0.0 if image_z is None else 0.2 * image_z  # illustrative gamma
    parts = {
        "collaborative": abs(cf_part),
        "semantic": abs(sem_part),
        "visual": abs(img_part),
    }
    parts = {k: v for k, v in parts.items() if v > 1e-9}
    if not parts:
        return "Ranking is driven by residual scores after seen-item filtering."
    top = max(parts, key=parts.get)
    n_signals = len(parts)
    if n_signals >= 2:
        others = [v for k, v in parts.items() if k != top]
        if max(others) < 0.15 * parts[top]:
            return f"{top.capitalize()} similarity contributed most strongly to this ranking."
        return f"This item received support from multiple signals (strongest: {top})."
    return f"{top.capitalize()} similarity contributed most strongly to this ranking."
