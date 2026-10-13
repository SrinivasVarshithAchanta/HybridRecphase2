"""Explainable logistic reranker over hybrid candidate features.

Leakage policy:
- The reranker is trained ONLY on a temporal carve-out of the *train* data
  (all train rows except the last one as feature history; the last train row as
  the label). The evaluation test set is never touched at fit time.
- Features are rank-free statistics available at serving time.

Scoring policy (memory-safe):
- The reranker NEVER scores the full (users x items) catalog. It scores only a
  candidate pool: the union of top-`pool_size` items from the base hybrid and
  semantic rankings per user (~a few hundred candidates). Features are built
  per candidate row, so memory is O(users x pool_size x features).

Train/serve consistency (critical):
- ``build_reranker_training_set`` builds its candidate pool with the SAME
  ``build_candidate_pool`` function the serving path uses (top-`k` base hybrid
  union top-`k` semantic). The first generation of this reranker trained on a
  top-20 base-only pool while serving reranked a top-100 base-union-semantic
  pool; the resulting distribution shift made the learned weights actively
  harmful at serving time (HR@10 0.0025 vs 0.0285 for the un-reranked hybrid).
- Negatives are all non-positive pool members, which matches what the reranker
  actually sees at serving time. If the held-out positive item never enters the
  pool, the user's pool is kept as an ALL-NEGATIVE example (this is exactly the
  situation serving faces), and the user is counted in ``skipped``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = (
    "cf_score",
    "text_sim",
    "image_sim",
    "popularity",
    "rating_count",
    "avg_rating",
    "recency",
    "brand_match",
    "category_match",
)

DEFAULT_POOL_SIZE = 100


def build_candidate_pool(base_scores: np.ndarray, text_scores: np.ndarray, pool_size: int) -> np.ndarray:
    """Candidate pool for ONE user: top-`pool_size` base hybrid union top-`pool_size` semantic.

    Returns a 1-D array of unique item columns. This function is the single
    source of truth for both training-pool construction and serving-pool
    construction so the two distributions cannot drift apart.
    """
    n_items = base_scores.shape[-1]
    if pool_size >= n_items:
        return np.arange(n_items)
    base_pool = np.argpartition(-base_scores, pool_size)[:pool_size]
    sem_pool = np.argpartition(-text_scores, pool_size)[:pool_size]
    return np.unique(np.concatenate([base_pool, sem_pool]))


def _safe_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x)
    return (x - np.mean(x)) / std if std > 1e-12 else np.zeros_like(x)


def _safe_z_per_row(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mean = np.mean(mat, axis=1, keepdims=True)
    std = np.std(mat, axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return ((mat - mean) / std).astype(np.float32)


def build_item_features(item_ids, pop_counts, train_df, catalog) -> pd.DataFrame:
    """Per-catalog-item features from train interactions only."""
    counts = pd.Series(pop_counts, index=list(item_ids), dtype="float64")
    agg = train_df.groupby("asin")["overall"].agg(["count", "mean"])
    avg = agg["mean"].reindex(item_ids).fillna(3.0).to_numpy()
    rc = agg["count"].reindex(item_ids).fillna(0.0).to_numpy()
    brand = catalog.set_index("asin")["brand"].reindex(item_ids) if "brand" in catalog.columns else None
    category = catalog.set_index("asin")["categories"].reindex(item_ids) if "categories" in catalog.columns else None
    max_time = float(train_df["unixReviewTime"].max()) if len(train_df) else 0.0
    last_seen = train_df.groupby("asin")["unixReviewTime"].max().reindex(item_ids).fillna(0.0).to_numpy()
    recency = (last_seen / max_time) if max_time else np.zeros(len(item_ids))
    return pd.DataFrame(
        {
            "asin": list(item_ids),
            "popularity": _safe_z(np.log1p(counts.to_numpy())),
            "rating_count": _safe_z(np.log1p(rc)),
            "avg_rating": _safe_z(avg),
            "recency": _safe_z(recency),
            "brand": brand.to_numpy() if brand is not None else [""] * len(item_ids),
            "category": category.astype(str).to_numpy() if category is not None else [""] * len(item_ids),
        }
    )


def candidate_features(
    cf_row: np.ndarray,
    text_row: np.ndarray,
    image_row: np.ndarray | None,
    pool_cols: np.ndarray,
    item_feature_df: pd.DataFrame,
    history_asins: list[str],
    item_index: dict,
) -> np.ndarray:
    """Feature matrix (len(pool_cols), n_features) for ONE user's candidates."""
    n_feat = len(FEATURE_NAMES)
    out = np.zeros((len(pool_cols), n_feat), dtype=np.float32)
    pop_z = item_feature_df["popularity"].to_numpy()
    rc_z = item_feature_df["rating_count"].to_numpy()
    avg_z = item_feature_df["avg_rating"].to_numpy()
    rec_z = item_feature_df["recency"].to_numpy()
    brand_arr = item_feature_df["brand"].to_numpy()
    cat_arr = item_feature_df["category"].to_numpy()

    hist_idx = [item_index[a] for a in history_asins if a in item_index]
    if hist_idx:
        user_brand = {str(brand_arr[i]) for i in hist_idx}
        user_cat = {str(cat_arr[i]) for i in hist_idx}
    else:
        user_brand, user_cat = set(), set()

    # z-score the user's FULL catalog row, then read the candidate entries so
    # the feature keeps its "how good is this item for this user" meaning.
    cf_z = _safe_z_per_row(cf_row[None, :])[0]
    text_z = _safe_z_per_row(text_row[None, :])[0]
    img_z = (
        _safe_z_per_row(image_row[None, :])[0]
        if image_row is not None
        else np.zeros(len(cf_row), dtype=np.float32)
    )
    for j, col in enumerate(pool_cols):
        out[j, 0] = cf_z[col]
        out[j, 1] = text_z[col]
        out[j, 2] = img_z[col]
        out[j, 3] = pop_z[col]
        out[j, 4] = rc_z[col]
        out[j, 5] = avg_z[col]
        out[j, 6] = rec_z[col]
        out[j, 7] = 1.0 if str(brand_arr[col]) in user_brand else 0.0
        out[j, 8] = 1.0 if str(cat_arr[col]) in user_cat else 0.0
    return out


class HybridReranker:
    """Logistic reranker over tri-modal scores + popularity/brand/recency."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.model: LogisticRegression | None = None
        self.feature_names = list(FEATURE_NAMES)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HybridReranker":
        # Accept flat (n_samples, n_features) matrices (the only format the
        # candidate-pool builder produces).
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[2])
        X = np.nan_to_num(X)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self.model = None  # degenerate labels -> reranker disabled, honest
            return self
        if sample_weight is not None and len(sample_weight) != len(y):
            sample_weight = None  # mismatched weights would crash the fit
        model = LogisticRegression(C=self.C, max_iter=200, class_weight="balanced")
        model.fit(X, y, sample_weight=sample_weight)
        self.model = model
        return self

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def rank_pool(
        self,
        cf_row: np.ndarray,
        text_row: np.ndarray,
        image_row: np.ndarray | None,
        pool_cols: np.ndarray,
        item_feature_df: pd.DataFrame,
        history_asins: list[str],
        item_index: dict,
    ) -> np.ndarray:
        """Rerank one user's candidate pool; returns reranker scores per candidate."""
        feats = candidate_features(
            cf_row, text_row, image_row, pool_cols, item_feature_df, history_asins, item_index
        )
        return self.model.decision_function(np.nan_to_num(feats)).astype(np.float32)

    def rank_pool_batch(
        self,
        cf: np.ndarray,
        text: np.ndarray,
        image: np.ndarray | None,
        pool_list: list[np.ndarray],
        item_feature_df: pd.DataFrame,
        history_lists: list[list[str]],
        item_index: dict,
    ) -> list[np.ndarray]:
        """Multi-user vectorized twin of ``rank_pool`` (numerically identical).

        Builds one (sum|pool|, n_features) matrix for the whole batch and makes
        a single decision_function call. The per-user version cost ~29 min per
        full-dataset pass (sklearn + DataFrame overhead x 183k users); this one
        is ~5x faster. An equivalence test pins the two together.
        """
        pop_z = item_feature_df["popularity"].to_numpy()
        rc_z = item_feature_df["rating_count"].to_numpy()
        avg_z = item_feature_df["avg_rating"].to_numpy()
        rec_z = item_feature_df["recency"].to_numpy()
        brand_arr = item_feature_df["brand"].to_numpy()
        cat_arr = item_feature_df["category"].to_numpy()

        cf_z = _safe_z_per_row(cf)
        text_z = _safe_z_per_row(text)
        img_z = _safe_z_per_row(image) if image is not None else None

        n_feat = len(FEATURE_NAMES)
        sizes = [len(p) for p in pool_list]
        offsets = np.zeros(len(pool_list) + 1, dtype=int)
        offsets[1:] = np.cumsum(sizes)
        feats = np.zeros((int(offsets[-1]), n_feat), dtype=np.float32)
        for row, pool in enumerate(pool_list):
            sl = slice(int(offsets[row]), int(offsets[row + 1]))
            feats[sl, 0] = cf_z[row][pool]
            feats[sl, 1] = text_z[row][pool]
            if img_z is not None:
                feats[sl, 2] = img_z[row][pool]
            feats[sl, 3] = pop_z[pool]
            feats[sl, 4] = rc_z[pool]
            feats[sl, 5] = avg_z[pool]
            feats[sl, 6] = rec_z[pool]
            hist_idx = [item_index[a] for a in (history_lists[row] or []) if a in item_index]
            if hist_idx:
                user_brand = {str(brand_arr[i]) for i in hist_idx}
                user_cat = {str(cat_arr[i]) for i in hist_idx}
                feats[sl, 7] = [1.0 if str(brand_arr[c]) in user_brand else 0.0 for c in pool]
                feats[sl, 8] = [1.0 if str(cat_arr[c]) in user_cat else 0.0 for c in pool]
        scores = self.model.decision_function(np.nan_to_num(feats)).astype(np.float32)
        return [scores[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(pool_list))]

    def coefficients(self) -> dict:
        """Explainable per-feature weights (standardized features)."""
        if self.model is None:
            return {name: None for name in self.feature_names}
        return {
            name: float(w)
            for name, w in zip(self.feature_names, self.model.coef_[0])
        }


def build_reranker_training_set(
    cf_batch_fn,
    text_batch_fn,
    image_batch_fn,
    item_index,
    item_feature_df,
    history_asins: dict[str, list[str]],
    users: list[str],
    label_by_user: dict[str, str] | None = None,
    k: int = DEFAULT_POOL_SIZE,
    max_users: int = 20000,
    batch_size: int = 512,
    pool_builder_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | tuple[None, None, None, int]:
    """Training data from a temporal carve-out of train (no test leakage).

    For each user: feature history = all carve-out train rows except the last;
    the last carve-out row = positive label (supplied via ``label_by_user``).
    The candidate pool is built with ``pool_builder_fn`` (default: the same
    ``build_candidate_pool`` used at serving time) over the base-hybrid and
    semantic scores. Negatives are ALL other pool members, mirroring serving.
    If the positive never enters the pool, the pool is kept as an all-negative
    example and the user is counted in ``skipped``.
    Returns (X flat, y, per-sample weights, n_skipped_users).
    """
    if not label_by_user:
        return None, None, None, 0
    if pool_builder_fn is None:
        pool_builder_fn = build_candidate_pool
    users = users[:max_users]
    eligible = [u for u in users if len(history_asins.get(u, [])) >= 3]
    if not eligible:
        return None, None, None, 0

    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    skipped = 0

    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start : start + batch_size]
        cf = cf_batch_fn(chunk)
        text = text_batch_fn(chunk)
        image = image_batch_fn(chunk) if image_batch_fn else None
        if cf is None or text is None:
            continue
        cf_z_all = _safe_z_per_row(cf)
        text_z_all = _safe_z_per_row(text)
        for row, user in enumerate(chunk):
            pos = label_by_user.get(user)
            pos_col = item_index.get(pos) if pos else None
            hist = set(history_asins.get(user, ()))
            base = 0.5 * cf_z_all[row] + 0.5 * text_z_all[row]
            if hist:
                base[sorted(item_index[a] for a in hist if a in item_index)] = -np.inf
            pool = pool_builder_fn(base, text_z_all[row], k)
            labels_row = np.zeros(len(pool), dtype=int)
            if pos_col is None or pos_col not in set(pool.tolist()):
                # Positive never reached the candidate pool: keep the pool as an
                # all-negative example (this is exactly what serving sees for
                # such users) but flag the user so reporting stays honest.
                skipped += 1
                if pos_col is None:
                    continue  # label not even in the catalog: no usable signal
            else:
                labels_row[np.where(pool == pos_col)[0][0]] = 1
            feats = candidate_features(
                cf[row], text[row],
                image[row] if image is not None else None,
                pool, item_feature_df, history_asins.get(user, []), item_index,
            )
            feature_rows.append(feats)
            labels.append(labels_row)
            user_weight = min(len(history_asins.get(user, [])), 20.0) / 20.0 + 0.5
            weights.append(np.full(len(pool), user_weight))

    if not feature_rows:
        return None, None, None, skipped
    return (
        np.vstack(feature_rows),
        np.concatenate(labels),
        np.concatenate(weights),
        skipped,
    )


def build_history_asins(train_df: pd.DataFrame) -> dict[str, list[str]]:
    """user -> ordered list of train asins (oldest first)."""
    ordered = train_df.sort_values(["reviewerID", "unixReviewTime"], kind="mergesort")
    return {user: g["asin"].tolist() for user, g in ordered.groupby("reviewerID")}
