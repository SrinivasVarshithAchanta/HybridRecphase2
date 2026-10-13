"""Fast offline evaluation for top-K recommendation.

Design notes (why this finishes on the full dataset):

- Recommendations are produced in *batches*: all eval users' score rows are
  stacked once, seen items are masked with -inf, and top-k is selected with
  ``np.argpartition`` (O(rows * items)) instead of full argsort
  (O(rows * items * log items)). No Python loop touches every user x item.
- Relevant test items are precomputed once as a dict user -> set(asin).
- The Random baseline never materializes candidate lists: it draws k uniform
  indices per user from the unseen remainder (O(users * k)).
- Metrics are accumulated with numpy counters, not per-user set intersections
  in Python where a vectorized form exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_KEYS = ("precision", "recall", "accuracy", "ndcg")

# ---------------------------------------------------------------------------
# Panel-friendly metrics (percentages on 0-100)
#
# Full-catalog exact-item recommendation is a very strict task (a perfect
# model with ONE hidden item per user scores HitRate@10 <= 0.1), so the
# dashboard reports preference-level metrics instead:
#   Accuracy  - % of users whose top-k list shares a category with the hidden item.
#   Recall    - % of the hidden item's category paths covered by the top-k list.
#   Sampled Accuracy - hidden item in the top-10 of 100 candidates (1 + 99 negatives).
#   Like Prediction Accuracy - held-out accuracy of a like/not-like classifier
#                              (rating >= 4 => liked).
#   Lift - supporting metric: strict HitRate@10 vs the Random baseline.
# All values in the "panel" dicts are already scaled to 0-100.
# ---------------------------------------------------------------------------

PANEL_LABELS = ("Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift")
PANEL_DEFINITIONS = {
    "Accuracy": (
        "% of users whose top-10 recommendation list contains at least one product "
        "from the same category path as the hidden test product (e.g. both are "
        "Electronics > Computers & Accessories > Tablets)."
    ),
    "Recall": (
        "% of the hidden test product's category paths covered by the top-10 "
        "recommendation list."
    ),
    "Sampled Accuracy": (
        "% of users whose hidden product ranks in the top 10 among 100 candidates "
        "(1 hidden product + 99 randomly sampled unseen products)."
    ),
    "Like Prediction Accuracy": (
        "Held-out accuracy of a logistic classifier predicting liked (rating >= 4) "
        "from model scores and train-only user/item statistics; the majority-class "
        "baseline is reported next to it in like_prediction.baseline_accuracy."
    ),
    "Lift": (
        "Strict exact-item HitRate@10 of the model divided by the Random baseline "
        "(supporting metric, not a percentage)."
    ),
}
SAMPLED_NEGATIVES = 99
SAMPLED_CANDIDATES = SAMPLED_NEGATIVES + 1
LIKE_RATING_THRESHOLD = 4.0

# ---------------------------------------------------------------------------
# Core metric helpers (single-user, kept for tests and tiny cohorts)
# ---------------------------------------------------------------------------

def metrics_from_recs(recommended, relevant, k=10):
    if not relevant:
        return None
    hits = len(set(recommended) & set(relevant))
    precision = hits / k
    recall = hits / len(relevant)
    accuracy = 1.0 if hits > 0 else 0.0
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended)
        if item in relevant
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "ndcg": float(dcg / idcg if idcg > 0 else 0.0),
    }


def mean_metrics(rows):
    rows = [r for r in rows if r]
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "ndcg": 0.0, "n_users": 0}
    out = {key: float(np.mean([x[key] for x in rows])) for key in METRIC_KEYS}
    out["n_users"] = int(len(rows))
    return out


# ---------------------------------------------------------------------------
# Batched top-k recommendation
# ---------------------------------------------------------------------------

def topk_batch(score_matrix: np.ndarray, seen_mask: np.ndarray, k: int):
    """Top-k unseen item indices per row.

    score_matrix: (n_rows, n_items) float
    seen_mask:    (n_rows, n_items) bool, True where the item must be excluded
    Returns (indices int64 (n_rows, k), scores float32 (n_rows, k)); rows with
    fewer than k unseen items are padded with -1 at the tail.
    """
    n_rows, n_items = score_matrix.shape
    k = int(min(k, n_items))
    masked = np.where(seen_mask, -np.inf, score_matrix)
    if k < n_items:
        # argpartition: O(n_items) per row; the top-k set is exact, order inside
        # the set is refined on the tiny k-slice only.
        part = np.argpartition(masked, n_items - k, axis=1)[:, n_items - k:]
        part_scores = np.take_along_axis(masked, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
    else:
        idx = np.argsort(-masked, axis=1)[:, :k]
    top_scores = np.take_along_axis(masked, idx, axis=1).astype(np.float32)
    idx[top_scores <= -np.inf] = -1
    return idx.astype(np.int64), top_scores


def build_seen_mask(user_ids, item_index: dict, seen_by_user: dict) -> np.ndarray:
    """Boolean (n_users, n_items) mask; True = item already seen by user."""
    n_items = len(item_index)
    mask = np.zeros((len(user_ids), n_items), dtype=bool)
    for row, user in enumerate(user_ids):
        for asin in seen_by_user.get(user, ()):  # bounded by user history length
            col = item_index.get(asin)
            if col is not None:
                mask[row, col] = True
    return mask


def recs_from_batch(idx: np.ndarray, item_ids) -> list[list[str]]:
    """Convert a batched index matrix into per-user asin lists (-1 padded)."""
    return [
        [item_ids[int(col)] for col in row if col >= 0]
        for row in idx
    ]


# ---------------------------------------------------------------------------
# Batched metrics for one scoring variant
# ---------------------------------------------------------------------------

def evaluate_score_batch(
    eval_users: list[str],
    score_fn,
    item_ids: list[str],
    item_index: dict,
    seen_by_user: dict,
    relevant_by_user: dict,
    k: int = 10,
    batch_size: int = 1024,
    progress: bool = False,
    *,
    panel_ctx: dict | None = None,
    compute_strict: bool = True,
    panel_seed: int = 42,
) -> dict:
    """Evaluate any variant that can produce a (batch, n_items) score matrix.

    score_fn(user_slice) -> np.ndarray of shape (len(slice), n_items) or None.
    Returning None (e.g. unbuildable user profiles) scores that user as no hits,
    which is counted but contributes zero to every metric.

    Panel metrics (optional):
      panel_ctx      - from build_panel_context(); enables category/sample/like
                       metrics computed in this same scoring pass.
      compute_strict  - when False, only panel metrics are computed (used to
                       backfill panel numbers over already-cached strict runs).
    """
    if not compute_strict and panel_ctx is None:
        raise ValueError("compute_strict=False requires panel_ctx")
    n_items = len(item_ids)
    totals = {key: 0.0 for key in METRIC_KEYS}
    n_scored = 0
    mean_rel = []
    panel = PanelAccumulator(panel_ctx, k=k, seed=panel_seed) if panel_ctx else None

    for start in range(0, len(eval_users), batch_size):
        chunk = eval_users[start : start + batch_size]
        scores = score_fn(chunk)
        if scores is None:
            scores = np.full((len(chunk), n_items), -np.inf, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(chunk), n_items):
            raise ValueError(
                f"score_fn returned {scores.shape}, expected {(len(chunk), n_items)}"
            )
        seen_mask = build_seen_mask(chunk, item_index, seen_by_user)
        idx, _ = topk_batch(scores, seen_mask, k)
        rec_lists = recs_from_batch(idx, item_ids)

        if panel is not None:
            panel.add(chunk, rec_lists, scores)

        if not compute_strict:
            continue
        for user, recs in zip(chunk, rec_lists):
            relevant = relevant_by_user.get(user)
            if not relevant:
                continue
            mean_rel.append(len(relevant))
            n_scored += 1
            rel_set = relevant if isinstance(relevant, set) else set(relevant)
            hit_positions = [i for i, item in enumerate(recs) if item in rel_set]
            hits = len(hit_positions)
            totals["precision"] += hits / k
            totals["recall"] += hits / len(rel_set)
            if hits > 0:
                totals["accuracy"] += 1.0
                totals["ndcg"] += float(
                    sum(1.0 / np.log2(pos + 2) for pos in hit_positions)
                    / sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
                )

    out: dict = {}
    if compute_strict:
        n = max(n_scored, 1)
        out.update({key: float(value / n) for key, value in totals.items()})
        out["n_users"] = int(n_scored)
        out["mean_relevant_items"] = float(np.mean(mean_rel)) if mean_rel else 0.0
        if progress:
            print(f"  evaluated {n_scored} users", flush=True)
    if panel is not None:
        out.update(panel.finalize())
    return out


# ---------------------------------------------------------------------------
# Legacy per-user interface (kept for tests / tiny cohorts)
# ---------------------------------------------------------------------------

def evaluate_users(users, test_df, rec_fn, k=10):
    relevant_by_user = {
        user: set(g["asin"]) for user, g in test_df.groupby("reviewerID")
    }
    relevant_sizes = []
    rows = []
    for user in users:
        relevant = relevant_by_user.get(user, set())
        relevant_sizes.append(len(relevant))
        rows.append(metrics_from_recs(rec_fn(user, k), relevant, k=k))
    out = mean_metrics(rows)
    out["mean_relevant_items"] = float(np.mean(relevant_sizes)) if relevant_sizes else 0.0
    return out


# ---------------------------------------------------------------------------
# Non-personalized baselines
# ---------------------------------------------------------------------------

def random_recs_batch(eval_users, item_ids, seen_by_user, k, seed=42):
    """Random top-k that never builds candidate lists: O(users * k).

    For each user, sample k distinct indices from the full catalog, then drop
    seen ones. With tiny k relative to the catalog the collision probability
    with seen items is k*|seen|/n_items; sampling ~k + extra candidates and
    deduplicating keeps the expected hit-rate-vs-random within a fraction of a
    percent on this dataset (catalog >= ~1k items, k = 10). This is a documented
    approximation to "uniform over unseen items" used only for the baseline.
    """
    n_items = len(item_ids)
    draw = min(n_items, k + 12)  # extra draws replace seen collisions
    out = []
    for i, user in enumerate(eval_users):
        rng = np.random.default_rng(seed + i)
        seen = seen_by_user.get(user, set())
        picks: list[str] = []
        tries = 0
        while len(picks) < min(k, n_items - len(seen)) and tries < 4:
            idx = rng.choice(n_items, size=draw, replace=False)
            for j in idx:
                asin = item_ids[int(j)]
                if asin not in seen and asin not in picks:
                    picks.append(asin)
                    if len(picks) >= k:
                        break
            tries += 1
        out.append(picks)
    return out


def evaluate_random_batch(
    eval_users, item_ids, seen_by_user, relevant_by_user, k=10, seed=42,
    *, panel_ctx=None, compute_strict=True,
) -> dict:
    n_items = len(item_ids)
    rec_lists = random_recs_batch(eval_users, item_ids, seen_by_user, k, seed=seed)
    totals = {key: 0.0 for key in METRIC_KEYS}
    n_scored = 0
    mean_rel = []
    panel = PanelAccumulator(panel_ctx, k=k, seed=seed) if panel_ctx else None
    if panel is not None:
        # A random ranker has no score matrix: candidate sampling and the
        # like-prediction score feature are simulated from the same uniform
        # randomness its recommendations come from.
        panel.add(eval_users, rec_lists, scores=None)
    if not compute_strict:
        return dict(panel.finalize()) if panel is not None else {}
    for user, recs in zip(eval_users, rec_lists):
        relevant = relevant_by_user.get(user)
        if not relevant:
            continue
        mean_rel.append(len(relevant))
        n_scored += 1
        rel_set = relevant if isinstance(relevant, set) else set(relevant)
        hit_positions = [i for i, item in enumerate(recs) if item in rel_set]
        hits = len(hit_positions)
        totals["precision"] += hits / k
        totals["recall"] += hits / len(rel_set)
        if hits > 0:
            totals["accuracy"] += 1.0
            totals["ndcg"] += float(
                sum(1.0 / np.log2(pos + 2) for pos in hit_positions)
                / sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
            )
    n = max(n_scored, 1)
    out = {key: float(value / n) for key, value in totals.items()}
    out["n_users"] = int(n_scored)
    out["mean_relevant_items"] = float(np.mean(mean_rel)) if mean_rel else 0.0
    if panel is not None:
        out.update(panel.finalize())
    return out


def evaluate_popularity_batch(
    eval_users,
    item_ids,
    pop_counts: np.ndarray,
    seen_by_user: dict,
    relevant_by_user,
    k=10,
    batch_size: int = 4096,
    item_index: dict | None = None,
    *, panel_ctx=None,
) -> dict:
    """Popularity baseline via batched top-k (one shared score row per chunk)."""
    if item_index is None:
        item_index = {a: i for i, a in enumerate(item_ids)}
    scores_row = np.asarray(pop_counts, dtype=np.float32)
    totals = {key: 0.0 for key in METRIC_KEYS}
    n_scored = 0
    mean_rel = []
    panel = PanelAccumulator(panel_ctx, k=k) if panel_ctx else None
    for start in range(0, len(eval_users), batch_size):
        chunk = eval_users[start : start + batch_size]
        sub = np.tile(scores_row, (len(chunk), 1))
        seen_mask = build_seen_mask(chunk, item_index, seen_by_user)
        idx, _ = topk_batch(sub, seen_mask, k)
        rec_lists = recs_from_batch(idx, item_ids)
        if panel is not None:
            panel.add(chunk, rec_lists, sub)
        for user, recs in zip(chunk, rec_lists):
            relevant = relevant_by_user.get(user)
            if not relevant:
                continue
            mean_rel.append(len(relevant))
            n_scored += 1
            rel_set = relevant if isinstance(relevant, set) else set(relevant)
            hit_positions = [i for i, item in enumerate(recs) if item in rel_set]
            hits = len(hit_positions)
            totals["precision"] += hits / k
            totals["recall"] += hits / len(rel_set)
            if hits > 0:
                totals["accuracy"] += 1.0
                totals["ndcg"] += float(
                    sum(1.0 / np.log2(pos + 2) for pos in hit_positions)
                    / sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
                )
    n = max(n_scored, 1)
    out = {key: float(value / n) for key, value in totals.items()}
    out["n_users"] = int(n_scored)
    out["mean_relevant_items"] = float(np.mean(mean_rel)) if mean_rel else 0.0
    if panel is not None:
        out.update(panel.finalize())
    return out


def lift_over(reference: dict, metrics: dict) -> float | None:
    """Multiplicative lift of a variant's accuracy over a baseline."""
    base = reference.get("accuracy")
    acc = metrics.get("accuracy")
    if not base or acc is None:
        return None
    return float(acc / base)


# ---------------------------------------------------------------------------
# Legacy per-user recommenders (tests + validate_pipeline.py compatibility)
# ---------------------------------------------------------------------------

def random_recommender(item_ids, seen_by_user, seed_base=42):
    import hashlib

    def _fn(user, k=10):
        seen = seen_by_user.get(user, set())
        candidates = [a for a in item_ids if a not in seen]
        digest = hashlib.md5(str(user).encode("utf-8")).hexdigest()
        rng = np.random.RandomState(seed_base + int(digest[:8], 16) % 10_000)
        if not candidates:
            return []
        take = min(k, len(candidates))
        return list(rng.choice(candidates, size=take, replace=False))

    return _fn


def popularity_recommender(item_ids, counts, seen_by_user):
    scores = np.array([counts.get(a, 0.0) for a in item_ids], dtype=np.float64)

    def _fn(user, k=10):
        seen = seen_by_user.get(user, set())
        order = np.argsort(-scores)
        recs = []
        for idx in order:
            item = item_ids[int(idx)]
            if item in seen:
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs

    return _fn


def relevant_by_user_from(test_df: pd.DataFrame) -> dict:
    """Precompute test relevant items per user (done once per evaluation)."""
    return {user: set(g["asin"]) for user, g in test_df.groupby("reviewerID")}

# ---------------------------------------------------------------------------
# Panel-friendly metrics: implementation
# ---------------------------------------------------------------------------

def parse_category_paths(value) -> tuple[tuple[str, ...], ...]:
    """Normalize raw metadata ``categories`` into a tuple of category paths.

    Accepts both the nested list-of-lists form from the raw Amazon metadata
    (``[["Electronics", "Computers & Accessories", "Tablets"]]``) and the
    flattened space-joined strings produced by ``build_item_catalog``. Two
    products are "same category" when any of their returned path tuples are
    equal (e.g. both are exactly ``Electronics > Computers & Accessories >
    Tablets``).
    """
    if value is None:
        return ()
    if isinstance(value, float) and np.isnan(value):
        return ()
    if isinstance(value, str):
        text = value.strip()
        return ((text,),) if text else ()
    if not isinstance(value, (list, tuple, set)):
        text = str(value).strip()
        return ((text,),) if text else ()
    out: list[tuple[str, ...]] = []
    for path in value:
        if isinstance(path, str):
            segments = (path.strip(),) if path.strip() else ()
        elif isinstance(path, (list, tuple, set)):
            segments = tuple(str(s).strip() for s in path if str(s).strip())
        else:
            segments = ()
        if segments and segments not in out:
            out.append(segments)
    return tuple(out)


def build_category_index(item_catalog) -> dict[str, tuple[tuple[str, ...], ...]]:
    """asin -> category paths. Accepts the item catalog DataFrame or a mapping."""
    if isinstance(item_catalog, pd.DataFrame):
        if "categories" not in item_catalog.columns:
            return {}
        pairs = zip(item_catalog["asin"].astype(str), item_catalog["categories"])
    else:
        pairs = ((str(a), v) for a, v in item_catalog.items())
    return {asin: parse_category_paths(value) for asin, value in pairs}


def build_panel_context(
    eval_users,
    item_catalog,
    relevant_by_user,
    test_df,
    train_df,
    item_index,
    seen_by_user,
    pop_counts,
    history_counts=None,
    seed: int = 42,
) -> dict:
    """Precompute everything the panel metrics need (once per split).

    Leak policy: user/item aggregates come from TRAIN rows only; the only test
    signal used is the held-out label itself (the definition of the like task).
    The held-out row per user is their most recent test row, matching the
    temporal leave-one-last protocol (other splits fall back to the same rule).
    """
    item_paths = build_category_index(item_catalog)

    hidden_paths: dict[str, frozenset] = {}
    for user in eval_users:
        paths = set()
        for asin in relevant_by_user.get(user) or ():
            paths.update(item_paths.get(asin, ()))
        if paths:
            hidden_paths[user] = frozenset(paths)

    # Held-out row per user -> column + like/not-like label (rating >= 4).
    ordered = (
        test_df.sort_values("unixReviewTime", kind="mergesort")
        if "unixReviewTime" in test_df.columns
        else test_df
    )
    last_rows = ordered.groupby("reviewerID", sort=False).tail(1)
    eval_set = set(eval_users)
    true_col: dict[str, int] = {}
    label: dict[str, int] = {}
    for row in last_rows.itertuples(index=False):
        if row.reviewerID not in eval_set:
            continue
        col = item_index.get(row.asin)
        if col is None:
            continue
        true_col[row.reviewerID] = int(col)
        label[row.reviewerID] = 1 if float(row.overall) >= LIKE_RATING_THRESHOLD else 0

    # Train-only user aggregates: (mean rating, like rate, interaction count).
    ugrp = train_df.groupby("reviewerID")["overall"].agg(["mean", "count"])
    ulike = (
        (train_df["overall"] >= LIKE_RATING_THRESHOLD)
        .astype(float)
        .groupby(train_df["reviewerID"])
        .mean()
    )
    user_stats = {
        u: (float(ugrp.at[u, "mean"]), float(ulike.at[u]), int(ugrp.at[u, "count"]))
        for u in ugrp.index
    }

    # Train-only item aggregates, aligned to score columns.
    n_items = (max(item_index.values()) + 1) if item_index else 0
    item_mean = np.full(n_items, 4.0, dtype=np.float64)
    item_like = np.full(n_items, 0.8, dtype=np.float64)
    item_n = np.zeros(n_items, dtype=np.float64)
    igrp = train_df.groupby("asin")["overall"].agg(["mean", "count"])
    ilike = (
        (train_df["overall"] >= LIKE_RATING_THRESHOLD)
        .astype(float)
        .groupby(train_df["asin"])
        .mean()
    )
    for asin, row in igrp.iterrows():
        col = item_index.get(asin)
        if col is None:
            continue
        item_mean[col] = float(row["mean"])
        item_n[col] = float(row["count"])
        item_like[col] = float(ilike.at[asin])

    pop = np.asarray(pop_counts, dtype=np.float64)
    log_pop = np.zeros(n_items, dtype=np.float64)
    shared = min(n_items, len(pop))
    log_pop[:shared] = np.log1p(pop[:shared])

    # Seen columns per user (for sampled-candidate negatives); int32 keeps the
    # copy of the asin-set version small at 183k users.
    seen_cols: dict[str, np.ndarray] = {}
    for user in eval_users:
        cols = [item_index[a] for a in seen_by_user.get(user, ()) if a in item_index]
        seen_cols[user] = np.asarray(cols, dtype=np.int32)

    if history_counts is None:
        history_counts = {
            u: int(stats[2]) for u, stats in user_stats.items()
        }

    return {
        "item_paths": item_paths,
        "hidden_paths": hidden_paths,
        "true_col": true_col,
        "label": label,
        "user_stats": user_stats,
        "seen_cols": seen_cols,
        "item_mean": item_mean,
        "item_like": item_like,
        "item_n": item_n,
        "log_pop": log_pop,
        "history_counts": history_counts,
        "seed": int(seed),
        "n_items": n_items,
    }


_DEFAULT_USER_STATS = (4.0, 0.8, 0.0)


class PanelAccumulator:
    """Streams one evaluation pass into panel-friendly metrics (0-100).

    Like-prediction feature order (8 columns):
        [model score of the held-out item (row z-score),
         log popularity, user history length, user mean rating, user like rate,
         item mean rating, item like rate, log item rating count]
    """

    def __init__(self, ctx: dict, k: int = 10, seed: int = 42):
        if not ctx:
            raise ValueError("PanelAccumulator requires a panel context")
        self.ctx = ctx
        self.k = int(k)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.n_cat = 0
        self._cat_skipped = 0
        self.acc_sum = 0.0
        self.recall_sum = 0.0
        self.sampled_hits = 0
        self.sampled_n = 0
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []

    # -- per-batch accumulation ------------------------------------------------

    def add(self, users, rec_lists, scores=None) -> None:
        ctx = self.ctx
        item_paths = ctx["item_paths"]
        hidden = ctx["hidden_paths"]
        n = len(users)
        true_cols = np.fromiter(
            (ctx["true_col"].get(u, -1) for u in users), dtype=np.int64, count=n
        )

        # Category Accuracy / Recall over the produced top-k lists.
        for user, recs in zip(users, rec_lists):
            hp = hidden.get(user)
            if not hp:
                self._cat_skipped += 1
                continue
            rec_paths: set = set()
            for asin in recs:
                paths = item_paths.get(asin)
                if paths:
                    rec_paths.update(paths)
            matched = hp & rec_paths
            self.acc_sum += 1.0 if matched else 0.0
            self.recall_sum += len(matched) / len(hp)
            self.n_cat += 1

        if scores is None:
            # Uniform random ranker: the held-out item's rank among 100
            # candidates is uniform on 1..100, so simulate exactly that.
            ranks = self.rng.integers(1, SAMPLED_CANDIDATES + 1, size=n)
            sampled_hits = ranks <= self.k
            sampled_valid = np.ones(n, dtype=bool)
            score_feat = self.rng.random(n)
        else:
            # Work in the score matrix's own dtype (no full float64 copy) and
            # keep +/-inf: a pool-based scorer (reranker) marks "not scored"
            # items with -inf, which must rank the held-out item as a miss.
            s = np.asarray(scores)
            rows = np.arange(n)
            have = true_cols >= 0
            t_raw = s[rows, np.maximum(true_cols, 0)]
            mean = np.nan_to_num(s.mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(s.std(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
            safe_std = np.where(std > 0, std, 1.0)
            with np.errstate(invalid="ignore", over="ignore"):
                feat = np.where(have & (std > 0), (t_raw - mean) / safe_std, 0.0)
            score_feat = np.where(np.isfinite(feat), feat, 0.0)
            t_clean = np.where(np.isnan(t_raw), 0.0, t_raw)  # NaN -> 0, inf kept
            sampled_hits, sampled_valid = self._sampled_hits(users, true_cols, s, t_clean)

        self.sampled_hits += int(np.count_nonzero(sampled_hits & sampled_valid))
        self.sampled_n += int(np.count_nonzero(sampled_valid))

        # Like-prediction training rows (only users with a held-out label).
        labels = ctx["label"]
        user_stats = ctx["user_stats"]
        keep = np.array([u in labels for u in users], dtype=bool) & (true_cols >= 0)
        if keep.any():
            idx = np.nonzero(keep)[0]
            cols = true_cols[idx]
            us = np.array(
                [user_stats.get(users[i], _DEFAULT_USER_STATS) for i in idx],
                dtype=np.float64,
            )
            feats = np.column_stack(
                [
                    score_feat[idx],
                    ctx["log_pop"][cols],
                    us[:, 2],  # history length
                    us[:, 0],  # user mean rating
                    us[:, 1],  # user like rate
                    ctx["item_mean"][cols],
                    ctx["item_like"][cols],
                    np.log1p(ctx["item_n"][cols]),
                ]
            )
            self._X.append(feats)
            self._y.append(
                np.fromiter(
                    (labels[users[i]] for i in idx), dtype=np.int8, count=len(idx)
                )
            )

    def _sampled_hits(self, users, true_cols: np.ndarray, scores: np.ndarray, t=None):
        """Rank the held-out item among 1 held-out + 99 sampled unseen negatives."""
        n = len(users)
        n_items = scores.shape[1]
        hits = np.zeros(n, dtype=bool)
        valid = np.zeros(n, dtype=bool)
        if n_items < SAMPLED_CANDIDATES:
            return hits, valid  # catalog too small for a 100-candidate draw
        neg = np.zeros((n, SAMPLED_NEGATIVES), dtype=np.int64)
        for i, user in enumerate(users):
            if true_cols[i] < 0:
                continue
            col = int(true_cols[i])
            seen = self.ctx["seen_cols"].get(user)
            pool = None
            take = SAMPLED_NEGATIVES + 32
            while True:
                take = min(take, n_items)
                draw = self.rng.choice(n_items, size=take, replace=False)
                ok = draw != col
                if seen is not None and seen.size:
                    ok &= ~np.isin(draw, seen)
                cand = draw[ok]
                if pool is not None:
                    cand = np.concatenate([pool, cand])
                if len(cand) >= SAMPLED_NEGATIVES:
                    neg[i] = cand[:SAMPLED_NEGATIVES]
                    valid[i] = True
                    break
                pool = cand
                if take >= n_items:
                    break  # not enough unseen items; user excluded from denominator
        rows = np.arange(n)
        if t is None:
            raw = scores[rows, np.maximum(true_cols, 0)]
            t = np.where(np.isnan(raw), 0.0, raw)
        ns = scores[rows[:, None], neg]
        ns = np.where(np.isnan(ns), 0.0, ns)
        gt = (ns > t[:, None]).sum(axis=1)
        eq = (ns == t[:, None]).sum(axis=1)
        rank = 1.0 + gt + 0.5 * eq  # midrank: exact score ties share a position
        hits = rank <= self.k
        # Diagnostics for tests/audits: the last batch's sampled negatives.
        self.last_negatives = neg
        return hits, valid

    # -- finalization ----------------------------------------------------------

    def finalize(self) -> dict:
        n = max(self.n_cat, 1)
        accuracy = 100.0 * self.acc_sum / n
        recall = 100.0 * self.recall_sum / n
        sampled = 100.0 * self.sampled_hits / self.sampled_n if self.sampled_n else 0.0
        like_pct, like_aux = self._like_prediction()
        panel = {
            "Accuracy": round(float(accuracy), 1),
            "Recall": round(float(recall), 1),
            "Sampled Accuracy": round(float(sampled), 1),
            "Like Prediction Accuracy": round(float(like_pct), 1),
        }
        aux = {
            "n_users": int(self.n_cat),
            "users_without_category": int(self._cat_skipped),
            "sampled": {
                "candidates": SAMPLED_CANDIDATES,
                "negatives": SAMPLED_NEGATIVES,
                "top_k": int(self.k),
                "n_users": int(self.sampled_n),
                "seed": int(self.seed),
            },
            "like_prediction": like_aux,
        }
        return {"panel": panel, "panel_aux": aux}

    def _like_prediction(self):
        """Leak-safe like/not-like classifier on a seeded 50/50 user split."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score

        if not self._y:
            return 0.0, {"n_users": 0, "note": "no labelled held-out rows"}
        X = np.vstack(self._X)
        y = np.concatenate(self._y).astype(int)
        rng = np.random.default_rng(self.seed + 1)
        perm = rng.permutation(len(y))
        h = max(len(y) // 2, 1)
        train, test = perm[:h], perm[h:]
        y_test = y[test] if len(test) else np.zeros(0, dtype=int)
        baseline = float(max(y_test.mean(), 1 - y_test.mean())) if len(y_test) else 0.0
        note = None
        if len(train) < 5 or len(np.unique(y[train])) < 2 or len(test) == 0:
            const = int(y[train].mean() >= 0.5) if len(train) else 1
            pred = np.full(len(test), const, dtype=int)
            proba = pred.astype(float)
            note = "degenerate split; majority-class predictor reported"
        else:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[train], y[train])
            proba = clf.predict_proba(X[test])[:, 1]
            pred = (proba >= 0.5).astype(int)
        acc = float((pred == y_test).mean()) if len(y_test) else 0.0
        auc = None
        f1 = None
        if len(y_test) and len(np.unique(y_test)) > 1:
            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:  # pragma: no cover - defensive
                auc = None
        if len(y_test):
            try:
                f1 = float(f1_score(y_test, pred, zero_division=0))
            except ValueError:  # pragma: no cover - defensive
                f1 = None
        aux = {
            "n_users": int(len(y)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "label": f"rating >= {LIKE_RATING_THRESHOLD:g} means liked",
            "accuracy": round(acc * 100, 1),
            # Majority-class ("always liked") predictor on the same test half:
            # quoted so the accuracy is never read without its baseline.
            "baseline_accuracy": round(baseline * 100, 1),
            "roc_auc": round(auc, 3) if auc is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "positive_rate": round(float(y.mean()), 3),
            "note": note,
        }
        return acc * 100.0, aux
