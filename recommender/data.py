from __future__ import annotations

import ast
import json
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from .paths import CACHE, META_PATH, REVIEWS_PATH

SplitName = Literal["userwise_random", "temporal_leave_one_last"]
DEFAULT_REVIEW_COLUMNS = ["reviewerID", "asin", "overall", "reviewText", "unixReviewTime"]
DEFAULT_META_COLUMNS = [
    "asin",
    "title",
    "description",
    "imUrl",
    "categories",
    "brand",
    "price",
]


def load_reviews_file(filepath, max_lines=None):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Reviews")):
            data.append(json.loads(line))
            if max_lines is not None and i + 1 >= max_lines:
                break
    return pd.DataFrame(data)


def load_meta_file(filepath, max_lines=None):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Meta")):
            data.append(ast.literal_eval(line))
            if max_lines is not None and i + 1 >= max_lines:
                break
    return pd.DataFrame(data)


def _existing_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    return [c for c in requested if c in df.columns]


def load_dataset(max_lines=None, cache_name: str | None = None) -> pd.DataFrame:
    """Load the merged review + product metadata table.

    max_lines defaults to None so the production path uses the full dataset.
    A max_lines value is still useful for local smoke tests.
    """
    cache = CACHE / cache_name if cache_name else None
    if cache is not None and cache.exists():
        return pd.read_pickle(cache)
    reviews = load_reviews_file(REVIEWS_PATH, max_lines=max_lines)
    meta = load_meta_file(META_PATH, max_lines=max_lines)
    raw_rows = int(len(reviews))
    reviews = reviews[_existing_columns(reviews, DEFAULT_REVIEW_COLUMNS)].dropna(
        subset=["reviewerID", "asin", "overall"]
    )
    if "reviewText" not in reviews.columns:
        reviews["reviewText"] = ""
    if "unixReviewTime" not in reviews.columns:
        reviews["unixReviewTime"] = 0
    meta = meta[_existing_columns(meta, DEFAULT_META_COLUMNS)]
    data = pd.merge(reviews, meta, on="asin", how="left").dropna(
        subset=["title"]
    )
    data.attrs["raw_rows"] = raw_rows
    if cache is not None:
        data.to_pickle(cache)
    return data


def load_subset(n=None, max_lines=None, random_state=42) -> pd.DataFrame:
    """Backward-compatible loader.

    The previous project default sampled 5,000 rows from 50,000 source lines.
    The new default is full-data loading; pass n/max_lines explicitly for a
    bounded development run.
    """
    cache_name = None
    if n is not None or max_lines is not None:
        cache_name = f"subset_n{n or 'all'}_lines{max_lines or 'all'}_seed{random_state}_with_time.pkl"
    data = load_dataset(max_lines=max_lines, cache_name=cache_name)
    if n is not None and len(data) > n:
        return data.sample(n=n, random_state=random_state).copy()
    return data.copy()


def apply_k_core_filter(
    df: pd.DataFrame,
    min_user_count: int = 5,
    min_item_count: int = 5,
) -> pd.DataFrame:
    """Iteratively retain users/items meeting the requested interaction counts."""
    filtered = df.copy()
    while True:
        before = len(filtered)
        if min_user_count > 1:
            user_counts = filtered["reviewerID"].value_counts()
            filtered = filtered[
                filtered["reviewerID"].isin(user_counts[user_counts >= min_user_count].index)
            ]
        if min_item_count > 1:
            item_counts = filtered["asin"].value_counts()
            filtered = filtered[
                filtered["asin"].isin(item_counts[item_counts >= min_item_count].index)
            ]
        if len(filtered) == before:
            break
    return filtered.reset_index(drop=True)


def build_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "title": ("title", "first"),
        "description": ("description", "first"),
        "imUrl": ("imUrl", "first"),
    }
    for col in ("categories", "brand", "price"):
        if col in df.columns:
            aggregations[col] = (col, "first")
    catalog = df.groupby("asin").agg(**aggregations).reset_index()
    for col in ("description", "categories", "brand", "price"):
        if col in catalog.columns:
            catalog[col] = catalog[col].map(_metadata_to_text)
    return catalog


def _metadata_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_metadata_to_text(v) for v in value)
    return str(value)


def split_userwise_random(df: pd.DataFrame, test_ratio=0.2, seed=42):
    """Per-user shuffle split. Does not preserve temporal causality.

    Vectorized: users with >=2 interactions are split by position within a
    stable-sorted (user, shuffled-order) frame instead of a per-user loop.
    """
    rng = np.random.default_rng(seed)
    eligible_counts = df.groupby("reviewerID").size()
    eligible = eligible_counts[eligible_counts >= 2].index
    work = df[df["reviewerID"].isin(eligible)].copy()
    work["_order"] = rng.permutation(len(work))
    work = work.sort_values(["reviewerID", "_order"], kind="mergesort")
    counts = work.groupby("reviewerID")["reviewerID"].transform("size")
    ranks = work.groupby("reviewerID").cumcount()
    split_idx = np.floor(counts * (1 - test_ratio)).astype(int)
    is_train = ranks < split_idx
    train_df = work.loc[is_train].drop(columns="_order").reset_index(drop=True)
    test_df = work.loc[~is_train].drop(columns="_order").reset_index(drop=True)
    return train_df, test_df


def split_temporal_leave_one_last(df: pd.DataFrame):
    """Latest interaction per eligible user is test; earlier rows are train.

    Vectorized: rank rows per user by time (stable, ties broken by original
    order) and hold out the last row of every user with >=2 interactions.
    """
    eligible_counts = df.groupby("reviewerID").size()
    eligible = eligible_counts[eligible_counts >= 2].index
    work = df[df["reviewerID"].isin(eligible)].copy()
    work = work.sort_values(["reviewerID", "unixReviewTime"], kind="mergesort")
    counts = work.groupby("reviewerID")["reviewerID"].transform("size")
    ranks = work.groupby("reviewerID").cumcount()
    is_train = ranks < (counts - 1)
    train_df = work.loc[is_train].drop(columns=[c for c in ["_order"] if c in work.columns]).reset_index(drop=True)
    test_df = work.loc[~is_train].reset_index(drop=True)
    return train_df, test_df


def dataset_summary(
    raw_rows: int,
    after_merge: pd.DataFrame,
    after_kcore: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_items: set,
    train_items: set,
    min_user_count: int,
    min_item_count: int,
    max_lines: int | None,
) -> dict:
    """Full provenance report: raw rows -> merge -> k-core -> split -> coverage."""
    merged_rows = int(len(after_merge))
    kcore_rows = int(len(after_kcore))
    test_total = int(len(test_df))
    warm_rows = int(test_df["asin"].isin(train_items).sum()) if test_total else 0
    reachable = int(test_df["asin"].isin(catalog_items).sum()) if test_total else 0
    return {
        "raw_rows": raw_rows,
        "rows_after_merge": merged_rows,
        "rows_dropped_no_title": max(raw_rows - merged_rows, 0),
        "rows_after_kcore": kcore_rows,
        "users_raw": int(after_merge["reviewerID"].nunique()) if merged_rows else 0,
        "items_raw": int(after_merge["asin"].nunique()) if merged_rows else 0,
        "users_after_kcore": int(after_kcore["reviewerID"].nunique()) if kcore_rows else 0,
        "items_after_kcore": int(after_kcore["asin"].nunique()) if kcore_rows else 0,
        "train_rows": int(len(train_df)),
        "test_rows": test_total,
        "train_users": int(train_df["reviewerID"].nunique()) if len(train_df) else 0,
        "train_items": int(len(train_items)),
        "candidate_catalog_size": int(len(catalog_items)),
        "reachable_test_rows": reachable,
        "reachable_test_fraction": float(reachable / test_total) if test_total else 0.0,
        "warm_test_rows": warm_rows,
        "cold_test_rows": test_total - warm_rows,
        "warm_test_fraction": float(warm_rows / test_total) if test_total else 0.0,
        "min_user_count": min_user_count,
        "min_item_count": min_item_count,
        "source_lines": max_lines,
        "source": "full dataset" if max_lines is None else f"first {max_lines} lines",
    }


def fit_encoders(train_df: pd.DataFrame):
    """Backward-compatible encoder helper (index-based, no LabelEncoder)."""
    user_encoder = {u: i for i, u in enumerate(sorted(train_df["reviewerID"].unique()))}
    item_encoder = {a: i for i, a in enumerate(sorted(train_df["asin"].unique()))}
    return train_df, user_encoder, item_encoder
