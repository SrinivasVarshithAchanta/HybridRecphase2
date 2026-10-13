from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .paths import CACHE

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_EMBEDDING_MODELS = (
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
)

EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
_MODEL_CACHE: dict = {}


def set_embedding_model(name: str) -> str:
    """Select the sentence-transformers model used by all subsequent builds."""
    global EMBEDDING_MODEL_NAME
    if name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Unsupported embedding model {name}; choose from {SUPPORTED_EMBEDDING_MODELS}")
    EMBEDDING_MODEL_NAME = name
    return name


def _load_model(device: str):
    from sentence_transformers import SentenceTransformer

    name = EMBEDDING_MODEL_NAME
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name, device=device)
    return _MODEL_CACHE[name]


def embedding_dim(model_name: str | None = None) -> int:
    name = model_name or EMBEDDING_MODEL_NAME
    return {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-base-en-v1.5": 768,
        "intfloat/e5-base-v2": 768,
    }.get(name, 384)


def _model_hash_part(model_name: str) -> str:
    return hashlib.md5(model_name.encode("utf-8")).hexdigest()[:10]


def _encode_texts(texts, cache_name: str | None = None) -> np.ndarray:
    """Encode with SentenceTransformers; cache keyed on model name + item texts.

    The cache digest combines the model name (so switching models never reads
    stale vectors) and a hash over the exact item texts (so a changed catalog
    text invalidates the cache too).
    """
    import torch

    texts = list(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_digest = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    digest = f"{_model_hash_part(EMBEDDING_MODEL_NAME)}_{text_digest}"
    cache = CACHE / f"{(cache_name or 'embeddings').replace('.npy', '')}_{digest}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            return arr
    model = _load_model(device)
    arr = np.asarray(
        model.encode(
            texts,
            batch_size=64 if device == "cuda" else 32,
            show_progress_bar=True,
            normalize_embeddings=False,
        ),
        dtype=np.float32,
    )
    np.save(cache, arr)
    return arr


def _stringify_feature(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify_feature(v) for v in value)
    return str(value)


def product_text(row) -> str:
    """Title + description + categories + brand + price (review-panel spec)."""
    parts = [
        _stringify_feature(row.get("title", "")),
        _stringify_feature(row.get("description", "")),
        _stringify_feature(row.get("categories", "")),
        _stringify_feature(row.get("brand", "")),
    ]
    price = _stringify_feature(row.get("price", ""))
    if price:
        parts.append(f"price {price}")
    return " ".join(part for part in parts if part).strip()


def weighted_user_profiles(
    train_df: pd.DataFrame,
    item_vecs: dict[str, np.ndarray],
    item_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """User profile = rating-weighted, recency-decayed mean of item vectors."""
    dim = item_dim or next(iter(item_vecs.values())).shape[0] if item_vecs else 384
    user_vecs: dict[str, np.ndarray] = {}
    has_time = "unixReviewTime" in train_df.columns
    max_time = float(train_df["unixReviewTime"].max()) if has_time and len(train_df) else 0.0
    min_time = float(train_df["unixReviewTime"].min()) if has_time and len(train_df) else 0.0
    time_span = max(max_time - min_time, 1.0)
    for user, group in train_df.groupby("reviewerID"):
        vecs = []
        weights = []
        for rec in group.itertuples(index=False):
            vec = item_vecs.get(getattr(rec, "asin"))
            if vec is None:
                continue
            rating = float(getattr(rec, "overall", 3.0))
            rating_weight = max(rating, 0.5)
            if has_time:
                rec_time = float(getattr(rec, "unixReviewTime", min_time) or min_time)
                recency_weight = 0.5 + 0.5 * ((rec_time - min_time) / time_span)
            else:
                recency_weight = 1.0
            vecs.append(vec)
            weights.append(rating_weight * recency_weight)
        if vecs:
            user_vecs[user] = np.average(np.vstack(vecs), axis=0, weights=np.asarray(weights))
    return user_vecs


def build_semantic_v1(train_df: pd.DataFrame):
    """One vector per review row (title+description+reviewText), then mean by ASIN/user."""
    combined = (
        train_df["title"].astype(str)
        + " "
        + train_df["description"].astype(str)
        + " "
        + train_df["reviewText"].astype(str)
    )
    row_emb = _encode_texts(combined.tolist(), "pkg_v1_row_embeddings.npy")
    item_vecs = {}
    user_vecs = {}
    tmp = train_df.copy()
    tmp["_emb"] = list(row_emb)
    for asin, group in tmp.groupby("asin"):
        item_vecs[asin] = np.mean(np.vstack(group["_emb"].values), axis=0)
    for user, group in tmp.groupby("reviewerID"):
        user_vecs[user] = np.mean(np.vstack(group["_emb"].values), axis=0)
    return item_vecs, user_vecs


def build_semantic_v2(train_df: pd.DataFrame):
    """One title+description embedding per unique ASIN. User = mean of history items."""
    item_meta = (
        train_df.groupby("asin")
        .agg(title=("title", "first"), description=("description", "first"))
        .reset_index()
    )
    texts = item_meta["title"].astype(str) + " " + item_meta["description"].astype(str)
    item_emb = _encode_texts(texts.tolist(), "pkg_v2_item_embeddings.npy")
    item_vecs = {
        asin: item_emb[i] for i, asin in enumerate(item_meta["asin"].tolist())
    }
    user_vecs = {}
    for user, group in train_df.groupby("reviewerID"):
        vecs = [item_vecs[a] for a in group["asin"].tolist() if a in item_vecs]
        if vecs:
            user_vecs[user] = np.mean(np.vstack(vecs), axis=0)
    return item_vecs, user_vecs


def build_semantic_catalog(item_catalog: pd.DataFrame, train_df: pd.DataFrame):
    """Product embeddings from the *global* catalog; weighted user profiles from train."""
    catalog = item_catalog.drop_duplicates("asin").reset_index(drop=True)
    if catalog.empty:
        return {}, {}
    texts = [product_text(row) for _, row in catalog.iterrows()]
    item_emb = _encode_texts(texts, "pkg_catalog_item_embeddings.npy")
    item_vecs = {asin: item_emb[i] for i, asin in enumerate(catalog["asin"].tolist())}
    user_vecs = weighted_user_profiles(train_df, item_vecs, item_dim=int(item_emb.shape[1]))
    return item_vecs, user_vecs


def stack_item_matrix(item_ids, item_vecs, dim=384) -> np.ndarray:
    if item_vecs:
        dim = next(iter(item_vecs.values())).shape[0]
    mat = np.zeros((len(item_ids), dim), dtype=np.float32)
    for i, asin in enumerate(item_ids):
        vec = item_vecs.get(asin)
        if vec is not None:
            mat[i] = vec
    return mat


def stack_user_matrix(user_ids, user_vecs, dim=384) -> np.ndarray:
    if user_vecs:
        dim = next(iter(user_vecs.values())).shape[0]
    mat = np.zeros((len(user_ids), dim), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        vec = user_vecs.get(uid)
        if vec is not None:
            mat[i] = vec
    return mat


def cosine_scores(user_vec: np.ndarray, item_mat: np.ndarray) -> np.ndarray:
    user_vec = np.asarray(user_vec, dtype=np.float64)
    item_mat = np.asarray(item_mat, dtype=np.float64)
    un = np.linalg.norm(user_vec)
    inorms = np.linalg.norm(item_mat, axis=1)
    denom = (un * inorms) + 1e-12
    return (item_mat @ user_vec) / denom


def cosine_batch(user_mat: np.ndarray, item_mat: np.ndarray) -> np.ndarray:
    """(n_users, n_items) cosine similarity in one matmul (the eval hot path)."""
    user_mat = np.asarray(user_mat, dtype=np.float32)
    item_mat = np.asarray(item_mat, dtype=np.float32)
    u_norm = np.linalg.norm(user_mat, axis=1, keepdims=True)
    i_norm = np.linalg.norm(item_mat, axis=1, keepdims=True)
    u_unit = user_mat / np.maximum(u_norm, 1e-12)
    i_unit = item_mat / np.maximum(i_norm, 1e-12)
    return (u_unit @ i_unit.T).astype(np.float32)
