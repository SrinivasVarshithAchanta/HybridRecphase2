"""
Technical validation of proj_updated2.ipynb.

Does not modify the original notebook. Reuses the same data loading, 5,000-row
sample (random_state=42), and user-wise split so new numbers are comparable.

Original notebook metrics are stored as a frozen snapshot and also re-run
under an "original_replicated" protocol.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import warnings
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "validation_cache"
RESULTS_PATH = ROOT / "validation_results.json"
CACHE_DIR.mkdir(exist_ok=True)

ORIGINAL_NOTEBOOK_SNAPSHOT = {
    "source": "datasets/proj_updated2.ipynb cell 0 saved output",
    "evaluation_cohort": "test_df['reviewerID'].unique()[:100]",
    "top_k": 10,
    "metrics": {
        "CF": {"precision": 0.002, "recall": 0.02, "ndcg": 0.006487982101190622},
        "Semantic": {"precision": 0.006, "recall": 0.06, "ndcg": 0.03433531571036666},
        "Hybrid_alpha_0.5": {
            "precision": 0.006000000000000001,
            "recall": 0.06,
            "ndcg": 0.028090309376660176,
        },
        "Hybrid_alpha_0.2": {
            "precision": 0.005,
            "recall": 0.05,
            "ndcg": 0.030628895028068964,
        },
        "Hybrid_alpha_0.4": {
            "precision": 0.005,
            "recall": 0.05,
            "ndcg": 0.030773243839286438,
        },
        "Hybrid_alpha_0.6": {
            "precision": 0.005,
            "recall": 0.05,
            "ndcg": 0.024820456477107903,
        },
        "Hybrid_alpha_0.8": {
            "precision": 0.0030000000000000005,
            "recall": 0.03,
            "ndcg": 0.011164948724497102,
        },
    },
    "sparse_history": {
        "label_in_notebook": "Cold start users identified",
        "validated_label": "few-interaction users (train interactions <= 2)",
        "users_identified": 183,
        "CF": {
            "precision": 0.002127659574468085,
            "recall": 0.02127659574468085,
            "ndcg": 0.006902108618287895,
        },
        "Hybrid": {
            "precision": 0.006382978723404255,
            "recall": 0.06382978723404255,
            "ndcg": 0.029883307847510834,
        },
        "Semantic": {
            "precision": 0.006382978723404255,
            "recall": 0.06382978723404255,
            "ndcg": 0.03652693160677304,
        },
    },
    "status": "frozen snapshot — not overwritten by this script",
}


def load_reviews_file(filepath, max_lines=None):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Reviews")):
            data.append(json.loads(line))
            if max_lines and i >= max_lines:
                break
    return pd.DataFrame(data)


def load_meta_file(filepath, max_lines=None):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Meta")):
            data.append(ast.literal_eval(line))
            if max_lines and i >= max_lines:
                break
    return pd.DataFrame(data)


def train_test_split_userwise(df, test_ratio=0.2):
    train_list, test_list = [], []
    for _, group in df.groupby("reviewerID"):
        if len(group) < 2:
            continue
        group = group.sample(frac=1, random_state=42)
        split_idx = int(len(group) * (1 - test_ratio))
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
    return pd.concat(train_list), pd.concat(test_list)


def metrics_from_recs(recommended, relevant, k=10):
    if not relevant:
        return None
    hits = len(set(recommended) & relevant)
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
    return {
        "precision": float(np.mean([x["precision"] for x in rows])),
        "recall": float(np.mean([x["recall"] for x in rows])),
        "accuracy": float(np.mean([x["accuracy"] for x in rows])),
        "ndcg": float(np.mean([x["ndcg"] for x in rows])),
        "n_users": int(len(rows)),
    }


def score_stats(arr):
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


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def topk_from_scores(scores, item_ids, seen, k=10):
    order = np.argsort(-scores)
    recs = []
    for idx in order:
        item = item_ids[idx]
        if item in seen:
            continue
        recs.append(item)
        if len(recs) == k:
            break
    return recs


def inspect_implicit_api():
    import implicit
    from implicit.cpu.als import AlternatingLeastSquares as CPUALS

    fit_doc = inspect.getdoc(CPUALS.fit) or ""
    expects_user_items = "user_items" in inspect.signature(CPUALS.fit).parameters
    rows_users = "rows of the matrix are the users" in (fit_doc or "")
    return {
        "implicit_version": implicit.__version__,
        "fit_parameter_name": "user_items" if expects_user_items else "unknown",
        "doc_says_rows_are_users": rows_users,
        "fit_doc_excerpt": fit_doc[:500],
    }


def load_subset():
    cache = CACHE_DIR / "data_subset.pkl"
    if cache.exists():
        print("Loading cached 5,000-row subset")
        return pd.read_pickle(cache)
    reviews = load_reviews_file(ROOT / "Electronics_5.json", max_lines=50000)
    meta = load_meta_file(ROOT / "meta_Electronics.json", max_lines=50000)
    reviews = reviews[["reviewerID", "asin", "overall", "reviewText"]].dropna()
    meta = meta[["asin", "title", "description", "imUrl"]]
    data = pd.merge(reviews, meta, on="asin", how="left").dropna(
        subset=["reviewText", "title"]
    )
    data_subset = data.sample(n=5000, random_state=42).copy()
    data_subset.to_pickle(cache)
    return data_subset


def encode_texts(texts, cache_name):
    cache = CACHE_DIR / cache_name
    texts = list(texts)
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            print(f"Loading cached embeddings {cache_name} {arr.shape}")
            return arr
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(texts)} texts with MiniLM on {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    arr = np.asarray(
        model.encode(texts, batch_size=32, show_progress_bar=True),
        dtype=np.float32,
    )
    np.save(cache, arr)
    return arr


def build_v1_embeddings(train_df):
    combined = (
        train_df["title"].astype(str)
        + " "
        + train_df["description"].astype(str)
        + " "
        + train_df["reviewText"].astype(str)
    )
    row_emb = encode_texts(combined.tolist(), "v1_row_embeddings.npy")
    train_df = train_df.copy()
    train_df["embedding"] = list(row_emb)

    item_vecs = {}
    for asin, group in train_df.groupby("asin"):
        item_vecs[asin] = np.mean(np.vstack(group["embedding"].values), axis=0)
    user_vecs = {}
    for user, group in train_df.groupby("reviewerID"):
        user_vecs[user] = np.mean(np.vstack(group["embedding"].values), axis=0)
    return item_vecs, user_vecs


def build_v2_embeddings(train_df):
    item_meta = (
        train_df.groupby("asin")
        .agg(title=("title", "first"), description=("description", "first"))
        .reset_index()
    )
    item_meta["item_text"] = (
        item_meta["title"].astype(str) + " " + item_meta["description"].astype(str)
    )
    item_emb = encode_texts(item_meta["item_text"].tolist(), "v2_item_embeddings.npy")
    item_vecs = {
        asin: item_emb[i] for i, asin in enumerate(item_meta["asin"].tolist())
    }
    user_vecs = {}
    for user, group in train_df.groupby("reviewerID"):
        vecs = [item_vecs[a] for a in group["asin"].tolist() if a in item_vecs]
        if vecs:
            user_vecs[user] = np.mean(np.vstack(vecs), axis=0)
    return item_vecs, user_vecs


def stacked_item_matrix(item_ids, item_vecs, dim=384):
    mat = np.zeros((len(item_ids), dim), dtype=np.float32)
    for i, asin in enumerate(item_ids):
        if asin in item_vecs:
            mat[i] = item_vecs[asin]
    return mat


def cosine_user_to_items(user_vec, item_mat):
    user_vec = np.asarray(user_vec, dtype=np.float64)
    item_mat = np.asarray(item_mat, dtype=np.float64)
    un = np.linalg.norm(user_vec)
    inorms = np.linalg.norm(item_mat, axis=1)
    denom = (un * inorms) + 1e-12
    return (item_mat @ user_vec) / denom


def fit_als(matrix, label):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=64, iterations=15, regularization=0.1, random_state=42
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(matrix, show_progress=False)
        warn_msgs = [str(w.message) for w in caught]
    return model, warn_msgs, label


def als_alignment(model, n_users, n_items):
    uf = np.asarray(model.user_factors)
    itf = np.asarray(model.item_factors)
    return {
        "user_factors_shape": list(uf.shape),
        "item_factors_shape": list(itf.shape),
        "user_axis_matches_n_users": bool(uf.shape[0] == n_users),
        "item_axis_matches_n_items": bool(itf.shape[0] == n_items),
        "factor_dims_match": bool(uf.shape[1] == itf.shape[1]),
        "can_score_all_users": bool(uf.shape[0] >= n_users),
        "can_score_all_items": bool(itf.shape[0] >= n_items),
    }


def evaluate_variant(users, test_df, rec_fn, k=10):
    rows = []
    for user in users:
        relevant = set(test_df.loc[test_df["reviewerID"] == user, "asin"])
        recs = rec_fn(user, k)
        rows.append(metrics_from_recs(recs, relevant, k=k))
    return mean_metrics(rows)


def main():
    report = {
        "original_notebook_snapshot": ORIGINAL_NOTEBOOK_SNAPSHOT,
        "experiment_status": {},
        "als_orientation": {},
        "score_scales": {},
        "validated_results": {},
        "notes": [],
        "decisions": [],
    }

    data_subset = load_subset()
    train_df, test_df = train_test_split_userwise(data_subset)
    train_df = train_df.copy()
    test_df = test_df.copy()

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    train_df["user_idx"] = user_encoder.fit_transform(train_df["reviewerID"])
    train_df["item_idx"] = item_encoder.fit_transform(train_df["asin"])

    n_users = int(train_df["user_idx"].nunique())
    n_items = int(train_df["item_idx"].nunique())
    item_ids = list(item_encoder.classes_)
    user_ids = list(user_encoder.classes_)

    train_sparse = sp.coo_matrix(
        (train_df["overall"], (train_df["user_idx"], train_df["item_idx"])),
        shape=(n_users, n_items),
    ).tocsr()

    nnz = int(train_sparse.nnz)
    sparsity = 1.0 - (nnz / (n_users * n_items))
    users_to_test = list(test_df["reviewerID"].unique()[:100])
    seen_by_user = {
        u: set(g["asin"]) for u, g in train_df.groupby("reviewerID")
    }
    sparse_history_users = [
        u for u, g in train_df.groupby("reviewerID") if len(g) <= 2
    ]
    sparse_in_eval = [u for u in sparse_history_users if u in users_to_test]

    api = inspect_implicit_api()
    print("\n=== implicit API ===")
    print(json.dumps({k: api[k] for k in api if k != "fit_doc_excerpt"}, indent=2))
    print("train_sparse.shape", train_sparse.shape, "n_users", n_users, "n_items", n_items)

    print("\n=== ALS fit(train_sparse.T)  [notebook behavior] ===")
    model_T, warn_T, _ = fit_als(train_sparse.T, "transpose")
    align_T = als_alignment(model_T, n_users, n_items)
    print(align_T, "warnings", warn_T)

    print("\n=== ALS fit(train_sparse)  [0.7 user_items convention] ===")
    model_U, warn_U, _ = fit_als(train_sparse, "user_items")
    align_U = als_alignment(model_U, n_users, n_items)
    print(align_U, "warnings", warn_U)

    transpose_incorrect = bool(
        api["doc_says_rows_are_users"]
        and train_sparse.shape == (n_users, n_items)
        and not align_T["user_axis_matches_n_users"]
        and align_U["user_axis_matches_n_users"]
        and align_U["item_axis_matches_n_items"]
    )

    library_recommend = {"available": False}
    probe_user = users_to_test[0]
    probe_idx = int(user_encoder.transform([probe_user])[0])
    if align_U["user_axis_matches_n_users"]:
        try:
            rec_idx, rec_scores = model_U.recommend(
                probe_idx,
                train_sparse[probe_idx],
                N=10,
                filter_already_liked_items=True,
            )
            rec_asins = [item_ids[i] for i in rec_idx]
            manual = model_U.user_factors[probe_idx] @ np.asarray(model_U.item_factors).T
            manual_top = topk_from_scores(
                np.asarray(manual), item_ids, seen_by_user[probe_user], k=10
            )
            library_recommend = {
                "available": True,
                "probe_user": probe_user,
                "library_top10": rec_asins,
                "manual_dot_top10": manual_top,
                "overlap": int(len(set(rec_asins) & set(manual_top))),
                "library_scores_sample": [float(s) for s in rec_scores[:5]],
            }
        except Exception as exc:
            library_recommend = {"available": False, "error": str(exc)}

    rec_change = {"comparable": False}
    if (
        align_T["can_score_all_users"]
        and align_T["can_score_all_items"]
        and align_U["user_axis_matches_n_users"]
    ):
        scores_T = np.asarray(model_T.user_factors[probe_idx]) @ np.asarray(
            model_T.item_factors
        ).T
        scores_U = np.asarray(model_U.user_factors[probe_idx]) @ np.asarray(
            model_U.item_factors
        ).T
        if scores_T.shape == scores_U.shape:
            top_T = topk_from_scores(scores_T, item_ids, seen_by_user[probe_user], 10)
            top_U = topk_from_scores(scores_U, item_ids, seen_by_user[probe_user], 10)
            rec_change = {
                "comparable": True,
                "probe_user": probe_user,
                "top10_with_transpose": top_T,
                "top10_without_transpose": top_U,
                "overlap": int(len(set(top_T) & set(top_U))),
            }

    report["als_orientation"] = {
        "implicit_api": api,
        "train_sparse_shape": list(train_sparse.shape),
        "n_users": n_users,
        "n_items": n_items,
        "conventional_users_x_items": bool(train_sparse.shape == (n_users, n_items)),
        "previous_call": "model_cf.fit(train_sparse.T)",
        "fit_transpose": {
            "warnings": warn_T,
            "alignment": align_T,
        },
        "fit_user_items": {
            "warnings": warn_U,
            "alignment": align_U,
        },
        "transpose_considered_incorrect_for_installed_api": transpose_incorrect,
        "library_recommend_check": library_recommend,
        "recommendation_change_on_probe_user": rec_change,
        "reliability_statement": (
            "The previous CF evaluation is considered unreliable until matrix "
            "orientation and factor alignment are verified."
        ),
        "verification_conclusion": (
            "fit(train_sparse.T) does not produce user_factors aligned with n_users "
            "under implicit {ver}. fit(train_sparse) does. Previous CF numbers are "
            "unreliable, not automatically 'garbage'."
        ).format(ver=api["implicit_version"]),
    }

    print("\nBuilding semantic embeddings (v1 review-averaged, v2 item-text)...")
    item_v1, user_v1 = build_v1_embeddings(train_df)
    item_v2, user_v2 = build_v2_embeddings(train_df)
    item_mat_v1 = stacked_item_matrix(item_ids, item_v1)
    item_mat_v2 = stacked_item_matrix(item_ids, item_v2)

    def cf_scores_for(model, user_id):
        if user_id not in user_encoder.classes_:
            return None
        uidx = int(user_encoder.transform([user_id])[0])
        uf = np.asarray(model.user_factors)
        itf = np.asarray(model.item_factors)
        if uidx >= uf.shape[0]:
            return None
        scores = uf[uidx] @ itf.T
        if scores.shape[0] != n_items:
            return None
        return np.asarray(scores, dtype=np.float64)

    def semantic_scores_for(user_id, user_vecs, item_mat):
        if user_id not in user_vecs:
            return None
        return cosine_user_to_items(user_vecs[user_id], item_mat)

    scale_rows = []
    for user in users_to_test[:8]:
        uidx = int(user_encoder.transform([user])[0]) if user in user_encoder.classes_ else None
        cf_raw_stats = "unaligned"
        if uidx is not None:
            uf_t = np.asarray(model_T.user_factors)
            itf_t = np.asarray(model_T.item_factors)
            if uidx < uf_t.shape[0]:
                cf_raw_stats = score_stats(uf_t[uidx] @ itf_t.T)
        cf_fix = cf_scores_for(model_U, user)
        sem = semantic_scores_for(user, user_v1, item_mat_v1)
        scale_rows.append(
            {
                "user": user,
                "cf_transpose": cf_raw_stats,
                "cf_user_items": score_stats(cf_fix) if cf_fix is not None else "unaligned",
                "semantic_v1_cosine": score_stats(sem) if sem is not None else "missing",
            }
        )
    report["score_scales"] = {
        "method_chosen_after_inspection": "per-user z-score on the full train-item score vector",
        "why": (
            "ALS dots and cosine similarities have incompatible min/max/std. "
            "Per-user z-score makes alpha a weight on standardized scores. "
            "Until this is applied, alpha=0.5 is a fusion weight, not 50%/50% influence."
        ),
        "sample_users": scale_rows,
    }

    pos_mask = train_df["overall"] >= 4
    train_pos = train_df.loc[pos_mask]
    pos_values = np.ones(len(train_pos), dtype=np.float32)
    train_sparse_pos = sp.coo_matrix(
        (pos_values, (train_pos["user_idx"], train_pos["item_idx"])),
        shape=(n_users, n_items),
    ).tocsr()
    model_pos, warn_pos, _ = fit_als(train_sparse_pos, "positive_ge4")

    pop_all = train_df.groupby("asin").size().to_dict()
    pop_pos = train_pos.groupby("asin").size().to_dict()
    pop_scores_all = np.array([pop_all.get(a, 0) for a in item_ids], dtype=np.float64)
    pop_scores_pos = np.array([pop_pos.get(a, 0) for a in item_ids], dtype=np.float64)

    def rec_random(user, k=10, seed_base=42):
        seen = seen_by_user.get(user, set())
        candidates = [a for a in item_ids if a not in seen]
        seed = seed_base + int(hashlib.md5(str(user).encode("utf-8")).hexdigest()[:8], 16) % 10_000
        rng = np.random.RandomState(seed)
        if not candidates:
            return []
        take = min(k, len(candidates))
        chosen = rng.choice(candidates, size=take, replace=False)
        return list(chosen)

    def rec_cf_notebook_style(model):
        """Score using whatever factor axis the fitted model has.

        Matches the notebook loop: rank score indices, skip IDs that are
        not valid item indices. Used only to replicate previous behavior.
        """
        def _fn(user, k=10):
            if user not in user_encoder.classes_:
                return []
            uidx = int(user_encoder.transform([user])[0])
            uf = np.asarray(model.user_factors)
            itf = np.asarray(model.item_factors)
            if uidx >= uf.shape[0]:
                return []
            scores = np.asarray(uf[uidx] @ itf.T)
            recs = []
            seen = seen_by_user.get(user, set())
            for idx in np.argsort(-scores):
                if idx >= n_items:
                    continue
                item = item_ids[int(idx)]
                if item not in seen:
                    recs.append(item)
                if len(recs) == k:
                    break
            return recs
        return _fn

    def rec_pop(pop_scores):
        def _fn(user, k=10):
            return topk_from_scores(
                pop_scores, item_ids, seen_by_user.get(user, set()), k
            )
        return _fn

    def rec_cf(model):
        def _fn(user, k=10):
            scores = cf_scores_for(model, user)
            if scores is None:
                return []
            return topk_from_scores(
                scores, item_ids, seen_by_user.get(user, set()), k
            )
        return _fn

    def rec_sem(user_vecs, item_mat):
        def _fn(user, k=10):
            scores = semantic_scores_for(user, user_vecs, item_mat)
            if scores is None:
                return []
            return topk_from_scores(
                scores, item_ids, seen_by_user.get(user, set()), k
            )
        return _fn

    def rec_hybrid(model, user_vecs, item_mat, alpha=0.5, normalize=False):
        def _fn(user, k=10):
            cf = cf_scores_for(model, user)
            sem = semantic_scores_for(user, user_vecs, item_mat)
            if cf is None and sem is None:
                return []
            if cf is None:
                scores = sem
            elif sem is None:
                scores = cf
            elif normalize:
                scores = alpha * zscore(cf) + (1 - alpha) * zscore(sem)
            else:
                scores = alpha * cf + (1 - alpha) * sem
            return topk_from_scores(
                scores, item_ids, seen_by_user.get(user, set()), k
            )
        return _fn

    print("\n=== Evaluating variants on the same 100-user cohort ===")
    variants = {
        "Random": rec_random,
        "Popularity_all_interactions": rec_pop(pop_scores_all),
        "Popularity_positive_ge4": rec_pop(pop_scores_pos),
        "CF_original_transpose": rec_cf_notebook_style(model_T),
        "CF_orientation_corrected": rec_cf(model_U),
        "CF_positive_ge4": rec_cf(model_pos),
        "Semantic_v1_title_desc_review": rec_sem(user_v1, item_mat_v1),
        "Semantic_v2_title_desc_only": rec_sem(user_v2, item_mat_v2),
        "Hybrid_v1_raw_alpha_0.5": rec_hybrid(
            model_U, user_v1, item_mat_v1, 0.5, False
        ),
        "Hybrid_v1_zscore_alpha_0.5": rec_hybrid(
            model_U, user_v1, item_mat_v1, 0.5, True
        ),
        "Hybrid_positive_v1_zscore_alpha_0.5": rec_hybrid(
            model_pos, user_v1, item_mat_v1, 0.5, True
        ),
        "Hybrid_v2_zscore_alpha_0.5": rec_hybrid(
            model_U, user_v2, item_mat_v2, 0.5, True
        ),
    }

    validated = {}
    for name, fn in variants.items():
        validated[name] = evaluate_variant(users_to_test, test_df, fn)
        print(name, validated[name])

    alpha_raw = {}
    alpha_z = {}
    for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        alpha_raw[str(a)] = evaluate_variant(
            users_to_test,
            test_df,
            rec_hybrid(model_U, user_v1, item_mat_v1, a, False),
        )
        alpha_z[str(a)] = evaluate_variant(
            users_to_test,
            test_df,
            rec_hybrid(model_U, user_v1, item_mat_v1, a, True),
        )
        print(f"alpha raw {a}", alpha_raw[str(a)])
        print(f"alpha z   {a}", alpha_z[str(a)])

    sparse_metrics = {
        "definition": "train interactions <= 2; not true cold-start (zero history)",
        "n_sparse_history_in_train": len(sparse_history_users),
        "n_sparse_history_in_eval_cohort": len(sparse_in_eval),
        "Semantic_v1": evaluate_variant(
            sparse_in_eval, test_df, rec_sem(user_v1, item_mat_v1)
        ),
        "CF_orientation_corrected": evaluate_variant(
            sparse_in_eval, test_df, rec_cf(model_U)
        ),
        "Hybrid_v1_zscore_alpha_0.5": evaluate_variant(
            sparse_in_eval,
            test_df,
            rec_hybrid(model_U, user_v1, item_mat_v1, 0.5, True),
        ),
        "Popularity_all_interactions": evaluate_variant(
            sparse_in_eval, test_df, rec_pop(pop_scores_all)
        ),
        "Random": evaluate_variant(sparse_in_eval, test_df, rec_random),
    }

    def ndcg_of(name):
        return validated[name]["ndcg"]

    ranking = sorted(validated.items(), key=lambda kv: kv[1]["ndcg"], reverse=True)
    best_name, best_row = ranking[0]

    report["validated_results"] = {
        "cohort": {
            "users": users_to_test,
            "n_users": len(users_to_test),
            "same_rule_as_notebook": "test_df['reviewerID'].unique()[:100]",
        },
        "variants": validated,
        "alpha_raw_orientation_corrected_v1": alpha_raw,
        "alpha_zscore_orientation_corrected_v1": alpha_z,
        "sparse_history_users": sparse_metrics,
        "ndcg_ranking": [
            {"name": n, "ndcg": m["ndcg"], "precision": m["precision"], "recall": m["recall"], "accuracy": m.get("accuracy", 0.0)}
            for n, m in ranking
        ],
        "best_among_tested": best_name,
        "hybrid_beats_semantic": bool(
            ndcg_of("Hybrid_v1_zscore_alpha_0.5")
            > ndcg_of("Semantic_v1_title_desc_review")
        ),
        "semantic_beats_baselines": bool(
            ndcg_of("Semantic_v1_title_desc_review")
            > max(
                ndcg_of("Random"),
                ndcg_of("Popularity_all_interactions"),
                ndcg_of("Popularity_positive_ge4"),
            )
        ),
    }

    report["dataset"] = {
        "name": "Amazon Electronics 5-core",
        "experimental_sample_rows": 5000,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_users": n_users,
        "n_items": n_items,
        "nnz": nnz,
        "sparsity": float(sparsity),
        "rating_counts_train": {
            str(k): int(v) for k, v in train_df["overall"].value_counts().sort_index().items()
        },
        "positive_ge4_train_rows": int(pos_mask.sum()),
        "reviews_per_item_train": {
            "mean": float(train_df.groupby("asin").size().mean()),
            "median": float(train_df.groupby("asin").size().median()),
            "max": int(train_df.groupby("asin").size().max()),
        },
    }

    report["experiment_status"] = {
        "dataset": "Amazon Electronics 5-core",
        "current_experimental_sample": "5,000 interactions",
        "evaluation_cohort": "100 users (same selection rule as original notebook)",
        "model": "ALS + MiniLM late fusion",
        "text_encoder": "Pretrained MiniLM transformer language encoder (all-MiniLM-L6-v2)",
        "vision": "Planned — imUrl may be used for display, not for scoring",
        "generative_llm": "Not currently used",
        "baseline_comparison": "Run in this validation (Random, Popularity)",
        "als_orientation": (
            "Verified incorrect for implicit "
            + api["implicit_version"]
            + " when using .T; corrected fit uses users x items"
            if transpose_incorrect
            else "Needs interpretation — see als_orientation section"
        ),
        "score_normalization": "Raw (original) and per-user z-score (validated)",
        "phase": "Phase-2 prototype",
        "true_cold_start": "Not supported (unknown reviewerID cannot be scored)",
        "few_shot_sparse_history": "Supported (1-2 train interactions)",
    }

    report["decisions"] = [
        "Keep original notebook numbers frozen as Original Notebook Results.",
        "Treat fit(train_sparse.T) as incorrect for installed implicit 0.7.x only after shape checks.",
        "Do not describe previous CF numbers as garbage; mark them unreliable.",
        "Positive implicit interaction defined as overall >= 4; original 1-5 confidence kept as a comparison arm.",
        "Fusion wording: alpha is a fusion weight. After z-score, it weights standardized scores, still not a literal 50/50 influence guarantee.",
        "Semantic-v1 = title+description+reviewText averaged by item. Semantic-v2 = one title+description embedding per ASIN.",
        "User-wise split retained so comparisons stay valid. Limitation: unrealistic temporal protocol, not classic feature leakage.",
        "Few-interaction users are not true cold-start users.",
    ]
    report["notes"] = [
        "MiniLM is a pretrained sentence-transformer encoder, not a generative LLM and not CoLLM.",
        "If semantic remains best after corrections, that is an observation on this sparse subset.",
        "A useful recommender claim requires beating Random and Popularity, not only beating CF.",
        "Item representation quality in v1 may depend on the number of training reviews per product.",
    ]

    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote", RESULTS_PATH)
    print("Best among tested (NDCG@10):", best_name, best_row)


if __name__ == "__main__":
    main()
