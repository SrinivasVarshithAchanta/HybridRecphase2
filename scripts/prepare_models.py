"""Train, evaluate, and persist HybridRec artifacts.

Smoke test (fast, clearly labeled):

    python scripts/prepare_models.py --max-lines 50000

Full dataset training + sampled evaluation (primary deliverable):

    python scripts/prepare_models.py --full-eval --eval-users 20000

The full run is resumable: the merged dataset, filtered dataset, ALS sweep
results, and serving artifacts are written to disk as they complete, so an
interrupted run skips straight to the missing stage on restart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.cf import (
    als_score_matrix,
    als_sweep_configs,
    build_user_item_matrix,
    confidence_values,
    fit_als_with,
    implicit_api_info,
)
from recommender.data import (
    apply_k_core_filter,
    build_item_catalog,
    dataset_summary,
    load_dataset,
    split_temporal_leave_one_last,
    split_userwise_random,
)
from recommender.evaluation import (
    PANEL_DEFINITIONS,
    PANEL_LABELS,
    build_panel_context,
    build_seen_mask,
    evaluate_popularity_batch,
    evaluate_random_batch,
    evaluate_score_batch,
    lift_over,
    relevant_by_user_from,
)
from recommender.hybrid import hybrid_scores, score_stats, topk_from_scores, tri_modal_scores, zscore_batch
from recommender.paths import CACHE
from recommender.persist import _atomic_write_text, save_bundle, save_metrics
from recommender.semantic import (
    build_semantic_catalog,
    cosine_batch,
    set_embedding_model,
    stack_item_matrix,
    stack_user_matrix,
)
from recommender.semantic import DEFAULT_EMBEDDING_MODEL

DEFAULT_TEMPORAL_ALPHA = 0.2
DEFAULT_USERWISE_ALPHA = 0.0
COOL_USER_THRESHOLD = 5
DEFAULT_BETA_TEXT = 0.6
DEFAULT_GAMMA_IMAGE = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Train HybridRec on Amazon Electronics.")
    parser.add_argument(
        "--dataset-mode",
        choices=["raw", "synthetic"],
        default="raw",
        help="raw trains/evaluates the Amazon review data; synthetic runs the generated preference-pattern evaluation.",
    )
    parser.add_argument(
        "--split-type",
        choices=["temporal_leave_one_last", "userwise_random", "both"],
        default="temporal_leave_one_last",
    )
    parser.add_argument("--min-user-count", type=int, default=5)
    parser.add_argument("--min-item-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--interaction-mode",
        choices=["weighted_confidence", "raw_ratings", "log_confidence", "centered_positive", "positive_ge4"],
        default="weighted_confidence",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional development cap. Omit for the full dataset.",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Evaluate on ALL test users (slow). Default evaluates a reproducible sample.",
    )
    parser.add_argument(
        "--eval-users",
        type=int,
        default=10000,
        help="Number of eval users sampled (seeded) when --full-eval is not given.",
    )
    parser.add_argument("--eval-seed", type=int, default=42, help="Seed for the eval user sample.")
    parser.add_argument("--skip-random", action="store_true", help="Skip the Random baseline (it is O(users*k) but stochastic).")
    parser.add_argument("--batch-size", type=int, default=2048, help="Users scored per evaluation batch.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformers model for the text modality.")
    parser.add_argument("--als-sweep", action="store_true", default=True, help="Sweep ALS configs on a train subsample and pick the best by NDCG@10.")
    parser.add_argument(
        "--als-sweep-subset",
        type=int,
        default=150000,
        help="Max train rows used for the ALS config sweep (0 = sweep on full train).",
    )
    parser.add_argument("--als-sweep-users", type=int, default=2000, help="Eval users used inside the ALS sweep.")
    parser.add_argument("--no-als-sweep", dest="als_sweep", action="store_false", help="Skip the sweep; use the single configured interaction mode/default ALS config.")
    parser.add_argument("--use-vision", action="store_true", help="Enable the CNN image encoder (downloads imUrl images).")
    parser.add_argument("--vision-model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--image-limit", type=int, default=None, help="Cap on images downloaded (smoke tests).")
    parser.add_argument("--image-cache-dir", default=str(ROOT / "image_cache"))
    parser.add_argument("--image-timeout", type=float, default=8.0)
    parser.add_argument("--skip-reranker", action="store_true", help="Disable the logistic reranker stage.")
    parser.add_argument("--skip-alpha-sweep", action="store_true", help="Skip the alpha (CF vs semantic) sweep to save eval time.")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Write artifacts to a custom directory (keeps the production artifacts/ untouched, e.g. for smoke tests).",
    )
    parser.add_argument("--synthetic-users", type=int, default=10000, help="Synthetic users to generate when --dataset-mode synthetic.")
    parser.add_argument("--synthetic-min-train-interactions", type=int, default=20)
    parser.add_argument("--synthetic-max-train-interactions", type=int, default=40)
    parser.add_argument("--synthetic-min-test-interactions", type=int, default=8)
    parser.add_argument("--synthetic-max-test-interactions", type=int, default=12)
    parser.add_argument("--synthetic-min-interactions", type=int, default=None, help="Compatibility alias for --synthetic-min-train-interactions.")
    parser.add_argument("--synthetic-max-interactions", type=int, default=None, help="Compatibility alias for --synthetic-max-train-interactions.")
    parser.add_argument("--synthetic-noise-rate", type=float, default=0.05)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--synthetic-catalog-path", default=str(ROOT / "artifacts" / "products.parquet"))
    parser.add_argument("--no-resume", action="store_true", help="Recompute cached stages (dataset prep, sweep).")
    return parser.parse_args()


def clamp_alpha(alpha: float) -> float:
    return float(min(1.0, max(0.0, alpha)))


# ---------------------------------------------------------------------------
# Resumable stage helpers
# ---------------------------------------------------------------------------


def evaluate_variant_cached(eval_users, item_ids, item_index, seen_by_user,
                            relevant_by_user, args, cache_key: str, panel_ctx: dict | None = None):
    """evaluate_variant + JSON checkpoint so a killed run costs minutes, not hours.

    A mid-run power loss previously threw away ~4 h of completed evaluations.
    Per-variant metrics are written to validation_cache/eval_cache_<key>.json
    as they complete; a re-run with the same cohort/seed/batch loads them and
    skips straight to the missing variants. Returns (evaluate_variant, get, flush).

    Panel (0-100) metrics are checkpointed SEPARATELY in
    eval_cache_<key>_panel.json: a strict cache that predates the panel work is
    still valid, and only the panel scoring pass is ever repeated for it.
    """
    cache_path = CACHE / f"eval_cache_{cache_key}.json"
    panel_cache_path = CACHE / f"eval_cache_{cache_key}_panel.json"
    cache: dict = {}
    panel_cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if panel_cache_path.exists():
        try:
            panel_cache = json.loads(panel_cache_path.read_text(encoding="utf-8"))
        except Exception:
            panel_cache = {}

    def _flush():
        _atomic_write_text(cache_path, json.dumps(cache, indent=2))

    def _flush_panel():
        _atomic_write_text(panel_cache_path, json.dumps(panel_cache, indent=2))

    def evaluate_variant(variant_name, score_fn, log_start=True):
        t0 = time.time()
        strict = cache.get(variant_name)
        panel_entry = panel_cache.get(variant_name) if panel_ctx is not None else None
        need_strict = strict is None
        need_panel = panel_ctx is not None and panel_entry is None
        if not need_strict and not need_panel:
            if panel_entry:
                print(f"[eval] cache hit: {variant_name} (strict + panel)", flush=True)
                return {**strict, **panel_entry}
            print(f"[eval] cache hit: {variant_name}", flush=True)
            return strict
        if log_start:
            mode = " [panel-only]" if not need_strict else ""
            print(f"[eval] variant: {variant_name} (batch_size={args.batch_size}, users={len(eval_users)}){mode}", flush=True)
        result = evaluate_score_batch(
            eval_users, score_fn, item_ids, item_index, seen_by_user,
            relevant_by_user, k=10, batch_size=args.batch_size,
            panel_ctx=panel_ctx if need_panel else None,
            compute_strict=need_strict,
            panel_seed=getattr(args, "eval_seed", 42),
        )
        panel_part = {k: result.pop(k) for k in ("panel", "panel_aux") if k in result}
        strict_result = dict(result) if need_strict else None
        if strict_result is not None:
            if "ndcg" in strict_result:
                print(f"[eval] {variant_name}: ndcg={strict_result['ndcg']:.4f} hitrate={strict_result['accuracy']:.4f} ({time.time()-t0:.1f}s)", flush=True)
            cache[variant_name] = strict_result
            _flush()
        if need_panel and panel_part:
            panel_cache[variant_name] = panel_part
            _flush_panel()
        if panel_part.get("panel"):
            pretty = ", ".join(f"{k}={v}" for k, v in panel_part["panel"].items())
            print(f"[eval] {variant_name} panel: {pretty} ({time.time()-t0:.1f}s)", flush=True)
        merged = dict(strict_result if strict_result is not None else (strict or {}))
        merged.update(panel_part or (panel_entry or {}))
        return merged

    def get(name, fn):
        # Always go through evaluate_variant: it decides (from the strict AND
        # panel caches) whether a scoring pass is still needed for this variant.
        return evaluate_variant(name, fn)

    return evaluate_variant, get, _flush

def stage_path(name: str, args) -> Path:
    tag = "full" if args.max_lines is None else f"lines{args.max_lines}"
    return CACHE / f"stage_{tag}_{name}"


def load_or_build_dataset(args):
    """Stage 1: load + k-core; cached so runs resume instantly."""
    path = stage_path("filtered.pkl", args)
    if not args.no_resume and path.exists():
        print(f"[stage] dataset cache hit: {path}")
        data = pd.read_pickle(path)
        raw_rows = int(path.with_suffix(".raw").read_text()) if path.with_suffix(".raw").exists() else len(data)
        data.attrs["raw_rows"] = raw_rows
        return data
    print("[stage] loading + merging dataset (this is the slow part the first time)")
    data = load_dataset(max_lines=args.max_lines)
    raw_rows = int(data.attrs.get("raw_rows", len(data)))
    data = apply_k_core_filter(data, args.min_user_count, args.min_item_count)
    data.attrs["raw_rows"] = raw_rows
    path.parent.mkdir(exist_ok=True)
    data.to_pickle(path)
    path.with_suffix(".raw").write_text(str(raw_rows))
    print(f"[stage] dataset cached: {len(data)} rows")
    return data


def train_subsample(train_df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Users-anchored subsample for the ALS sweep (keeps per-user histories intact).

    Vectorized: one groupby instead of a per-user scan (a python-loop version
    here cost hours on the full 1.42M-row train set). Always keeps at least
    two users so the sweep has both users and items to learn from.
    """
    if max_rows <= 0 or len(train_df) <= max_rows:
        return train_df
    rng = np.random.default_rng(seed)
    counts = train_df.groupby("reviewerID", sort=False).size()
    shuffled_users = counts.index[rng.permutation(len(counts))]
    cum = counts[shuffled_users].cumsum()
    # Take the smallest user prefix whose cumulative size reaches the cap.
    n_keep = int(np.searchsorted(cum.to_numpy(), max_rows, side="left") + 1)
    n_keep = max(2, min(n_keep, len(shuffled_users)))
    keep_users = set(shuffled_users[:n_keep])
    return train_df.loc[train_df["reviewerID"].isin(keep_users)]


def sweep_als(train_df, n_users, n_items, args, eval_users, relevant_by_user, item_ids, seen_by_user):
    """Sweep ALS configs x confidence modes on a subsample; return best by NDCG@10."""
    path = stage_path("als_sweep.json", args)
    if not args.no_resume and path.exists():
        print(f"[stage] ALS sweep cache hit: {path}")
        return json.loads(path.read_text())

    sub = train_subsample(train_df, args.als_sweep_subset)
    sub = sub.copy()
    uenc = {u: i for i, u in enumerate(sorted(sub["reviewerID"].unique()))}
    ienc = {a: i for i, a in enumerate(sorted(sub["asin"].unique()))}
    sub["user_idx"] = sub["reviewerID"].map(uenc)
    sub["item_idx"] = sub["asin"].map(ienc)
    s_users, s_items = len(uenc), len(ienc)
    # Evaluate the sweep WITHIN the subsample: rank each user's last
    # subsample-row (temporal hold-out) against the other items that user
    # interacted with in the subset. Using the global test set here would be
    # broken at full scale: a sampled user's held-out item is almost never in
    # the sampled item subset, so every config would score ~0 (this actually
    # happened on the first full run).
    sub_sorted = sub.sort_values(["reviewerID", "unixReviewTime"], kind="mergesort")
    sub_ranks = sub_sorted.groupby("reviewerID").cumcount()
    sub_sizes = sub_sorted.groupby("reviewerID")["reviewerID"].transform("size")
    sub_test = sub_sorted.loc[sub_ranks == sub_sizes - 1]
    sub_train = sub_sorted.loc[sub_ranks < sub_sizes - 1]  # sweep-fit rows ONLY
    sub_train_hist = {
        u: set(g["asin"]) for u, g in sub_train.groupby("reviewerID")
    }
    relevant_by_user = {u: set(g["asin"]) for u, g in sub_test.groupby("reviewerID")}
    seen_by_user = sub_train_hist
    sweep_users = [u for u in relevant_by_user if u in uenc][: args.als_sweep_users]
    print(
        f"[stage] ALS sweep on {len(sub)} rows, {s_users} users x {s_items} items, "
        f"{len(sweep_users)} eval users (self-contained temporal hold-out)",
        flush=True,
    )

    # Checkpoint-resumable: every evaluated (factors, reg, iterations, mode)
    # combination is appended as one JSON line. An interrupted run reloads the
    # lines and only trains the remaining combinations.
    results = []
    done_keys = set()
    checkpoint = path.with_suffix(".jsonl")
    if not args.no_resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from a kill: ignore
            done_keys.add((row["factors"], row["regularization"], row["iterations"], row["confidence_mode"]))
            results.append(row)
        print(f"[stage] ALS sweep resuming: {len(results)}/{len(als_sweep_configs()) * 4} configs checkpointed")

    configs = als_sweep_configs()
    modes = ["weighted_confidence", "raw_ratings", "log_confidence", "centered_positive"]
    total = len(configs) * len(modes)
    done = len(results)
    sweep_item_ids = sorted(sub["asin"].unique())
    sweep_item_index = {a: i for i, a in enumerate(sweep_item_ids)}
    uidx = np.array([uenc[u] for u in sweep_users], dtype=np.int64)
    with checkpoint.open("a", encoding="utf-8") as ckpt:
        for mode in modes:
            # Fit ONLY on the sweep-temporal-train rows; the hold-out rows are
            # never in the confidence matrix (otherwise every config scores ~1).
            matrix, _ = build_user_item_matrix(sub_train, s_users, s_items, mode=mode)
            print(f"[sweep] confidence mode '{mode}': matrix nnz={matrix.nnz}", flush=True)
            for cfg in configs:
                key = (cfg["factors"], cfg["regularization"], cfg["iterations"], mode)
                if key in done_keys:
                    continue
                t0 = time.time()
                fit = fit_als_with(
                    matrix, s_users, s_items,
                    factors=cfg["factors"], iterations=cfg["iterations"],
                    regularization=cfg["regularization"],
                )
                # Batched eval on the sweep cohort, within the subsample's own
                # item space (self-contained temporal hold-out).
                sweep_scores = als_score_matrix(fit["user_factors"][uidx], fit["item_factors"])

                def sweep_scorer(chunk, _s=sweep_scores, _users=sweep_users):
                    start = _users.index(chunk[0])
                    return _s[start : start + len(chunk)]

                metrics = evaluate_score_batch(
                    sweep_users, sweep_scorer, sweep_item_ids, sweep_item_index, seen_by_user,
                    relevant_by_user, k=10, batch_size=1024,
                )
                row = {
                    **cfg,
                    "confidence_mode": mode,
                    "ndcg": metrics["ndcg"],
                    "accuracy": metrics["accuracy"],
                    "seconds": round(time.time() - t0, 1),
                }
                results.append(row)
                ckpt.write(json.dumps(row) + "\n")
                ckpt.flush()
                done += 1
                print(f"  sweep {done}/{total}: {json.dumps(row)}", flush=True)

    if not results:
        raise RuntimeError("ALS sweep produced no evaluated configs")
    best = max(results, key=lambda r: (r["ndcg"], r["accuracy"]))
    payload = {"best": best, "all": results, "n_rows_swept": int(len(sub)), "n_evaluated": len(results), "n_expected": total}
    path.write_text(json.dumps(payload, indent=2))
    print(f"[stage] ALS sweep complete: {len(results)}/{total} configs evaluated; best: {json.dumps(best)}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Batched scoring helpers used by the evaluation stage
# ---------------------------------------------------------------------------

def make_batch_scorers(user_ids, user_index, als_user_factors, als_item_factors,
                       user_mat, item_mat, user_img_mat, item_img_mat):
    n_items = item_mat.shape[0]

    def cf_batch(chunk):
        idx = np.array([user_index[u] for u in chunk], dtype=np.int64)
        return als_score_matrix(als_user_factors[idx], als_item_factors)

    def text_batch(chunk):
        idx = np.array([user_index[u] for u in chunk], dtype=np.int64)
        return cosine_batch(user_mat[idx], item_mat)

    def image_batch(chunk):
        if user_img_mat is None or item_img_mat is None:
            return None
        idx = np.array([user_index[u] for u in chunk], dtype=np.int64)
        return cosine_batch(user_img_mat[idx], item_img_mat)

    def hybrid_batch_factory(alpha, beta, gamma):
        def hybrid_batch(chunk):
            cf = cf_batch(chunk)
            text = text_batch(chunk)
            img = image_batch(chunk) if gamma > 0 else None
            if img is None:
                return hybrid_scores(cf, text, alpha, normalize=True).astype(np.float32)
            return tri_modal_scores(cf, text, img, alpha, beta, gamma, normalize=True).astype(np.float32)
        return hybrid_batch

    return cf_batch, text_batch, image_batch, hybrid_batch_factory


def choose_eval_users(test_users: list[str], args) -> list[str]:
    if args.full_eval or len(test_users) <= args.eval_users:
        mode = "all test users" if args.full_eval else "all test users (below sample cap)"
        print(f"[eval] cohort: {mode} ({len(test_users)})")
        return list(test_users)
    rng = np.random.default_rng(args.eval_seed)
    picked = sorted(rng.choice(len(test_users), size=args.eval_users, replace=False))
    print(f"[eval] cohort: seeded sample of {args.eval_users} / {len(test_users)} test users (seed={args.eval_seed})")
    return [test_users[i] for i in picked]


def evaluate_split(
    name, train_df, test_df, item_catalog, args, dataset_summary_report, sweep_payload=None
):
    t_start = time.time()
    train_df = train_df.copy()
    test_df = test_df.copy()

    # ---- encoders / index maps (global catalog before split) ----
    user_ids = sorted(train_df["reviewerID"].unique())
    item_ids = sorted(item_catalog["asin"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {a: i for i, a in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)
    train_df["user_idx"] = train_df["reviewerID"].map(user_index)
    train_df["item_idx"] = train_df["asin"].map(item_index)

    train_items = set(train_df["asin"])
    catalog_items = set(item_ids)
    seen_by_user = {u: set(g["asin"]) for u, g in train_df.groupby("reviewerID")}
    history_counts = {u: int(len(g)) for u, g in train_df.groupby("reviewerID")}

    relevant_by_user = relevant_by_user_from(test_df)
    eval_users = choose_eval_users(list(test_df["reviewerID"].unique()), args)

    # ---- interaction mode / ALS config selection ----
    als_cfg = {"factors": 128, "iterations": 30, "regularization": 0.05}
    sweep_report = None
    if args.als_sweep:
        sweep_report = sweep_als(
            train_df, n_users, n_items, args,
            list(test_df["reviewerID"].unique()), relevant_by_user, item_ids, seen_by_user,
        )
        best = sweep_report["best"]
        als_cfg = {"factors": best["factors"], "iterations": best["iterations"], "regularization": best["regularization"]}
        interaction_mode = best["confidence_mode"]
    else:
        interaction_mode = args.interaction_mode

    print(f"[train] full-train ALS: mode={interaction_mode} config={als_cfg} device=cpu (implicit gpu extension not installed)", flush=True)
    matrix, nnz = build_user_item_matrix(train_df, n_users, n_items, mode=interaction_mode)
    print(f"[train] confidence matrix: {n_users} users x {n_items} items, nnz={nnz}", flush=True)
    als = fit_als_with(matrix, n_users, n_items, **als_cfg)
    print(f"[train] ALS fit done: factors {als['user_factors'].shape[1]}", flush=True)

    # ---- semantic (text) modality ----
    print(f"[embeddings] text encoder: {args.embedding_model} on cuda={torch.cuda.is_available()}", flush=True)
    set_embedding_model(args.embedding_model)
    item_vecs, user_vecs = build_semantic_catalog(item_catalog, train_df)
    item_mat = stack_item_matrix(item_ids, item_vecs)
    user_mat = stack_user_matrix(user_ids, user_vecs)
    print(f"[embeddings] text done: item_mat {item_mat.shape}, user_mat {user_mat.shape}", flush=True)

    # ---- vision (CNN) modality (optional) ----
    vision_report = {"enabled": False}
    item_img_mat = user_img_mat = None
    if args.use_vision:
        from recommender.vision import build_image_embeddings, visual_user_profiles, vision_model_dim

        if args.image_limit is None and not args.full_eval:
            print(
                "[vision] WARNING: --use-vision without --image-limit will download and encode "
                "images for the ENTIRE catalog (tens of thousands of URLs; expect 1-3 hours). "
                "Pass --image-limit N for a bounded run.",
                flush=True,
            )
        print(f"[vision] building {args.vision_model} image embeddings (limit={args.image_limit})", flush=True)
        img_vecs, vision_report = build_image_embeddings(
            item_catalog,
            model_name=args.vision_model,
            cache_dir=Path(args.image_cache_dir),
            limit=args.image_limit,
            timeout=args.image_timeout,
        )
        dim = vision_model_dim(args.vision_model)
        item_img_mat = stack_item_matrix(item_ids, img_vecs, dim=dim)
        img_user_vecs = visual_user_profiles(train_df, img_vecs)
        user_img_mat = stack_user_matrix(user_ids, img_user_vecs, dim=dim)
        vision_report["enabled"] = True
        print(f"[vision] {vision_report}")

    pop_series = train_df.groupby("asin").size()
    pop_counts = np.array([float(pop_series.get(a, 0.0)) for a in item_ids], dtype=np.float32)

    alpha = clamp_alpha(args.alpha if args.alpha is not None else DEFAULT_TEMPORAL_ALPHA)
    beta = DEFAULT_BETA_TEXT if args.use_vision else 1.0
    gamma = DEFAULT_GAMMA_IMAGE if args.use_vision else 0.0

    cf_batch, text_batch, image_batch, hybrid_batch_factory = make_batch_scorers(
        user_ids, user_index, als["user_factors"], als["item_factors"],
        user_mat, item_mat, user_img_mat, item_img_mat,
    )

    item_index_global = item_index
    seen_mask_full = build_seen_mask(eval_users, item_index_global, seen_by_user)

    # Headline metrics (0-100): category Accuracy/Recall, sampled-candidate
    # accuracy, like-prediction accuracy. Built once per split; all aggregates
    # come from TRAIN rows only (the held-out label is the like task itself).
    print("[eval] building panel-metric context (categories, held-out labels, train aggregates)...", flush=True)
    panel_ctx = build_panel_context(
        eval_users,
        item_catalog,
        relevant_by_user,
        test_df,
        train_df,
        item_index_global,
        seen_by_user,
        pop_counts,
        history_counts=history_counts,
        seed=args.eval_seed,
    )
    panel_by_model: dict = {}

    def _take_panel(name, result):
        """Split panel keys off a merged result so metrics stay strict-only."""
        if isinstance(result, dict) and "panel" in result:
            panel_by_model[name] = {
                "panel": result.pop("panel"),
                "panel_aux": result.pop("panel_aux", None),
            }
        return result

    # Kill-safe evaluation: every completed variant is checkpointed to
    # validation_cache/ so a crashed/killed run resumes instead of repeating
    # hours of scoring (which already happened once mid-run at full scale).
    cache_key = (
        "full" if args.full_eval else f"sample{args.eval_users}_seed{args.eval_seed}"
    ) + (
        "_vision" if (args.use_vision and item_img_mat is not None) else ""
    ) + "_" + interaction_mode + "_" + "".join(
        c if c.isalnum() else "_" for c in args.embedding_model
    )[:24]
    evaluate_variant, eval_get, flush_eval_cache = evaluate_variant_cached(
        eval_users, item_ids, item_index_global, seen_by_user, relevant_by_user, args, cache_key,
        panel_ctx=panel_ctx,
    )

    # Expected hit rate if recommendations were uniformly random over the
    # candidates each user can actually be shown (catalog minus seen items).
    seen_sizes = np.array([len(seen_by_user.get(u, ())) for u in eval_users], dtype=np.float64)
    n_candidates_mean = max(float(np.mean(np.maximum(n_items - seen_sizes, 1))), 1.0)
    random_expected_rate = min(1.0, 10.0 / n_candidates_mean)
    print(
        f"[eval] expected random HitRate@10 ~ k/catalog = 10/{n_candidates_mean:.0f} "
        f"= {random_expected_rate:.5f}",
        flush=True,
    )

    metrics = {}
    if not args.skip_random:
        print("[eval] variant: Random (O(users*k) sampler)", flush=True)
        metrics["Random"] = _take_panel(
            "Random",
            evaluate_random_batch(
                eval_users, item_ids, seen_by_user, relevant_by_user, k=10,
                panel_ctx=panel_ctx,
            ),
        )
        print(f"[eval] Random: ndcg={metrics['Random']['ndcg']:.4f} hitrate={metrics['Random']['accuracy']:.4f}", flush=True)
    print("[eval] variant: Popularity (batched top-k)", flush=True)
    metrics["Popularity"] = _take_panel(
        "Popularity",
        evaluate_popularity_batch(
            eval_users, item_ids, pop_counts, seen_by_user, relevant_by_user, k=10,
            item_index=item_index_global, panel_ctx=panel_ctx,
        ),
    )
    print(f"[eval] Popularity: ndcg={metrics['Popularity']['ndcg']:.4f} hitrate={metrics['Popularity']['accuracy']:.4f}", flush=True)

    metrics["CF_" + interaction_mode] = _take_panel(
        "CF_" + interaction_mode, eval_get("CF_" + interaction_mode, cf_batch)
    )
    metrics["Semantic"] = _take_panel("Semantic", eval_get("Semantic", text_batch))

    hybrid_batch = hybrid_batch_factory(alpha, beta, gamma)
    metrics[f"Hybrid_alpha_{alpha:g}"] = _take_panel(
        f"Hybrid_alpha_{alpha:g}", eval_get("Hybrid", hybrid_batch)
    )

    if args.use_vision and item_img_mat is not None:
        metrics["Vision"] = _take_panel("Vision", evaluate_variant("Vision", image_batch))

    # ---- dynamic tri-modal fusion (cool users -> content-only) ----
    # Vectorized: per-row weights instead of a per-user Python loop.
    def dynamic_hybrid_batch(chunk):
        cf = cf_batch(chunk)
        text = text_batch(chunk)
        img = image_batch(chunk) if gamma > 0 else None
        cool = np.array([history_counts.get(u, 0) < COOL_USER_THRESHOLD for u in chunk])
        cf_z = zscore_batch(cf)
        text_z = zscore_batch(text)
        if img is None:
            a_row = np.where(cool, 0.0, alpha)[:, None]
            return (a_row * cf_z + (1.0 - a_row) * text_z).astype(np.float32)
        img_z = zscore_batch(img)
        a_row = np.where(cool, 0.0, alpha)[:, None]
        b_row = np.where(cool, 0.7, beta)[:, None]
        g_row = np.where(cool, 0.3, gamma)[:, None]
        return (a_row * cf_z + b_row * text_z + g_row * img_z).astype(np.float32)

    dynamic_name = f"Hybrid_dynamic_alpha_{alpha:g}" + ("_vision" if args.use_vision else "")
    metrics[dynamic_name] = _take_panel(dynamic_name, eval_get("Hybrid(dynamic)", dynamic_hybrid_batch))

    # ---- alpha sweep (bi-modal, serving diagnosis) ----
    alpha_sweep = {}
    alpha_sweep_cache_path = CACHE / f"eval_cache_{cache_key}_alpha_sweep.json"
    alpha_sweep_cache: dict = {}
    if alpha_sweep_cache_path.exists():
        try:
            alpha_sweep_cache = json.loads(alpha_sweep_cache_path.read_text(encoding="utf-8"))
        except Exception:
            alpha_sweep_cache = {}
    if getattr(args, "skip_alpha_sweep", False):
        print("[eval] alpha sweep skipped (--skip-alpha-sweep)", flush=True)
    for a in ([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] if not getattr(args, "skip_alpha_sweep", False) else []):
        if str(a) in alpha_sweep_cache:
            print(f"[eval] cache hit: alpha={a}", flush=True)
            alpha_sweep[str(a)] = alpha_sweep_cache[str(a)]
            continue
        sweep_batch = hybrid_batch_factory(a, 1.0, 0.0)
        alpha_sweep[str(a)] = evaluate_score_batch(
            eval_users, sweep_batch, item_ids, item_index_global, seen_by_user,
            relevant_by_user, k=10, batch_size=args.batch_size,
        )
        alpha_sweep_cache[str(a)] = alpha_sweep[str(a)]
        _atomic_write_text(alpha_sweep_cache_path, json.dumps(alpha_sweep_cache, indent=2))
        print(f"[eval] alpha={a}: ndcg={alpha_sweep[str(a)]['ndcg']:.4f} hitrate={alpha_sweep[str(a)]['accuracy']:.4f}", flush=True)

    # ---- reranker (logistic, no-leakage carve-out) ----
    reranker_report = {"enabled": False, "failure_reason": "disabled via --skip-reranker"}
    reranker_model = None
    rerank_batch = None
    item_feature_df = None  # persisted for serving when the reranker is enabled
    RERANK_POOL_SIZE = 100
    if not args.skip_reranker:
        try:
            from recommender.reranker import (
                HybridReranker,
                build_candidate_pool,
                build_history_asins,
                build_item_features,
                build_reranker_training_set,
                _safe_z_per_row as _rr_zrows,
            )

            print("[reranker] stage: training logistic reranker on temporal carve-out", flush=True)
            # Carve-out inside TRAIN only (no test leakage): the last TRAIN row
            # (rank sizes-2; rank sizes-1 is the held-out test row) is the label;
            # all earlier train rows are the feature history.
            ordered = train_df.sort_values(["reviewerID", "unixReviewTime"], kind="mergesort")
            ranks = ordered.groupby("reviewerID").cumcount()
            sizes = ordered.groupby("reviewerID")["reviewerID"].transform("size")
            feature_hist = ordered.loc[ranks < sizes - 2]
            labels = ordered.loc[ranks == sizes - 2]
            label_by_user = dict(zip(labels["reviewerID"], labels["asin"]))
            feat_history_asins = build_history_asins(feature_hist)

            item_feature_df = build_item_features(item_ids, pop_counts, feature_hist, item_catalog)
            history_asins = build_history_asins(train_df)
            # k = RERANK_POOL_SIZE so the TRAINING pool has exactly the same
            # size and construction as the serving pool (train/serve parity).
            X, y, w, skipped = build_reranker_training_set(
                cf_batch, text_batch, image_batch,
                item_index_global, item_feature_df, feat_history_asins,
                eval_users, label_by_user=label_by_user,
                k=RERANK_POOL_SIZE, max_users=20000,
            )
            if X is not None:
                reranker_model = HybridReranker().fit(X, y, sample_weight=w)
                reranker_report = {
                    "enabled": reranker_model.enabled,
                    "n_train_samples": int(X.shape[0]),
                    "n_features": int(X.shape[-1]) if X.ndim > 1 else 0,
                    "n_skipped_users_label_not_in_pool": int(skipped),
                    "pool_size": RERANK_POOL_SIZE,
                    "train_serve_parity": True,
                    "coefficients": reranker_model.coefficients(),
                    "failure_reason": None if reranker_model.enabled else "degenerate training labels (single class)",
                }
                print(f"[reranker] samples={reranker_report['n_train_samples']} enabled={reranker_report['enabled']}", flush=True)
                if reranker_model.enabled:
                    # Candidate-pool reranking: union of top-100 base hybrid and
                    # top-100 semantic items per user; rerank ONLY that pool
                    # (never the full users x items grid).
                    def rerank_batch(chunk, _rm=reranker_model, _ifd=item_feature_df):
                        cf = cf_batch(chunk)
                        text = text_batch(chunk)
                        img = image_batch(chunk) if gamma > 0 else None
                        cf_z = _rr_zrows(cf)
                        text_z = _rr_zrows(text)
                        base = 0.5 * cf_z + 0.5 * text_z
                        pool_list, hist_list = [], []
                        for row, user in enumerate(chunk):
                            # Exclude already-seen items BEFORE pool selection so
                            # the serving pool matches the training pool, where
                            # history rows are masked out the same way.
                            b = base[row]
                            hist = list(history_asins.get(user, ()))
                            hist_idx = [
                                item_index_global[a] for a in hist if a in item_index_global
                            ]
                            if hist_idx:
                                b = b.copy()
                                b[hist_idx] = -np.inf
                            pool_list.append(build_candidate_pool(b, text_z[row], RERANK_POOL_SIZE))
                            hist_list.append(hist)
                        # One vectorized decision_function call for the whole
                        # batch (identical math to rank_pool, ~5x faster).
                        scores = _rm.rank_pool_batch(
                            cf, text, img, pool_list, _ifd, hist_list, item_index_global
                        )
                        out = np.full_like(cf, -np.inf)
                        for row, pool in enumerate(pool_list):
                            out[row, pool] = scores[row]
                        return out

                    metrics["Reranker"] = _take_panel("Reranker", eval_get("Reranker", rerank_batch))
            else:
                reranker_report["failure_reason"] = (
                    f"no eligible training users (need >=3 carve-out history rows; "
                    f"{skipped} users skipped because the label missed its candidate pool)"
                )
                print(f"[reranker] disabled: {reranker_report['failure_reason']}", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            import traceback
            traceback.print_exc()
            reranker_report = {"enabled": False, "failure_reason": f"exception: {exc}"}
            print(f"[reranker] skipped: {exc}", flush=True)

    # ---- warm / cold cohorts (bounded: evaluate the same cohort on subsets) ----
    print("[eval] stage: computing warm/cold cohorts...", flush=True)
    warm_test = test_df.loc[test_df["asin"].isin(train_items)].reset_index(drop=True)
    cold_test = test_df.loc[~test_df["asin"].isin(train_items)].reset_index(drop=True)
    # NOTE: build the membership sets ONCE. The original comprehensions called
    # set(warm_test["reviewerID"]) inside the per-user condition — O(n^2) that
    # burned ~2 h silently at 183k users (run #3 was killed there; it was
    # actually healthy, just silent).
    warm_user_set = set(warm_test["reviewerID"])
    cold_user_set = set(cold_test["reviewerID"])
    warm_users = [u for u in eval_users if u in warm_user_set]
    cold_users = [u for u in eval_users if u in cold_user_set]
    print(f"[eval] stage: warm/cold cohorts -> warm={len(warm_users)} users, cold={len(cold_users)} users", flush=True)

    # Cohort metrics are the last expensive stage before the bundle save, and
    # they were the exact point where run #2 was killed — checkpoint them too.
    # Per-VARIANT checkpointing + progress prints: the warm cohort takes ~70
    # silent minutes otherwise, which led to a manual ^C of a healthy run.
    cohort_cache_path = CACHE / f"eval_cache_{cache_key}_cohorts.json"
    cohort_cache: dict = {}
    if cohort_cache_path.exists():
        try:
            cohort_cache = json.loads(cohort_cache_path.read_text(encoding="utf-8"))
        except Exception:
            cohort_cache = {}

    def _cohort_variant(users, cohort_name: str, variant: str, fn):
        ck = f"{cohort_name}:{variant}"
        if ck in cohort_cache:
            print(f"[eval] cache hit: {ck}", flush=True)
            return cohort_cache[ck]
        t0 = time.time()
        print(f"[eval] cohort {cohort_name}: evaluating {variant} ({len(users)} users)...", flush=True)
        res = fn()
        print(f"[eval] cohort {cohort_name} {variant}: ndcg={res['ndcg']:.4f} hitrate={res['accuracy']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        cohort_cache[ck] = res
        _atomic_write_text(cohort_cache_path, json.dumps(cohort_cache, indent=2))
        return res

    def cohort_metrics(users, cohort_name: str):
        if not users:
            return {k: {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "ndcg": 0.0, "n_users": 0} for k in metrics}
        out = {}
        if not args.skip_random:
            out["Random"] = _cohort_variant(users, cohort_name, "Random", lambda: evaluate_random_batch(users, item_ids, seen_by_user, relevant_by_user, k=10))
        out["Popularity"] = _cohort_variant(users, cohort_name, "Popularity", lambda: evaluate_popularity_batch(users, item_ids, pop_counts, seen_by_user, relevant_by_user, k=10, item_index=item_index_global))
        out["CF_" + interaction_mode] = _cohort_variant(users, cohort_name, "CF", lambda: evaluate_score_batch(users, cf_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        out["Semantic"] = _cohort_variant(users, cohort_name, "Semantic", lambda: evaluate_score_batch(users, text_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        out[f"Hybrid_alpha_{alpha:g}"] = _cohort_variant(users, cohort_name, "Hybrid", lambda: evaluate_score_batch(users, hybrid_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        if args.use_vision and item_img_mat is not None:
            out["Vision"] = _cohort_variant(users, cohort_name, "Vision", lambda: evaluate_score_batch(users, image_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        out[dynamic_name] = _cohort_variant(users, cohort_name, "Hybrid(dynamic)", lambda: evaluate_score_batch(users, dynamic_hybrid_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        if rerank_batch is not None:
            out["Reranker"] = _cohort_variant(users, cohort_name, "Reranker", lambda: evaluate_score_batch(users, rerank_batch, item_ids, item_index_global, seen_by_user, relevant_by_user, k=10, batch_size=args.batch_size))
        cohort_cache[cohort_name + ":__complete__"] = True
        _atomic_write_text(cohort_cache_path, json.dumps(cohort_cache, indent=2))
        return out

    # Per-variant cache hits print inside _cohort_variant; a run killed mid
    # cohort resumes with only the unfinished variants recomputed.
    warm_metrics = cohort_metrics(warm_users, "warm")
    cold_metrics = cohort_metrics(cold_users, "cold")
    print("[eval] stage: warm/cold metrics done", flush=True)

    # ---- lift summary ----
    random_metrics = metrics.get("Random") or {}
    lift = {}
    for variant, m in metrics.items():
        lift[variant] = {
            "vs_random": lift_over(random_metrics, m),
            "vs_popularity": lift_over(metrics.get("Popularity", {}), m),
        }

    # ---- panel metrics (0-100) per model, plus the supporting Lift ----
    panel_metrics_by_model = {}
    for panel_name, entry in panel_by_model.items():
        values = dict(entry["panel"])
        lv = lift.get(panel_name) or {}
        lift_value = lv.get("vs_random")
        if lift_value is None:
            lift_value = lv.get("vs_popularity")
        values["Lift"] = round(float(lift_value), 1) if lift_value is not None else None
        panel_metrics_by_model[panel_name] = values
    panel_detail = {
        "headline_model": None,  # filled once the ranking is computed below
        "definitions": PANEL_DEFINITIONS,
        "labels": list(PANEL_LABELS),
        "n_eval_users": len(eval_users),
        "seed": args.eval_seed,
        "like_prediction": {
            panel_name: (aux.get("like_prediction") if aux else None)
            for panel_name, aux in (
                (n, e.get("panel_aux")) for n, e in panel_by_model.items()
            )
        },
        "sampled": (
            (next(iter(panel_by_model.values())).get("panel_aux") or {}).get("sampled")
            if panel_by_model else None
        ),
    }

    coverage = {
        "test_rows": int(len(test_df)),
        "warm_test_rows": int(len(warm_test)),
        "cold_test_rows": int(len(cold_test)),
        "reachable_test_rows": int(test_df["asin"].isin(catalog_items).sum()),
        "warm_test_fraction": float(len(warm_test) / len(test_df)) if len(test_df) else 0.0,
        "reachable_test_fraction": float(test_df["asin"].isin(catalog_items).sum() / len(test_df)) if len(test_df) else 0.0,
        "catalog_items": int(len(catalog_items)),
        "train_items": int(len(train_items)),
        "train_catalog_item_fraction": float(len(train_items) / len(catalog_items)) if catalog_items else 0.0,
    }

    scale_rows = []
    for user in eval_users[:8]:
        idx = user_index[user]
        scale_rows.append(
            {
                "user": user,
                "cf_weighted": score_stats(cf_batch([user])[0]),
                "semantic": score_stats(text_batch([user])[0]),
                "train_interactions": history_counts.get(user, 0),
                "effective_alpha": alpha,
            }
        )

    ranking = sorted(metrics.items(), key=lambda kv: kv[1]["ndcg"], reverse=True)
    panel_detail["headline_model"] = ranking[0][0]

    # ---- serving bundle ----
    products = item_catalog.copy()
    histories = train_df[
        [c for c in ["reviewerID", "asin", "overall", "title", "imUrl", "unixReviewTime"] if c in train_df.columns]
    ]
    counts = train_df.groupby("reviewerID").size().sort_values(ascending=False)
    demo_users = [{"user_id": str(u), "train_interactions": int(n)} for u, n in counts.head(5).items()]

    run_mode = "full_dataset" if args.max_lines is None else "smoke_test"
    payload = {
        "split": name,
        "run_mode": run_mode,
        "n_users": n_users,
        "n_items": n_items,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_eval_users": len(eval_users),
        "eval_mode": "full" if args.full_eval else f"sample_{args.eval_users}_seed_{args.eval_seed}",
        "interaction_mode": interaction_mode,
        "matrix_nnz": nnz,
        "random_expected_rate_at_10": random_expected_rate,
        "als_device": "cpu (implicit gpu extension not installed)",
        "default_alpha": alpha,
        "fusion_weights": {"alpha_cf": alpha, "beta_text": beta, "gamma_image": gamma},
        "cool_user_threshold": COOL_USER_THRESHOLD,
        "item_coverage": coverage,
        "dataset_summary": dataset_summary_report,
        "als_selection": {
            "swept": bool(sweep_report),
            "best_config": als_cfg,
            "confidence_mode": interaction_mode,
            "sweep_best": sweep_report["best"] if sweep_report else None,
        },
        "als_orientation": {
            "train_sparse_shape": list(matrix.shape),
            "user_factors": list(als["user_factors"].shape),
            "item_factors": list(als["item_factors"].shape),
            "assertions_passed": True,
            "api": als["api"],
            "warnings": als["warnings"],
            "not_transposed": True,
        },
        "score_scales": scale_rows,
        "metrics": metrics,
        "lift": lift,
        "panel_metrics_by_model": panel_metrics_by_model,
        "panel_detail": panel_detail,
        "warm_item_metrics": warm_metrics,
        "cold_item_metrics": cold_metrics,
        "alpha_zscore": alpha_sweep,
        "vision": vision_report,
        "reranker": reranker_report,
        "ndcg_ranking": [
            {
                "name": k,
                "ndcg": v["ndcg"],
                "precision": v["precision"],
                "recall": v["recall"],
                "accuracy": v.get("accuracy", 0.0),
                "n_users": v["n_users"],
            }
            for k, v in ranking
        ],
        "best_among_tested": ranking[0][0],
        "demo_users": demo_users,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "bundle": {
            "user_factors": als["user_factors"],
            "item_factors": als["item_factors"],
            "item_emb_v1": item_mat,
            "item_emb_v2": item_mat,
            "user_emb_v1": user_mat,
            "user_emb_v2": user_mat,
            "item_img_emb": item_img_mat,
            "user_img_emb": user_img_mat,
            "pop_counts": pop_counts,
            "user_encoder": None,
            "item_encoder": None,
            "user_ids": user_ids,
            "item_ids": item_ids,
            "seen_by_user": seen_by_user,
            "history_counts": history_counts,
            "products": products,
            "histories": histories,
            "n_users": n_users,
            "n_items": n_items,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split": name,
            "als_config": {**als_cfg, "orientation": "users x items, no transpose"},
            "semantic_primary": "v2",
            "semantic_model": args.embedding_model,
            "interaction": {
                "mode": interaction_mode,
                "confidence": "all observed ratings retained; confidence transform: " + interaction_mode,
            },
            "normalization": "per-user z-score on catalog scores",
            "demo_users": demo_users,
            "item_coverage": coverage,
            "reranker": reranker_model,
            # Persist the eval-time per-item feature table so the serving path
            # reranks with exactly the feature distribution training saw.
            "reranker_item_features": (item_feature_df if reranker_model is not None and reranker_model.enabled else None),
            "reranker_config": reranker_report,
            "hybrid": {
                "default_alpha": alpha,
                "alpha_cf": alpha,
                "beta_text": beta,
                "gamma_image": gamma,
                "temporal_default_alpha": DEFAULT_TEMPORAL_ALPHA,
                "userwise_default_alpha": DEFAULT_USERWISE_ALPHA,
                "cool_user_threshold": COOL_USER_THRESHOLD,
                "cool_user_alpha": 0.0,
            },
            "vision": vision_report,
            "run_mode": run_mode,
            "min_user_count": args.min_user_count,
            "min_item_count": args.min_item_count,
        },
    }
    return payload


def main():
    args = parse_args()
    if args.synthetic_min_interactions is not None:
        args.synthetic_min_train_interactions = args.synthetic_min_interactions
    if args.synthetic_max_interactions is not None:
        args.synthetic_max_train_interactions = args.synthetic_max_interactions
    if args.dataset_mode == "synthetic":
        import logging
        from scripts.synthetic import SyntheticConfig, run_synthetic_generation

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        output_dir = args.artifacts_dir or "artifacts_synthetic"
        config = SyntheticConfig(
            n_users=args.synthetic_users,
            min_train_interactions=args.synthetic_min_train_interactions,
            max_train_interactions=args.synthetic_max_train_interactions,
            min_test_interactions=args.synthetic_min_test_interactions,
            max_test_interactions=args.synthetic_max_test_interactions,
            noise_rate=args.synthetic_noise_rate,
            seed=args.synthetic_seed,
            output_dir=Path(output_dir),
            catalog_path=Path(args.synthetic_catalog_path),
        )
        run_synthetic_generation(config)
        return

    if args.artifacts_dir:
        from recommender.paths import set_artifacts_dir

        set_artifacts_dir(args.artifacts_dir)
        print(f"[stage] artifacts will be written to: {args.artifacts_dir}", flush=True)
    print("implicit API", json.dumps(implicit_api_info(), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if args.embedding_model != DEFAULT_EMBEDDING_MODEL:
        set_embedding_model(args.embedding_model)
        print(f"text encoder: {args.embedding_model}")

    data = load_or_build_dataset(args)
    raw_rows = int(data.attrs.get("raw_rows", len(data)))
    print(f"[stage] dataset: {len(data)} rows after k-core (raw {raw_rows})", flush=True)

    item_catalog = build_item_catalog(data)
    print(f"[stage] global item catalog built BEFORE split: {len(item_catalog)} items", flush=True)
    splitters = {
        "temporal_leave_one_last": split_temporal_leave_one_last,
        "userwise_random": split_userwise_random,
    }
    split_names = (
        ["temporal_leave_one_last", "userwise_random"]
        if args.split_type == "both"
        else [args.split_type]
    )

    results = {}
    bundles = {}
    for split_name in split_names:
        print(f"[stage] split: {split_name}", flush=True)
        train_df, test_df = splitters[split_name](data)
        print(f"[stage] split done: train={len(train_df)} test={len(test_df)}", flush=True)
        summary = dataset_summary(
            raw_rows=raw_rows,
            after_merge=data,
            after_kcore=data,
            train_df=train_df,
            test_df=test_df,
            catalog_items=set(item_catalog["asin"]),
            train_items=set(train_df["asin"]),
            min_user_count=args.min_user_count,
            min_item_count=args.min_item_count,
            max_lines=args.max_lines,
        )
        print(f"[stage] evaluating split '{split_name}' (ALS sweep -> fit -> embeddings -> metrics)", flush=True)
        payload = evaluate_split(split_name, train_df, test_df, item_catalog, args, summary)
        bundles[split_name] = payload.pop("bundle")
        results[split_name] = payload
        # Partial artifact: metrics are saved after EVERY split so an
        # interruption during a later split keeps completed results.
        save_partial_metrics(results, args, primary=None)
        print(f"[stage] partial metrics saved for '{split_name}'", flush=True)

    primary = (
        "temporal_leave_one_last"
        if "temporal_leave_one_last" in bundles
        else split_names[0]
    )
    print(f"[stage] saving serving bundle (primary split: {primary})", flush=True)
    save_bundle(bundles[primary])
    save_partial_metrics(results, args, primary=primary)
    print("[stage] done: wrote artifacts/ (config.json, metrics.json, factors.npz, encoders.pkl, products.parquet, histories.parquet)", flush=True)


def save_partial_metrics(results: dict, args, primary: str | None):
    # Headline panel metrics (0-100) for the primary split's best model.
    primary_payload = results.get(primary) if primary else None
    if primary_payload is None:
        primary_payload = results.get("temporal_leave_one_last") or next(
            iter(results.values()), None
        )
    panel_metrics = None
    if primary_payload and primary_payload.get("panel_metrics_by_model"):
        table = primary_payload["panel_metrics_by_model"]
        headline = table.get(primary_payload.get("best_among_tested"))
        # Only publish the headline row once the BEST model has panel numbers;
        # a partial run must never mislabel another model's metrics as the
        # headline.
        if headline is not None:
            panel_metrics = {label: headline.get(label) for label in PANEL_LABELS}
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": "full_dataset" if args.max_lines is None else "smoke_test",
        "dataset": {
            "name": "Amazon Electronics 5-core",
            "source": "full dataset" if args.max_lines is None else f"first {args.max_lines} lines",
            "min_user_count": args.min_user_count,
            "min_item_count": args.min_item_count,
        },
        "eval_config": {
            "full_eval": bool(args.full_eval),
            "eval_users": args.eval_users,
            "eval_seed": args.eval_seed,
            "batch_size": args.batch_size,
            "skip_random": bool(args.skip_random),
            "use_vision": bool(args.use_vision),
            "vision_model": args.vision_model if args.use_vision else None,
            "embedding_model": args.embedding_model,
            "als_sweep": bool(args.als_sweep),
            "reranker": not args.skip_reranker,
        },
        "panel_metrics": panel_metrics,
        "validated_temporal": results.get("temporal_leave_one_last"),
        "validated_userwise": results.get("userwise_random"),
        "serving": {
            "split": primary or "pending",
            "semantic_primary": "v2",
            "semantic_model": args.embedding_model,
            "cf": args.interaction_mode,
            "fusion": "dynamic tri-modal (alpha_cf, beta_text, gamma_image) with cool-user content fallback",
            "vision_enabled": bool(args.use_vision),
            "note": "Serving artifacts are generated from the primary split.",
        },
        "conclusion": (
            "Temporal leave-one-last is the primary protocol. Metrics include full, "
            "warm-item, cold-item, item coverage, lift vs Random/Popularity, and reachable test item reporting."
        ),
    }
    save_metrics(metrics)


if __name__ == "__main__":
    main()
