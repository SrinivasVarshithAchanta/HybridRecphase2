from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from recommender.service import RecommenderService, UnknownUserError

app = FastAPI(title="HybridRec", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal artifact model names -> dashboard display names.
# Used by /api/panel-metrics so the frontend can render clean labels directly.
PANEL_MODEL_NAMES = {
    "Hybrid_alpha_0.2": "Hybrid Model",
    "CF_weighted_confidence": "Collaborative Filtering",
    "Semantic": "Semantic Model",
    "Vision": "Visual Model",
    "Hybrid_dynamic_alpha_0.2_vision": "Dynamic Hybrid",
    "Reranker": "Reranked Hybrid",
    "Popularity": "Popularity Baseline",
    "Random": "Random Baseline",
}

_service: RecommenderService | None = None
MAX_RECOMMENDATIONS = 100
ALLOWED_METHODS = {"hybrid", "semantic", "cf", "popularity", "image"}
RAW_ARTIFACTS = ROOT / "artifacts"
SYNTHETIC_ARTIFACTS = ROOT / "artifacts_synthetic"
SYNTHETIC_COMMAND = (
    "python scripts\\prepare_models.py --dataset-mode synthetic "
    "--synthetic-users 10000 --synthetic-min-train-interactions 20 "
    "--synthetic-max-train-interactions 40 --synthetic-min-test-interactions 8 "
    "--synthetic-max-test-interactions 12 --synthetic-noise-rate 0.05 "
    "--artifacts-dir artifacts_synthetic"
)


def service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service


def normalize_mode(mode: str | None) -> str:
    selected = (mode or "synthetic").lower()
    if selected not in {"raw", "synthetic"}:
        raise HTTPException(status_code=400, detail="mode must be 'raw' or 'synthetic'")
    return selected


def synthetic_missing_error() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "error": "synthetic_artifacts_missing",
            "message": "Synthetic artifacts were not found. Run the synthetic pipeline first.",
            "command": SYNTHETIC_COMMAND,
            "artifacts_dir": str(SYNTHETIC_ARTIFACTS),
        },
    )


def load_metrics_for_mode(mode: str | None) -> dict:
    selected = normalize_mode(mode)
    if selected == "raw":
        return service().metrics
    metrics_path = SYNTHETIC_ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        raise synthetic_missing_error()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["run_mode"] = "generated_dataset_evaluation"
    for split_name in ("validated_temporal", "validated_userwise"):
        if isinstance(payload.get(split_name), dict):
            payload[split_name]["run_mode"] = "generated_dataset_evaluation"
    return payload


def primary_split(metrics_payload: dict) -> dict:
    return metrics_payload.get("validated_temporal") or metrics_payload.get("validated_userwise") or {}


def strict_percent_row(row: dict) -> dict:
    return {
        "Precision@10": round(float(row.get("precision", 0.0)) * 100, 2),
        "Recall@10": round(float(row.get("recall", 0.0)) * 100, 2),
        "Accuracy / Hit Rate@10": round(float(row.get("accuracy", 0.0)) * 100, 2),
        "NDCG@10": round(float(row.get("ndcg", 0.0)) * 100, 2),
    }


def clamp_alpha(alpha: float | None) -> float | None:
    if alpha is None:
        return None
    return min(1.0, max(0.0, float(alpha)))


def clamp_weight(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, float(value)))


def clamp_k(k: int) -> int:
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="k must be an integer")
    return min(MAX_RECOMMENDATIONS, max(1, k))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    s = service()
    cfg = s.bundle["config"]
    m = s.metrics
    run_mode = cfg.get("run_mode", m.get("run_mode", "smoke_test"))
    dataset = m.get("dataset", {})
    source = str(dataset.get("source", ""))
    return {
        "phase": "Phase-3 tri-modal prototype",
        "run_mode": run_mode,
        "artifact_is_full_dataset": "full" in source,
        "dataset": dataset,
        "config": cfg,
        "implemented": [
            "Collaborative filtering (implicit ALS, tuned)",
            "Transformer semantic encoder (configurable SBERT models)",
            "CNN image encoder (ResNet-50 / EfficientNet-B0)" if s.vision_enabled else "Text+CF hybrid ranking",
            "Dynamic tri-modal late fusion",
            "Explainable logistic reranker" if s.reranker_enabled else "Offline evaluation with lift vs Random/Popularity",
        ],
        "planned": [
            "Full-catalog vision coverage",
            "True cold-start strategy",
            "Production deployment",
        ],
        "not_active": [] if s.vision_enabled else ["ResNet visual encoder (run with --use-vision)"],
        "vision": s.vision_status(),
        "conclusion": m.get("conclusion"),
        "true_cold_start": "Unknown reviewerIDs are served the popularity fallback with an explicit flag.",
        "sparse_history": "Users with fewer than the cool-user threshold of training interactions get content-only (semantic/visual) fusion.",
    }


@app.get("/api/metrics")
def metrics(mode: str = Query("synthetic")):
    return load_metrics_for_mode(mode)


@app.get("/api/panel-metrics")
def panel_metrics(mode: str = Query("synthetic")):
    """Clean, panel-facing payload: dataset summary + headline panel metrics
    + per-model metrics with friendly display names.

    Strict full-catalog metrics (HR@10, NDCG@10, ...) are intentionally NOT
    included here; use /api/metrics for technical debugging.
    """
    selected = normalize_mode(mode)
    if selected == "synthetic":
        m = load_metrics_for_mode(selected)
        primary = primary_split(m)
        rows = {
            PANEL_MODEL_NAMES.get(name, name): strict_percent_row(row)
            for name, row in (primary.get("metrics") or {}).items()
        }
        best_name = primary.get("best_among_tested") or next(iter(primary.get("metrics") or {}), None)
        return {
            "dataset": {
                "run_mode": m.get("run_mode") or primary.get("run_mode", "generated_dataset_evaluation"),
                "users": primary.get("n_users"),
                "products": primary.get("n_items"),
                "interactions": primary.get("train_rows", 0) + primary.get("test_rows", 0),
                "split": primary.get("split", "synthetic_multi_held_out"),
            },
            "headline": strict_percent_row((primary.get("metrics") or {}).get(best_name, {})) if best_name else {},
            "models": rows,
        }

    s = service()
    m = s.metrics
    cfg = s.bundle["config"]
    primary = primary_split(m)
    ds = primary.get("dataset_summary") or {}
    headline = m.get("panel_metrics") or {}
    by_model = primary.get("panel_metrics_by_model") or {}
    models = {
        PANEL_MODEL_NAMES.get(name, name): row
        for name, row in by_model.items()
    }
    return {
        "dataset": {
            "run_mode": m.get("run_mode") or cfg.get("run_mode", "smoke_test"),
            "users": ds.get("users_after_kcore") or primary.get("n_users") or cfg.get("n_users"),
            "products": ds.get("items_after_kcore") or primary.get("n_items") or cfg.get("n_items"),
            "interactions": ds.get("rows_after_kcore") or primary.get("train_rows", 0) + primary.get("test_rows", 0),
            "split": primary.get("split") or cfg.get("split") or cfg.get("split_type") or "-",
        },
        "headline": headline,
        "models": models,
    }


@app.get("/api/artifact-summary")
def artifact_summary(mode: str = Query("synthetic")):
    """Run-mode banner data: smoke vs full, timestamp, sizes, split type."""
    selected = normalize_mode(mode)
    if selected == "synthetic":
        m = load_metrics_for_mode(selected)
        primary = primary_split(m)
        metadata_path = SYNTHETIC_ARTIFACTS / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        dataset = m.get("dataset", {})
        return {
            "run_mode": m.get("run_mode", "generated_dataset_evaluation"),
            "artifact_is_full_dataset": False,
            "source": dataset.get("source", "synthetic (derived from Amazon Electronics catalog metadata)"),
            "created_utc": m.get("created_utc") or metadata.get("generated_at"),
            "n_users": primary.get("n_users") or dataset.get("n_users"),
            "n_items": primary.get("n_items") or dataset.get("n_items"),
            "train_rows": primary.get("train_rows") or dataset.get("train_rows"),
            "test_rows": primary.get("test_rows") or dataset.get("test_rows"),
            "split_type": primary.get("split") or dataset.get("split_type", "synthetic_multi_held_out"),
            "dataset_summary": primary.get("dataset_summary") or metadata.get("dataset_summary"),
            "eval_mode": primary.get("eval_mode", "all_test_users"),
            "vision": {"enabled": False},
            "reranker_enabled": False,
        }

    s = service()
    cfg = s.bundle["config"]
    m = s.metrics
    primary = primary_split(m)
    source = str(m.get("dataset", {}).get("source", ""))
    is_full = "full" in source
    return {
        "run_mode": cfg.get("run_mode", "full_dataset" if is_full else "smoke_test"),
        "artifact_is_full_dataset": is_full,
        "source": source,
        "created_utc": cfg.get("created_utc") or m.get("created_utc"),
        "n_users": cfg.get("n_users"),
        "n_items": cfg.get("n_items"),
        "train_rows": cfg.get("train_rows"),
        "test_rows": cfg.get("test_rows"),
        "split_type": cfg.get("split") or cfg.get("split_type"),
        "dataset_summary": primary.get("dataset_summary"),
        "eval_mode": primary.get("eval_mode"),
        "vision": s.vision_status(),
        "reranker_enabled": s.reranker_enabled,
    }


@app.get("/api/model-config")
def model_config():
    """Active model configuration: ALS, semantic, vision, fusion, reranker."""
    s = service()
    cfg = s.bundle["config"]
    primary = s.metrics.get("validated_temporal") or {}
    return {
        "als": cfg.get("als"),
        "als_selection": primary.get("als_selection"),
        "interaction": cfg.get("interaction"),
        "semantic": {
            "primary": cfg.get("semantic_primary"),
            "model": cfg.get("semantic_model"),
        },
        "vision": s.vision_status(),
        "fusion": cfg.get("hybrid"),
        "reranker": {
            "enabled": s.reranker_enabled,
            "coefficients": (cfg.get("reranker") or {}).get("coefficients"),
            "n_train_samples": (cfg.get("reranker") or {}).get("n_train_samples"),
        },
        "normalization": cfg.get("normalization"),
        "cool_user_threshold": (cfg.get("hybrid") or {}).get("cool_user_threshold"),
    }


@app.get("/api/users")
def users(q: str = ""):
    s = service()
    demo = s.bundle["config"]["demo_users"]
    all_ids = list(s.bundle["user_ids"])
    if q:
        ql = q.lower()
        matches = [u for u in all_ids if ql in u.lower()][:20]
    else:
        matches = []
    return {"demo_users": demo, "matches": matches, "n_known_users": len(all_ids)}


@app.get("/api/users/{user_id}/history")
def history(user_id: str):
    s = service()
    if not s.known_user(user_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_user",
                "true_cold_start": True,
                "message": "This reviewer has no training history.",
            },
        )
    return s.user_history(user_id)


@app.get("/api/recommend")
def recommend(
    user_id: str,
    method: str = "hybrid",
    alpha: float | None = None,
    k: int = 10,
):
    if method not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail=f"method must be one of {sorted(ALLOWED_METHODS)}")
    s = service()
    if method == "image" and not s.vision_enabled:
        raise HTTPException(status_code=400, detail="Vision modality is not enabled in this artifact. Rebuild with --use-vision.")
    alpha = clamp_alpha(alpha)
    k = clamp_k(k)
    try:
        return s.recommend(user_id, method=method, alpha=alpha, k=k)
    except UnknownUserError:
        # Unknown users get the popularity fallback instead of a hard 404,
        # clearly labeled so the UI can say "not personalized".
        return {
            "user_id": user_id,
            "method": "popularity",
            "fallback": "popularity",
            "unknown_user": True,
            "k": k,
            "alpha": None,
            "train_interactions": 0,
            "dynamic_fusion": True,
            "semantic_only_fallback": True,
            "contributions": {"cf": 0.0, "text": 0.0, "image": 0.0},
            "vision_enabled": s.vision_enabled,
            "items": s.popularity_fallback(k=k),
        }


@app.get("/api/compare")
def compare(user_id: str, k: int = 10, alpha: float | None = None):
    s = service()
    alpha = clamp_alpha(alpha)
    k = clamp_k(k)
    try:
        return s.compare(user_id, k=k, alpha=alpha)
    except UnknownUserError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_user",
                "true_cold_start": True,
                "message": "This reviewer has no training history.",
            },
        )


@app.get("/api/model-config/weights")
def validate_weights(
    alpha_cf: float = Query(0.2),
    beta_text: float = Query(0.6),
    gamma_image: float = Query(0.2),
):
    """Echo back clamped fusion weights so the UI can preview them."""
    return {
        "alpha_cf": clamp_weight(alpha_cf),
        "beta_text": clamp_weight(beta_text),
        "gamma_image": clamp_weight(gamma_image),
    }
