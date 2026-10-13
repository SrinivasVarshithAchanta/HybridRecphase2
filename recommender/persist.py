from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths


def _atomic_write(path: Path, writer) -> None:
    """Write to a temp file then atomically replace, with retries.

    On OneDrive-synced folders antivirus/sync can hold a freshly written file
    open for a moment, which surfaces as OSError [Errno 22] (or PermissionError)
    on plain open() — this killed a finished 3.5 h full run right before it
    saved encoders.pkl. Writing to *.tmp and os.replace() also guarantees no
    reader ever sees a half-written artifact.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            with open(tmp, "wb") as f:
                writer(f)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def save_bundle(bundle: dict):
    ARTIFACTS = paths.ARTIFACTS  # read at call time so --artifacts-dir redirects
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    def _write_factors(f):
        np.savez_compressed(
            f,
            user_factors=bundle["user_factors"],
            item_factors=bundle["item_factors"],
            item_emb_v1=bundle["item_emb_v1"],
            item_emb_v2=bundle["item_emb_v2"],
            user_emb_v1=bundle["user_emb_v1"],
            user_emb_v2=bundle["user_emb_v2"],
            pop_counts=bundle["pop_counts"],
            **(
                {
                    "item_img_emb": bundle["item_img_emb"],
                    "user_img_emb": bundle["user_img_emb"],
                }
                if bundle.get("item_img_emb") is not None and bundle.get("user_img_emb") is not None
                else {}
            ),
        )

    _atomic_write(ARTIFACTS / "factors.npz", _write_factors)

    def _write_encoders(f):
        pickle.dump(
            {
                "user_encoder": bundle["user_encoder"],
                "item_encoder": bundle["item_encoder"],
                "user_ids": bundle["user_ids"],
                "item_ids": bundle["item_ids"],
                "seen_by_user": {k: list(v) for k, v in bundle["seen_by_user"].items()},
                "history_counts": bundle.get("history_counts", {}),
                "reranker": bundle.get("reranker"),
                # Eval-time per-item features so serving reranks with EXACTLY
                # the feature distribution the model was trained on.
                "reranker_item_features": bundle.get("reranker_item_features"),
            },
            f,
        )

    _atomic_write(ARTIFACTS / "encoders.pkl", _write_encoders)
    _atomic_write(ARTIFACTS / "products.parquet", lambda f: bundle["products"].to_parquet(f, index=False))
    _atomic_write(ARTIFACTS / "histories.parquet", lambda f: bundle["histories"].to_parquet(f, index=False))
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_users": bundle["n_users"],
        "n_items": bundle["n_items"],
        "train_rows": bundle["train_rows"],
        "test_rows": bundle["test_rows"],
        "split": bundle["split"],
        "als": bundle["als_config"],
        "semantic_primary": bundle["semantic_primary"],
        "semantic_model": bundle.get("semantic_model"),
        "interaction": bundle["interaction"],
        "normalization": bundle["normalization"],
        "demo_users": bundle["demo_users"],
        "item_coverage": bundle.get("item_coverage", {}),
        "hybrid": bundle.get("hybrid", {}),
        "vision": bundle.get("vision", {}),
        "reranker": bundle.get("reranker_config", {}),
        "run_mode": bundle.get("run_mode", "smoke_test"),
        "split_type": bundle.get("split", "temporal_leave_one_last"),
        "min_user_count": bundle.get("min_user_count"),
        "min_item_count": bundle.get("min_item_count"),
    }
    _atomic_write_text(ARTIFACTS / "config.json", json.dumps(config, indent=2))


def load_bundle():
    ARTIFACTS = paths.ARTIFACTS  # read at call time so --artifacts-dir redirects
    factors = np.load(ARTIFACTS / "factors.npz")
    with open(ARTIFACTS / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    products = pd.read_parquet(ARTIFACTS / "products.parquet")
    histories = pd.read_parquet(ARTIFACTS / "histories.parquet")
    config = json.loads((ARTIFACTS / "config.json").read_text(encoding="utf-8"))
    out = {
        "user_factors": factors["user_factors"],
        "item_factors": factors["item_factors"],
        "item_emb_v1": factors["item_emb_v1"],
        "item_emb_v2": factors["item_emb_v2"],
        "user_emb_v1": factors["user_emb_v1"],
        "user_emb_v2": factors["user_emb_v2"],
        "pop_counts": factors["pop_counts"],
        "products": products,
        "histories": histories,
        "config": config,
        "reranker": enc.get("reranker"),
        "reranker_item_features": enc.get("reranker_item_features"),
        **enc,
        "seen_by_user": {k: set(v) for k, v in enc["seen_by_user"].items()},
        "history_counts": enc.get("history_counts", {}),
    }
    # Vision arrays are optional; older artifacts do not contain them.
    if "item_img_emb" in factors and "user_img_emb" in factors:
        out["item_img_emb"] = factors["item_img_emb"]
        out["user_img_emb"] = factors["user_img_emb"]
    return out


def save_metrics(payload: dict):
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(paths.METRICS_PATH, json.dumps(payload, indent=2))


def load_metrics() -> dict:
    return json.loads(paths.METRICS_PATH.read_text(encoding="utf-8"))
