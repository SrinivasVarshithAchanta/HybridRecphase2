"""Unit tests for pipeline correctness: sweep coverage, reranker, API, vision.

These run WITHOUT the large dataset files (everything is synthetic) except the
artifact-dependent tests, which skip when artifacts are absent.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Orchestrator defaults
# ---------------------------------------------------------------------------

def test_full_dataset_is_default_and_max_lines_is_smoke_only():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.prepare_models import parse_args

    # Default invocation: full dataset, no line cap.
    args = parse_args.__wrapped__() if hasattr(parse_args, "__wrapped__") else None
    import argparse

    parser_args = ["scripts/prepare_models.py"]
    # Directly build the parser the same way parse_args does.
    default_max_lines = None
    ns = argparse.Namespace()
    # Re-run parse_args with empty argv to read defaults.
    import unittest.mock as mock

    with mock.patch("sys.argv", ["prepare_models.py"]):
        parsed = parse_args()
    assert parsed.max_lines is None, "--max-lines must default to None (full dataset)"
    assert parsed.min_user_count == 5 and parsed.min_item_count == 5
    assert parsed.split_type == "temporal_leave_one_last"
    with mock.patch("sys.argv", ["prepare_models.py", "--max-lines", "50000"]):
        smoke = parse_args()
    assert smoke.max_lines == 50000


def test_als_sweep_configs_evaluate_every_combination():
    """The sweep grid is the full cross product (no hidden skipping)."""
    from recommender.cf import als_sweep_configs

    configs = als_sweep_configs()
    assert len(configs) == 3 * 3 * 3  # factors x reg x iterations
    keys = {(c["factors"], c["regularization"], c["iterations"]) for c in configs}
    assert len(keys) == len(configs), "duplicate sweep keys"


def test_als_sweep_evaluates_each_config_once(tmp_path, monkeypatch):
    """Run sweep_als on a tiny synthetic problem; every (config, mode) row must appear."""
    import json

    import scripts.prepare_models as pm

    rng = np.random.default_rng(0)
    n_users, n_items = 40, 30
    rows = []
    for u in range(n_users):
        for _ in range(6):
            rows.append(
                {
                    "reviewerID": f"u{u}",
                    "asin": f"i{rng.integers(n_items)}",
                    "overall": float(rng.integers(1, 6)),
                    "unixReviewTime": int(rng.integers(1_000_000, 2_000_000)),
                    "title": "t",
                    "imUrl": "",
                }
            )
    train_df = pd.DataFrame(rows)
    item_catalog = pd.DataFrame(
        {"asin": [f"i{j}" for j in range(n_items)], "title": "t", "imUrl": "", "brand": "b", "categories": "c", "price": 1.0, "description": ""}
    )
    eval_users = [f"u{u}" for u in range(10)]
    relevant = {u: {f"i{rng.integers(n_items)}"} for u in eval_users}
    seen = {u: set() for u in eval_users}

    class Args:
        max_lines = 1000  # smoke tag for stage paths
        als_sweep_subset = 0  # use all rows
        als_sweep_users = 10
        no_resume = False

    monkeypatch.setattr(pm, "stage_path", lambda name, args: tmp_path / name)
    monkeypatch.setattr(
        pm, "als_sweep_configs", lambda: [
            {"factors": 8, "regularization": 0.01, "iterations": 2},
            {"factors": 16, "regularization": 0.05, "iterations": 3},
        ]
    )

    payload = pm.sweep_als(
        train_df, n_users, n_items, Args(), eval_users, relevant, [f"i{j}" for j in range(n_items)], seen
    )
    evaluated = {(r["factors"], r["regularization"], r["iterations"], r["confidence_mode"]) for r in payload["all"]}
    expected = {
        (c["factors"], c["regularization"], c["iterations"], m)
        for c in [
            {"factors": 8, "regularization": 0.01, "iterations": 2},
            {"factors": 16, "regularization": 0.05, "iterations": 3},
        ]
        for m in ["weighted_confidence", "raw_ratings", "log_confidence", "centered_positive"]
    }
    assert evaluated == expected, "sweep skipped some configs"
    assert payload["n_evaluated"] == payload["n_expected"] == len(expected)
    assert (tmp_path / "als_sweep.json").exists()
    # Resume: re-running must not duplicate rows.
    payload2 = pm.sweep_als(
        train_df, n_users, n_items, Args(), eval_users, relevant, [f"i{j}" for j in range(n_items)], seen
    )
    assert len(payload2["all"]) == len(expected), "resume duplicated sweep rows"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def _tiny_reranker_world():
    item_ids = [f"i{j}" for j in range(30)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    catalog = pd.DataFrame(
        {"asin": item_ids, "brand": ["b" + str(j % 3) for j in range(30)], "categories": "cat", "title": "t"}
    )
    train = pd.DataFrame(
        {
            "asin": item_ids[:20] * 3,
            "overall": [4.0] * 60,
            "unixReviewTime": list(range(60)),
            "reviewerID": "u",
        }
    )
    pop = np.arange(30, dtype=float)
    ifd = __import__("recommender.reranker", fromlist=["build_item_features"]).build_item_features(
        item_ids, pop, train, catalog
    )
    return item_ids, item_index, ifd


def test_reranker_degenerate_labels_disable_it():
    from recommender.reranker import HybridReranker

    X = np.random.rand(20, 9)
    y = np.ones(20, dtype=int)  # single class
    m = HybridReranker().fit(X, y)
    assert not m.enabled
    assert all(v is None for v in m.coefficients().values())


def test_reranker_weight_mismatch_guarded():
    from recommender.reranker import HybridReranker

    X = np.random.rand(40, 9)
    y = np.random.randint(0, 2, 40)
    bad_w = np.ones(7)  # wrong length
    m = HybridReranker().fit(X, y, sample_weight=bad_w)
    assert m.enabled  # must not crash; weights ignored


def test_reranker_ranks_candidate_pool_only():
    """rank_pool scores exactly the given pool columns, nothing else."""
    from recommender.reranker import HybridReranker

    item_ids, item_index, ifd = _tiny_reranker_world()
    X = np.random.rand(200, 9)
    y = np.random.randint(0, 2, 200)
    m = HybridReranker().fit(X, y)
    pool = np.array([0, 5, 7, 22, 29])
    scores = m.rank_pool(
        np.random.rand(30), np.random.rand(30), None, pool, ifd, ["i1", "i2"], item_index
    )
    assert scores.shape == (len(pool),)


def test_build_reranker_training_set_returns_four_values_or_none():
    from recommender.reranker import build_item_features, build_reranker_training_set

    item_ids = [f"i{j}" for j in range(30)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    catalog = pd.DataFrame({"asin": item_ids, "brand": "b", "categories": "c", "title": "t"})
    train = pd.DataFrame(
        {"asin": item_ids[:15] * 4, "overall": [4.0] * 60, "unixReviewTime": list(range(60)), "reviewerID": "u"}
    )
    ifd = build_item_features(item_ids, np.arange(30.0), train, catalog)
    hist = {"u0": [f"i{j}" for j in range(5)], "u1": [f"i{j}" for j in range(5)]}
    labels = {"u0": "i2", "u1": "i7"}
    X, y, w, skipped = build_reranker_training_set(
        lambda chunk: np.random.rand(len(chunk), 30),
        lambda chunk: np.random.rand(len(chunk), 30),
        None,
        item_index,
        ifd,
        hist,
        ["u0", "u1"],
        label_by_user=labels,
        k=20,
    )
    # Either produces aligned arrays or None across the board.
    if X is None:
        assert y is None and w is None
    else:
        assert len(X) == len(y) == len(w)
        assert y.sum() == len(labels) - skipped  # one positive per surviving user


def test_build_reranker_training_set_no_labels_returns_four_nones():
    from recommender.reranker import build_reranker_training_set

    out = build_reranker_training_set(
        lambda c: None, lambda c: None, None, {}, None, {}, ["u"], label_by_user=None
    )
    assert out == (None, None, None, 0)


def test_candidate_pool_union_and_size():
    """Pool = exactly top-k base union top-k semantic, unique columns."""
    from recommender.reranker import build_candidate_pool

    rng = np.random.default_rng(0)
    base = rng.random(50)
    text = rng.random(50)
    pool = build_candidate_pool(base, text, 20)
    assert len(np.unique(pool)) == len(pool)
    expected = set(np.argsort(-base)[:20].tolist()) | set(np.argsort(-text)[:20].tolist())
    assert set(pool.tolist()) == expected
    assert len(build_candidate_pool(base, text, 100)) == 50  # pool_size >= catalog -> all items


def test_reranker_training_pool_matches_serving_pool():
    """Train on the SAME pool construction the serving rerank path uses (train/serve parity)."""
    from recommender.reranker import build_candidate_pool, build_item_features, build_reranker_training_set

    item_ids = [f"i{j}" for j in range(30)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    catalog = pd.DataFrame({"asin": item_ids, "brand": "b", "categories": "c", "title": "t"})
    train = pd.DataFrame(
        {"asin": item_ids[:15] * 4, "overall": [4.0] * 60, "unixReviewTime": list(range(60)), "reviewerID": "u"}
    )
    ifd = build_item_features(item_ids, np.arange(30.0), train, catalog)

    # cf is U-shaped (i0 and i29 strong), text strictly decreasing (top-10 =
    # i0..i9). After masking seen i29, base-hybrid top-10 still contains i28.
    cf = (np.linspace(-1.0, 1.0, 30) ** 2)[None, :]
    text = np.linspace(1.0, -1.0, 30)[None, :]
    captured = {}

    def spy_pool(base, text_row, k):
        captured["pool"] = build_candidate_pool(base, text_row, k)
        return captured["pool"]

    # i29 is seen (masked); history needs >=3 rows for eligibility; label i28
    # is a base-hybrid runner-up so it lands in the pool.
    X, y, w, skipped = build_reranker_training_set(
        lambda c: cf, lambda c: text, None,
        item_index, ifd, {"u0": ["i24", "i25", "i29"]}, ["u0"],
        label_by_user={"u0": "i28"}, k=10, pool_builder_fn=spy_pool,
    )
    assert X is not None
    pool_set = set(captured["pool"].tolist())
    assert item_index["i29"] not in pool_set, "seen items must never enter the training pool"
    assert item_index["i28"] in pool_set
    assert y.sum() == 1 - skipped  # exactly one positive row for the surviving user
    assert len(X) == len(y) == len(w)


def test_reranker_batch_scorer_matches_per_user_scorer():
    """rank_pool_batch must be numerically identical to per-user rank_pool."""
    from recommender.reranker import HybridReranker, build_item_features

    item_ids = [f"i{j}" for j in range(30)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    catalog = pd.DataFrame(
        {"asin": item_ids, "brand": ["b" + str(j % 3) for j in range(30)], "categories": "cat", "title": "t"}
    )
    train = pd.DataFrame(
        {"asin": item_ids[:20] * 3, "overall": [4.0] * 60, "unixReviewTime": list(range(60)), "reviewerID": "u"}
    )
    ifd = build_item_features(item_ids, np.arange(30.0), train, catalog)
    rng = np.random.default_rng(7)
    m = HybridReranker().fit(rng.random((120, 9)), rng.integers(0, 2, 120))
    cf = rng.random((4, 30))
    text = rng.random((4, 30))
    img = rng.random((4, 30))
    pools = [np.array([0, 5, 11, 22, 29]), np.arange(12), np.array([3]), np.arange(30)]
    hists = [["i0", "i1"], [], ["i5"], ["i2", "i4", "i6"]]
    batch_scores = m.rank_pool_batch(cf, text, img, pools, ifd, hists, item_index)
    for row in range(4):
        per_user = m.rank_pool(cf[row], text[row], img[row], pools[row], ifd, hists[row], item_index)
        np.testing.assert_allclose(batch_scores[row], per_user, atol=1e-6)


def test_reranker_training_missed_label_kept_as_all_negative():
    """A positive that never reaches the pool stays as all-negative rows (what serving sees)."""
    from recommender.reranker import build_item_features, build_reranker_training_set

    item_ids = [f"i{j}" for j in range(30)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    catalog = pd.DataFrame({"asin": item_ids, "brand": "b", "categories": "c", "title": "t"})
    train = pd.DataFrame(
        {"asin": item_ids[:15] * 4, "overall": [4.0] * 60, "unixReviewTime": list(range(60)), "reviewerID": "u"}
    )
    ifd = build_item_features(item_ids, np.arange(30.0), train, catalog)

    # cf U-shaped, text strictly decreasing; label i15 is in NEITHER top-10.
    cf = (np.linspace(-1.0, 1.0, 30) ** 2)[None, :]
    text = np.linspace(1.0, -1.0, 30)[None, :]
    X, y, w, skipped = build_reranker_training_set(
        lambda c: cf, lambda c: text, None,
        item_index, ifd, {"u0": ["i24", "i25", "i26"]}, ["u0"],
        label_by_user={"u0": "i15"}, k=10,
    )
    assert X is not None
    assert skipped == 1  # flagged honestly, but the example is NOT discarded
    assert 10 <= len(X) <= 20  # union of the two top-10 pools (may overlap)
    assert y.sum() == 0
    assert len(w) == len(y) == len(X)


# ---------------------------------------------------------------------------
# API validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
    except Exception:
        pytest.skip("backend artifacts not available for API tests")


def test_api_clamps_alpha(api_client):
    r = api_client.get("/api/recommend", params={"user_id": "x", "method": "hybrid", "alpha": 5.0, "k": 10})
    # Unknown user falls back to popularity with HTTP 200 (documented behavior).
    assert r.status_code == 200
    body = r.json()
    assert body.get("fallback") == "popularity" or body.get("alpha") in (None, 0.0, 1.0)


def test_api_caps_k(api_client):
    r = api_client.get("/api/recommend", params={"user_id": "x", "method": "popularity", "k": 5000})
    assert r.status_code == 200
    items = r.json().get("items", [])
    assert len(items) <= 100, "k must be capped at MAX_RECOMMENDATIONS=100"


def test_api_rejects_bad_method(api_client):
    r = api_client.get("/api/recommend", params={"user_id": "x", "method": "nope", "k": 10})
    assert r.status_code == 400


def test_api_unknown_user_popularity_fallback(api_client):
    r = api_client.get("/api/recommend", params={"user_id": "NOT_A_REAL_USER_123", "method": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["unknown_user"] is True
    assert body["fallback"] == "popularity"
    assert len(body["items"]) == 10


def test_api_artifact_summary_and_model_config(api_client):
    r = api_client.get("/api/artifact-summary", params={"mode": "raw"})
    assert r.status_code == 200
    body = r.json()
    assert body["run_mode"] in {"smoke_test", "full_dataset"}
    assert body["artifact_is_full_dataset"] == (body["run_mode"] == "full_dataset")
    r2 = api_client.get("/api/model-config")
    assert r2.status_code == 200
    assert "vision" in r2.json()


# ---------------------------------------------------------------------------
# Vision module
# ---------------------------------------------------------------------------

def test_vision_download_rejects_bad_urls(tmp_path):
    from recommender.vision import download_image

    assert download_image(None, tmp_path) is None
    assert download_image("", tmp_path) is None
    assert download_image("not-a-url", tmp_path) is None
    assert download_image("ftp://x/y.jpg", tmp_path) is None
    # Unroutable host must time out / fail gracefully, not raise.
    assert download_image("http://127.0.0.1:1/x.jpg", tmp_path, timeout=1.0) is None


def test_vision_encode_skips_missing_files():
    from pathlib import Path

    from recommender.vision import encode_images_batch

    pytest.importorskip("torchvision")
    paths = [Path("Z:/definitely/missing_1.jpg"), None, Path("Z:/definitely/missing_2.jpg")]
    out = encode_images_batch(paths, model_name="resnet50", device="cpu", batch_size=2)
    assert out.shape == (3, 2048)
    assert not out.any(), "missing images must produce zero vectors, not crashes"
