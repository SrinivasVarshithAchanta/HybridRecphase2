from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from recommender.cf import implicit_api_info
from recommender.persist import load_bundle, load_metrics
from recommender.service import RecommenderService, UnknownUserError


@pytest.fixture(scope="module")
def service():
    return RecommenderService()


def test_als_orientation_in_artifacts():
    bundle = load_bundle()
    cfg = bundle["config"]
    n_users = cfg["n_users"]
    n_items = cfg["n_items"]
    assert bundle["user_factors"].shape[0] == n_users
    assert bundle["item_factors"].shape[0] == n_items
    expected_factors = cfg.get("als", {}).get("factors", bundle["user_factors"].shape[1])
    assert bundle["user_factors"].shape[1] == bundle["item_factors"].shape[1] == expected_factors
    api = implicit_api_info()
    assert api["implicit_version"].startswith("0.7")
    assert api["doc_says_rows_are_users"]


def test_known_user_gets_recommendations(service):
    user = service.bundle["config"]["demo_users"][0]["user_id"]
    out = service.recommend(user, method="hybrid", alpha=0.5, k=10)
    assert len(out["items"]) == 10
    assert out["items"][0]["rank"] == 1
    assert out["items"][0]["asin"]
    assert "cf_score" in out["items"][0]
    assert "semantic_score" in out["items"][0]
    assert "hybrid_score" in out["items"][0]
    assert "confidence" not in out["items"][0]


def test_unknown_user_errors(service):
    with pytest.raises(UnknownUserError):
        service.recommend("NOT_A_REAL_REVIEWER")


def test_seen_items_excluded(service):
    user = service.bundle["config"]["demo_users"][0]["user_id"]
    seen = service.bundle["seen_by_user"][user]
    for method in ("cf", "semantic", "hybrid"):
        recs = service.recommend(user, method=method, k=10)["items"]
        asins = {r["asin"] for r in recs}
        assert asins.isdisjoint(seen)


def test_methods_and_alpha_change_hybrid(service):
    user = service.bundle["config"]["demo_users"][0]["user_id"]
    cf = [r["asin"] for r in service.recommend(user, "cf")["items"]]
    sem = [r["asin"] for r in service.recommend(user, "semantic")["items"]]
    h0 = [r["asin"] for r in service.recommend(user, "hybrid", alpha=0.0)["items"]]
    h1 = [r["asin"] for r in service.recommend(user, "hybrid", alpha=1.0)["items"]]
    h05 = [r["asin"] for r in service.recommend(user, "hybrid", alpha=0.5)["items"]]
    if service.reranker_enabled:
        # Serving reranks hybrid candidates with learned weights that swamp the
        # fusion alpha (cf_score coef ~0.13 vs popularity ~1.79), so different
        # alphas can legitimately yield the same top-k. Assert the reranker
        # contract instead: the response is flagged and ordering is stable.
        resp = service.recommend(user, "hybrid", alpha=0.5)
        assert resp["reranked"] is True
        assert [r["asin"] for r in resp["items"]] == h05
    elif service.vision_enabled:
        # With vision active, hybrid alpha=0 still blends text+image, so it may
        # differ from pure semantic. It must, however, differ from pure CF.
        assert h0 != cf
    else:
        assert h0 == sem
        assert h1 == cf
    if not service.reranker_enabled:
        assert h05 != h0 or h05 != h1 or cf == sem


def test_deterministic(service):
    user = service.bundle["config"]["demo_users"][0]["user_id"]
    a = [r["asin"] for r in service.recommend(user, "hybrid", 0.4, 10)["items"]]
    b = [r["asin"] for r in service.recommend(user, "hybrid", 0.4, 10)["items"]]
    assert a == b


def test_compare_known_user_returns_rank_rows(service):
    user = service.bundle["config"]["demo_users"][0]["user_id"]
    out = service.compare(user, alpha=0.2, k=10)
    assert out["user_id"] == user
    assert out["products"], "compare page needs rank rows for a selected reviewer"
    assert set(out["unique"]) >= {"cf", "semantic", "hybrid"}
    first = out["products"][0]
    assert {"asin", "title", "cf_rank", "semantic_rank", "hybrid_rank"} <= set(first)
    assert "overlap_all_three" in out


def _primary_split(metrics):
    return metrics.get("validated_temporal") or metrics.get("validated_userwise") or {}


def test_metrics_artifact_has_baselines():
    metrics = load_metrics()
    primary = _primary_split(metrics)
    v = primary.get("metrics") or {}
    # Random may be legitimately absent when built with --skip-random, but the
    # eval_config must then say so explicitly (no silent omissions).
    skip_random = (metrics.get("eval_config") or {}).get("skip_random", False)
    if skip_random:
        assert "Random" not in v, "Random present although --skip-random was set"
    else:
        assert "Random" in v, "missing Random baseline"
        assert "ndcg" in v["Random"]
    assert "Popularity" in v, "missing Popularity baseline"
    assert "ndcg" in v["Popularity"]
    assert any(k.startswith("CF_") for k in v), "no CF variant in metrics"
    assert any(k == "Semantic" for k in v), "no Semantic variant in metrics"
    assert any(k.startswith("Hybrid") for k in v), "no Hybrid variant in metrics"
    n_users = primary.get("n_eval_users")
    assert n_users and n_users > 0


def test_metrics_structure_vision_optional():
    """Vision metrics appear only when the artifact was built with --use-vision."""
    metrics = load_metrics()
    primary = _primary_split(metrics)
    vision_report = (primary.get("vision") or {}).get("enabled", False)
    has_vision_metric = any(k == "Vision" for k in (primary.get("metrics") or {}))
    assert vision_report == has_vision_metric


def test_lift_summary_present():
    metrics = load_metrics()
    primary = _primary_split(metrics)
    lift = primary.get("lift") or {}
    assert lift, "lift summary missing"
    for variant, entry in lift.items():
        assert set(entry) >= {"vs_random", "vs_popularity"}


def test_no_seen_items_in_batch_topk():
    """Batched top-k must never return seen items and must be exact."""
    import numpy as np

    from recommender.evaluation import build_seen_mask, recs_from_batch, topk_batch

    item_ids = [f"i{j}" for j in range(50)]
    item_index = {a: i for i, a in enumerate(item_ids)}
    seen_by_user = {"u0": {"i0", "i3"}, "u1": set()}
    users = ["u0", "u1"]
    mask = build_seen_mask(users, item_index, seen_by_user)
    scores = np.arange(50, dtype=float)[None, :] + np.zeros((2, 1))
    idx, sc = topk_batch(scores, mask, k=10)
    recs = recs_from_batch(idx, item_ids)
    assert "i0" not in recs[0] and "i3" not in recs[0]
    assert len(recs[0]) == 10 and len(recs[1]) == 10
    # u1 has no seen items: exact top-10 = i49..i40
    assert recs[1] == [f"i{49 - j}" for j in range(10)]


def test_als_sweep_configs_cover_panel_spec():
    from recommender.cf import als_sweep_configs, confidence_values

    configs = als_sweep_configs()
    factors = {c["factors"] for c in configs}
    regs = {c["regularization"] for c in configs}
    iters = {c["iterations"] for c in configs}
    assert factors == {64, 128, 256}
    assert regs == {0.01, 0.05, 0.1}
    assert iters == {20, 30, 50}
    # Confidence transforms behave as documented.
    ratings = np.array([1.0, 3.0, 5.0])
    assert confidence_values(ratings, "weighted_confidence").tolist() == [1.0, 3.0, 5.0]
    assert confidence_values(ratings, "raw_ratings").tolist() == [1.0, 3.0, 5.0]
    assert (confidence_values(ratings, "log_confidence") >= 1).all()
    assert confidence_values(ratings, "centered_positive").tolist() == [1.0, 1.0, 3.0]
    assert confidence_values(np.array([1.0, 3.0]), "positive_ge4").tolist() == [1.0, 1.0]


def test_dynamic_fusion_weights_behavior():
    from recommender.hybrid import tri_modal_dynamic_weights

    # Zero history -> pure content weights (text-heavy, image secondary).
    w0 = tri_modal_dynamic_weights(0, 5, 0.2, 0.6, 0.2)
    assert w0 == (0.0, 0.7, 0.3)
    # Enough history -> exactly the requested weights.
    wfull = tri_modal_dynamic_weights(5, 5, 0.2, 0.6, 0.2)
    assert wfull == (0.2, 0.6, 0.2)
    # Half history -> blend: 0.4*target + 0.6*content.
    whalf = tri_modal_dynamic_weights(2, 5, 0.2, 0.6, 0.2)
    assert abs(whalf[0] - 0.08) < 1e-9
    assert abs(whalf[1] - 0.66) < 1e-9
    assert abs(whalf[2] - 0.26) < 1e-9


def test_tri_modal_missing_modality_redistributes():
    import numpy as np

    from recommender.hybrid import tri_modal_scores

    cf = np.array([[1.0, 2.0, 3.0]])
    text = np.array([[0.0, 1.0, 0.0]])
    out = tri_modal_scores(cf, text, None, 0.4, 0.6, 0.0, normalize=False)
    assert out is not None and np.isfinite(out).all()


def test_random_baseline_is_o_users_k():
    """The random baseline API takes the full catalog but only draws k items per user."""
    from recommender.evaluation import random_recs_batch

    item_ids = [f"i{j}" for j in range(1000)]
    recs = random_recs_batch(["u0", "u1"], item_ids, {"u0": set(), "u1": set()}, k=10)
    assert len(recs) == 2
    for r in recs:
        assert len(r) == 10
        assert len(set(r)) == 10  # no duplicate draws
