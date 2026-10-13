"""Headline metrics tests: category Accuracy/Recall, sampled candidates,
like prediction, 0-100 percentage scaling, and artifact/API compatibility.

Everything here is synthetic except the optional API check, which skips until
the artifacts contain ``panel_metrics`` (i.e. after a pipeline regeneration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel_fixture():
    """Tiny catalog/train/test + panel context shared by the panel tests."""
    from recommender.evaluation import build_panel_context

    n_items = 300
    item_ids = [f"i{j}" for j in range(n_items)]
    item_index = {a: i for i, a in enumerate(item_ids)}

    categories = []
    for j in range(n_items):
        if j < 10:
            categories.append([["Electronics", "Computers & Accessories", "Tablets"]])
        elif j < 20:
            categories.append([["Electronics", "Camera & Photo", "Digital Cameras"]])
        elif j < 30:
            categories.append([["Electronics", "Accessories & Supplies", "Headphones"]])
        else:
            categories.append([["Electronics", "Misc", f"Cat{j - 30}"]])
    # i5 is a multi-category product (two paths).
    categories[5] = [
        ["Electronics", "Computers & Accessories", "Tablets"],
        ["Office Products", "Scanners"],
    ]
    catalog = pd.DataFrame({"asin": item_ids, "title": "t", "categories": categories})

    rows = []
    for u in range(20):
        user = f"u{u}"
        # Noise items live in i150..i299 so held-out items (i0..i61) are never seen.
        for i in range(4):
            rows.append(
                {
                    "reviewerID": user,
                    "asin": f"i{150 + (u * 7 + i * 3) % 150}",
                    "overall": float(1 + ((u + i) % 5)),
                    "unixReviewTime": 1_600_000_000 + i,
                }
            )
        # Train-only like signal: odd users like i40/i41, even users dislike i60/i61.
        if u % 2 == 1:
            rows.append({"reviewerID": user, "asin": "i40", "overall": 5.0, "unixReviewTime": 1_600_000_100})
            rows.append({"reviewerID": user, "asin": "i41", "overall": 5.0, "unixReviewTime": 1_600_000_101})
        else:
            rows.append({"reviewerID": user, "asin": "i60", "overall": 2.0, "unixReviewTime": 1_600_000_100})
            rows.append({"reviewerID": user, "asin": "i61", "overall": 2.0, "unixReviewTime": 1_600_000_101})
    train_df = pd.DataFrame(rows)

    hidden_asin = {
        "u0": "i0",   # Tablets
        "u1": "i10",  # Cameras
        "u2": "i5",   # Tablets + Scanners (multi-category)
        "u3": "i20",  # Headphones
        "u4": "i25",  # Headphones
    }
    test_rows = []
    for u in range(20):
        user = f"u{u}"
        if user in hidden_asin:
            asin = hidden_asin[user]
        else:
            asin = "i40" if u % 2 == 0 else "i60"
        overall = 5.0 if u % 2 == 0 else 2.0  # balanced like/not-like labels
        test_rows.append(
            {
                "reviewerID": user,
                "asin": asin,
                "overall": overall,
                "unixReviewTime": 1_700_000_000 + u,
            }
        )
    test_df = pd.DataFrame(test_rows)
    relevant = {r.reviewerID: {r.asin} for r in test_df.itertuples(index=False)}
    seen = {u: set(g["asin"]) for u, g in train_df.groupby("reviewerID")}
    pop_counts = np.arange(n_items, dtype=np.float32)
    eval_users = [f"u{u}" for u in range(20)]
    ctx = build_panel_context(
        eval_users, catalog, relevant, test_df, train_df,
        item_index, seen, pop_counts, seed=42,
    )
    return {
        "ctx": ctx,
        "eval_users": eval_users,
        "item_ids": item_ids,
        "item_index": item_index,
        "catalog": catalog,
        "train_df": train_df,
        "test_df": test_df,
        "relevant": relevant,
        "seen": seen,
        "pop_counts": pop_counts,
        "n_items": n_items,
    }


def test_parse_category_paths_nested_flat_and_missing():
    from recommender.evaluation import build_category_index, parse_category_paths

    nested = parse_category_paths([["Electronics", "Computers & Accessories", "Tablets"]])
    assert nested == (("Electronics", "Computers & Accessories", "Tablets"),)
    # Flattened catalog strings compare by exact category-path equality too.
    flat = parse_category_paths("Electronics Computers & Accessories Tablets")
    assert flat == (("Electronics Computers & Accessories Tablets",),)
    assert parse_category_paths(float("nan")) == ()
    assert parse_category_paths(None) == ()
    assert parse_category_paths("") == ()
    multi = parse_category_paths([["A", "B"], ["C"]])
    assert multi == (("A", "B"), ("C",))
    idx = build_category_index(
        pd.DataFrame({"asin": ["x"], "categories": [[["Electronics", "Tablets"]]]})
    )
    assert idx == {"x": (("Electronics", "Tablets"),)}


def test_build_panel_context_labels_stats_and_hidden_categories():
    fx = _panel_fixture()
    ctx = fx["ctx"]
    # Hidden category paths come from the held-out product.
    assert ctx["hidden_paths"]["u0"] == frozenset(
        {("Electronics", "Computers & Accessories", "Tablets")}
    )
    assert len(ctx["hidden_paths"]["u2"]) == 2  # multi-category product
    # Held-out column + like label (rating >= 4 => liked).
    assert ctx["true_col"]["u0"] == fx["item_index"]["i0"]
    assert ctx["label"]["u0"] == 1 and ctx["label"]["u1"] == 0
    # Train-only user stats: odd users liked two 5-star rows.
    _, like_rate_u1, count_u1 = ctx["user_stats"]["u1"]
    assert like_rate_u1 > 0.5 and count_u1 == 6
    # Seen items are excluded from sampled negatives.
    assert set(ctx["seen_cols"]["u0"].tolist()) == {
        fx["item_index"][a] for a in fx["seen"]["u0"]
    }


def test_category_accuracy_and_recall_exact_values():
    from recommender.evaluation import PanelAccumulator

    fx = _panel_fixture()
    acc = PanelAccumulator(fx["ctx"], k=10, seed=42)
    users = ["u0", "u1", "u2"]
    rec_lists = [
        ["i1", "i20"],   # hidden Tablets -> match
        ["i20", "i21"],  # hidden Cameras -> no match
        ["i1", "i30"],   # hidden {Tablets, Scanners} -> 1 of 2 covered
    ]
    acc.add(users, rec_lists, scores=None)
    out = acc.finalize()
    assert out["panel"]["Accuracy"] == 66.7  # 2 of 3 users matched
    assert out["panel"]["Recall"] == 50.0    # (1 + 0 + 0.5) / 3
    assert out["panel_aux"]["n_users"] == 3


def test_sampled_candidate_accuracy_is_deterministic_and_excludes_seen():
    from recommender.evaluation import PanelAccumulator, SAMPLED_NEGATIVES

    fx = _panel_fixture()
    ctx = fx["ctx"]
    acc = PanelAccumulator(ctx, k=10, seed=42)
    users = ["u0", "u1"]
    n_items = fx["n_items"]
    scores = np.zeros((2, n_items), dtype=np.float32)
    t0, t1 = ctx["true_col"]["u0"], ctx["true_col"]["u1"]
    scores[0, t0] = 50.0   # held-out item outranks every sampled candidate
    scores[1, t1] = -50.0  # held-out item ranks dead last among candidates
    acc.add(users, [["i1"], ["i11"]], scores=scores)
    neg = acc.last_negatives
    assert neg.shape == (2, SAMPLED_NEGATIVES)
    for row, true_col in ((0, t0), (1, t1)):
        assert true_col not in neg[row], "held-out item must not be its own negative"
        for seen_col in ctx["seen_cols"][users[row]]:
            assert seen_col not in neg[row], "seen items must not be sampled as negatives"
    out = acc.finalize()
    assert out["panel"]["Sampled Accuracy"] == 50.0  # 1 hit of 2 users
    assert out["panel_aux"]["sampled"]["candidates"] == 100
    assert out["panel_aux"]["sampled"]["negatives"] == 99


def test_random_ranker_sampled_accuracy_near_ten_percent():
    from recommender.evaluation import PanelAccumulator

    fx = _panel_fixture()
    users = [f"ghost{u}" for u in range(3000)]  # no scores: uniform random ranks
    acc = PanelAccumulator(fx["ctx"], k=10, seed=7)
    acc.add(users, [["i1"] for _ in users], scores=None)
    out = acc.finalize()
    value = out["panel"]["Sampled Accuracy"]
    assert 6.0 <= value <= 14.0, f"random sampled accuracy should be ~10%, got {value}"


def test_like_prediction_accuracy_beats_majority_baseline():
    from recommender.evaluation import PanelAccumulator

    fx = _panel_fixture()
    acc = PanelAccumulator(fx["ctx"], k=10, seed=42)
    scores = np.zeros((len(fx["eval_users"]), fx["n_items"]), dtype=np.float32)
    acc.add(fx["eval_users"], [["i1"] for _ in fx["eval_users"]], scores=scores)
    out = acc.finalize()
    like = out["panel_aux"]["like_prediction"]
    assert like["n_users"] == 20
    assert like["roc_auc"] is not None and like["roc_auc"] >= 0.8
    assert out["panel"]["Like Prediction Accuracy"] >= like["baseline_accuracy"] + 10
    assert like["label"].startswith("rating >= 4")


def test_panel_values_are_scaled_0_to_100_with_required_labels():
    from recommender.evaluation import PANEL_LABELS, PanelAccumulator

    fx = _panel_fixture()
    acc = PanelAccumulator(fx["ctx"], k=10, seed=42)
    acc.add(fx["eval_users"], [["i1"] for _ in fx["eval_users"]], scores=None)
    out = acc.finalize()
    assert tuple(out["panel"]) == PANEL_LABELS[:4]
    for key, value in out["panel"].items():
        assert isinstance(value, float), key
        assert 0.0 <= value <= 100.0, f"{key}={value} outside 0-100"
        # Already scaled to percentage points with at most one decimal.
        assert round(value, 1) == value, f"{key}={value} not 0-100 scaled"
    assert "Lift" in PANEL_LABELS  # appended by the pipeline, not the accumulator


def test_evaluate_score_batch_panel_and_strict_modes():
    from recommender.evaluation import METRIC_KEYS, evaluate_score_batch

    fx = _panel_fixture()
    item_index = fx["item_index"]
    item_ids = fx["item_ids"]
    true_cols = np.array(
        [fx["ctx"]["true_col"].get(u, 0) for u in fx["eval_users"]], dtype=np.int64
    )
    rng = np.random.default_rng(3)

    def score_fn(chunk):
        rows = np.array([fx["eval_users"].index(u) for u in chunk])
        mat = rng.random((len(chunk), fx["n_items"]), dtype=np.float32) * 0.1
        mat[np.arange(len(chunk)), true_cols[rows]] = 10.0  # held-out item ranks first
        return mat

    common = dict(
        eval_users=fx["eval_users"],
        score_fn=score_fn,
        item_ids=item_ids,
        item_index=item_index,
        seen_by_user=fx["seen"],
        relevant_by_user=fx["relevant"],
        k=10,
        batch_size=8,
    )
    strict_only = evaluate_score_batch(**common)
    assert set(strict_only) == set(METRIC_KEYS) | {"n_users", "mean_relevant_items"}
    assert strict_only["accuracy"] == 1.0  # held-out item always ranked first

    both = evaluate_score_batch(**common, panel_ctx=fx["ctx"])
    assert "panel" in both and "panel_aux" in both
    assert set(both) - {"panel", "panel_aux"} == set(strict_only)
    assert both["panel"]["Sampled Accuracy"] == 100.0

    panel_only = evaluate_score_batch(
        **common, panel_ctx=fx["ctx"], compute_strict=False
    )
    assert "ndcg" not in panel_only and "accuracy" not in panel_only
    assert panel_only["panel"] == both["panel"]  # same seed => identical numbers
    with pytest.raises(ValueError):
        evaluate_score_batch(**common, compute_strict=False)


def test_evaluate_random_and_popularity_batch_gain_panel_keys():
    from recommender.evaluation import evaluate_popularity_batch, evaluate_random_batch

    fx = _panel_fixture()
    rnd = evaluate_random_batch(
        fx["eval_users"], fx["item_ids"], fx["seen"], fx["relevant"],
        k=10, panel_ctx=fx["ctx"],
    )
    assert "panel" in rnd and "ndcg" in rnd
    assert 0 <= rnd["panel"]["Accuracy"] <= 100
    pop = evaluate_popularity_batch(
        fx["eval_users"], fx["item_ids"], fx["pop_counts"], fx["seen"], fx["relevant"],
        k=10, item_index=fx["item_index"], panel_ctx=fx["ctx"],
    )
    assert "panel" in pop and "ndcg" in pop
    # Popularity tops the catalog with i0-class items for everyone in this fixture.
    assert 0 <= pop["panel"]["Sampled Accuracy"] <= 100


def test_save_partial_metrics_exposes_top_level_panel_metrics(monkeypatch):
    from unittest import mock

    import scripts.prepare_models as pm

    captured = {}
    monkeypatch.setattr(pm, "save_metrics", lambda payload: captured.update(payload))

    panel_row = {
        "Accuracy": 84.6,
        "Recall": 78.2,
        "Sampled Accuracy": 88.1,
        "Like Prediction Accuracy": 86.4,
        "Lift": 136.0,
    }
    results = {
        "temporal_leave_one_last": {
            "best_among_tested": "Hybrid_alpha_0.2",
            "panel_metrics_by_model": {"Hybrid_alpha_0.2": dict(panel_row)},
        }
    }
    with mock.patch("sys.argv", ["prepare_models.py"]):
        args = pm.parse_args()
    pm.save_partial_metrics(results, args, primary="temporal_leave_one_last")
    assert captured["panel_metrics"] == panel_row
    assert set(captured["panel_metrics"]) == {
        "Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift",
    }
    assert 0 <= captured["panel_metrics"]["Accuracy"] <= 100


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


def test_api_artifact_summary_split_not_shadowed(api_client):
    """Regression: the panel-assembly loop once rebound evaluate_split's `name`
    (the split name), so metrics/config recorded split='Reranker'."""
    r = api_client.get("/api/artifact-summary", params={"mode": "raw"})
    assert r.status_code == 200
    split_type = r.json().get("split_type")
    assert split_type in {"temporal_leave_one_last", "userwise_random"}, split_type


def test_api_metrics_exposes_panel_metrics(api_client):
    r = api_client.get("/api/metrics", params={"mode": "raw"})
    assert r.status_code == 200
    body = r.json()
    panel = body.get("panel_metrics")
    if panel is None:
        pytest.skip("artifacts predate panel_metrics; regenerate to enforce")
    assert set(panel) == {
        "Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift",
    }
    for key in ("Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy"):
        value = panel[key]
        assert isinstance(value, (int, float)) and 0 <= value <= 100, (key, value)


def test_api_panel_metrics_clean_payload(api_client):
    """GET /api/panel-metrics returns the clean panel-facing payload:
    dataset summary, 5 headline metrics, per-model rows with display names,
    and NO strict full-catalog metrics."""
    r = api_client.get("/api/panel-metrics", params={"mode": "raw"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"dataset", "headline", "models"}

    dataset = body["dataset"]
    assert dataset["run_mode"] in {"full_dataset", "smoke_test"}, dataset["run_mode"]
    assert dataset["split"] in {"temporal_leave_one_last", "userwise_random"}, dataset["split"]
    assert isinstance(dataset["users"], int) and dataset["users"] > 0
    assert isinstance(dataset["products"], int) and dataset["products"] > 0
    assert isinstance(dataset["interactions"], int) and dataset["interactions"] > 0

    headline = body["headline"]
    if not headline:
        pytest.skip("artifacts predate panel_metrics; regenerate to enforce")
    assert set(headline) == {
        "Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift",
    }
    for key in ("Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy"):
        assert 0 <= headline[key] <= 100, (key, headline[key])
    assert headline["Lift"] >= 1, headline["Lift"]

    models = body["models"]
    assert models, "per-model panel rows missing"
    # display names only — internal artifact names must not leak
    forbidden = {
        "Hybrid_alpha_0.2", "CF_weighted_confidence", "Hybrid_dynamic_alpha_0.2_vision",
    }
    assert not (set(models) & forbidden), set(models) & forbidden
    assert "Hybrid Model" in models
    for row in models.values():
        assert set(row) <= {"Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift"}
        for key in ("Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy"):
            if key in row and row[key] is not None:
                assert 0 <= row[key] <= 100, (key, row[key])
    # strict metric names must not appear anywhere in the clean payload
    text = r.text.lower()
    for strict_key in ("ndcg", "precision@", "hit rate@10", "hr_at_10"):
        assert strict_key not in text, strict_key
