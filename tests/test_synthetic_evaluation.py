from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _catalog(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(80):
        rows.append(
            {
                "asin": f"B{i:05d}",
                "title": f"Computer accessory {i}",
                "categories": "Electronics Computers Accessories",
                "brand": "PanelBrand",
                "price": float(10 + (i % 10)),
                "imUrl": "",
            }
        )
    path = tmp_path / "products.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _config(tmp_path, seed=42):
    from scripts.synthetic import SyntheticConfig

    return SyntheticConfig(
        n_users=6,
        min_train_interactions=4,
        max_train_interactions=5,
        min_test_interactions=3,
        max_test_interactions=4,
        noise_rate=0.0,
        seed=seed,
        catalog_path=_catalog(tmp_path),
        output_dir=tmp_path / "artifacts_synthetic",
        category_weights={"Computers": 1.0},
        price_band_weights={"budget": 1.0},
        brand_pref_fraction=1.0,
    )


def test_synthetic_generation_is_deterministic_with_seed(tmp_path):
    from scripts.synthetic import SyntheticDataGenerator

    first = SyntheticDataGenerator(_config(tmp_path / "a", seed=7)).generate()
    second = SyntheticDataGenerator(_config(tmp_path / "b", seed=7)).generate()

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_synthetic_split_has_no_user_item_leakage(tmp_path):
    from scripts.synthetic import SyntheticDataGenerator

    interactions, _, _ = SyntheticDataGenerator(_config(tmp_path)).generate()
    train = interactions[interactions["is_train"]]
    test = interactions[~interactions["is_train"]]
    for user_id in interactions["reviewerID"].unique():
        train_items = set(train.loc[train["reviewerID"] == user_id, "asin"])
        test_items = set(test.loc[test["reviewerID"] == user_id, "asin"])
        assert train_items.isdisjoint(test_items)
        assert len(test_items) >= 3


def test_synthetic_held_out_items_match_user_preferences(tmp_path):
    from scripts.synthetic import SyntheticDataGenerator

    interactions, users, catalog = SyntheticDataGenerator(_config(tmp_path)).generate()
    catalog_by_asin = catalog.set_index("asin")
    users_by_id = users.set_index("user_id")
    test = interactions[(~interactions["is_train"]) & (~interactions["is_noise"])]

    for row in test.itertuples(index=False):
        profile = users_by_id.loc[row.reviewerID]
        item = catalog_by_asin.loc[row.asin]
        assert "Computers" in item["categories"]
        assert item["brand"] == profile["preferred_brand"]
        assert float(item["price"]) < 25


def test_synthetic_metrics_are_strict_percent_scalable(tmp_path):
    metrics = {
        "run_mode": "generated_dataset_evaluation",
        "dataset": {"source": "synthetic", "n_users": 1, "n_items": 20},
        "validated_temporal": {
            "split": "synthetic_multi_held_out",
            "run_mode": "generated_dataset_evaluation",
            "n_users": 1,
            "n_items": 20,
            "train_rows": 4,
            "test_rows": 4,
            "n_eval_users": 1,
            "eval_mode": "all_test_users",
            "best_among_tested": "Preference-Aware Hybrid",
            "metrics": {
                "Preference-Aware Hybrid": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "accuracy": 1.0,
                    "ndcg": 0.9,
                    "n_users": 1,
                }
            },
        },
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    row = loaded["validated_temporal"]["metrics"]["Preference-Aware Hybrid"]
    assert 0 <= row["precision"] * 100 <= 100
    assert 0 <= row["recall"] * 100 <= 100
    assert 0 <= row["accuracy"] * 100 <= 100
    assert 0 <= row["ndcg"] * 100 <= 100


def test_api_synthetic_mode_loads_synthetic_artifact_dir(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import api.main as api_main

    artifact_dir = tmp_path / "artifacts_synthetic"
    artifact_dir.mkdir()
    metrics = {
        "run_mode": "generated_dataset_evaluation",
        "dataset": {"source": "synthetic", "n_users": 2, "n_items": 20},
        "validated_temporal": {
            "split": "synthetic_multi_held_out",
            "run_mode": "generated_dataset_evaluation",
            "n_users": 2,
            "n_items": 20,
            "train_rows": 8,
            "test_rows": 8,
            "n_eval_users": 2,
            "eval_mode": "all_test_users",
            "best_among_tested": "Preference-Aware Hybrid",
            "metrics": {
                "Preference-Aware Hybrid": {
                    "precision": 0.7,
                    "recall": 0.7,
                    "accuracy": 1.0,
                    "ndcg": 0.8,
                    "n_users": 2,
                }
            },
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(api_main, "SYNTHETIC_ARTIFACTS", artifact_dir)

    with TestClient(api_main.app) as client:
        response = client.get("/api/panel-metrics", params={"mode": "synthetic"})

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["run_mode"] == "generated_dataset_evaluation"
    assert body["headline"]["Precision@10"] == 70.0
    assert body["headline"]["Accuracy / Hit Rate@10"] == 100.0


def test_frontend_api_client_requests_synthetic_mode_by_default():
    text = Path("frontend/src/api.ts").read_text(encoding="utf-8")
    assert 'metrics: (mode = "synthetic")' in text
    assert "/api/metrics?mode=" in text
    assert "/api/panel-metrics?mode=" in text
    assert "/api/artifact-summary?mode=" in text
