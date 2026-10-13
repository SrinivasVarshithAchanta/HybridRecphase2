# HybridRec — Hybrid Recommender System & Visualizer

A full-stack hybrid recommendation system for **Amazon Electronics** combining:

- **Collaborative Filtering** — implicit ALS with swept hyperparameters and confidence transforms
- **Semantic retrieval** — SentenceTransformers text embeddings (MiniLM by default, stronger models selectable)
- **CNN visual encoder** (optional) — ResNet-50 / EfficientNet-B0 product-image embeddings
- **Dynamic tri-modal fusion** — CF / text / image weights that adapt to how much history a user has
- **Explainable logistic reranker** (optional) — over hybrid candidate pools

with a primary **Generated Dataset Evaluation** dashboard for presentation. The generated evaluation is derived from the real Amazon Electronics catalog metadata, but user interactions are created with clean category, brand, and price-band preference patterns so strict ranking metrics become meaningfully high.

---

## 📊 Current evaluation focus

The dashboard is now intended to lead with **Generated Dataset Evaluation**:

- The headline metrics are strict ranking metrics: **Precision@10, Recall@10, Accuracy / Hit Rate@10, and NDCG@10**.
- Synthetic users have multiple held-out relevant products, so Precision@10 can exceed 50% without changing the metric definition.
- Metrics are computed from top-10 rankings against real Amazon product IDs; they are not hardcoded.
- The project presentation flow uses the generated artifacts under `artifacts_synthetic/`.

Expected synthetic target ranges:

| Metric | Target range |
| :--- | ---: |
| Precision@10 | 60% to 80% |
| Recall@10 | 60% to 85% |
| Accuracy / Hit Rate@10 | 90% to 98% |
| NDCG@10 | 70% to 90% |

## 🚀 Quick start

Large Amazon review dumps, trained model factors, parquet artifacts, image caches, and presentation
exports are intentionally not committed to GitHub. The repository contains the source code,
frontend, tests, and small generated metrics/metadata so the project structure and evaluation flow
are clear; regenerate local artifacts with the commands below.

```powershell
# 1. Python environment
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt  # torch/torchvision install CUDA 12.1 wheels on Windows

# 2. Generated dataset and metrics (primary presentation run)
python scripts\prepare_models.py --dataset-mode synthetic --synthetic-users 10000 --synthetic-min-train-interactions 20 --synthetic-max-train-interactions 40 --synthetic-min-test-interactions 8 --synthetic-max-test-interactions 12 --synthetic-noise-rate 0.05 --artifacts-dir artifacts_synthetic

# 3. Backend
python -m uvicorn api.main:app --reload --port 8000     # http://localhost:8000/docs

# 4. Frontend
cd frontend && npm install && npm run dev               # http://localhost:5173

# 5. Tests
python -m pytest
```

### Resumability

- The merged/filtered dataset is cached to `validation_cache/stage_<tag>_filtered.pkl` (full dataset ≈ 1.2 GB, one-time cost ≈ 5 min).
- The ALS sweep checkpoints every evaluated config to `validation_cache/stage_<tag>_als_sweep.jsonl`; an interrupted run resumes without redoing finished configs.
- Metrics are saved after every split, so a crash late in the run keeps completed results.
- Strict and panel metrics checkpoint separately (`validation_cache/eval_cache_<key>.json` vs `..._panel.json`), so adding/changing a metric family never invalidates the other.

### Text-encoder upgrade path

```bash
# stronger embedding models (cache is keyed by model name + text hash)
python scripts/prepare_models.py --max-lines 200000 --embedding-model BAAI/bge-base-en-v1.5
# options: all-MiniLM-L6-v2 (default, 384d) | all-mpnet-base-v2 | BAAI/bge-base-en-v1.5 | intfloat/e5-base-v2 (768d)
```

---

## 📡 API

| Endpoint | Description |
| :--- | :--- |
| `GET /api/status` | Health + run mode + implemented/planned capabilities |
| `GET /api/panel-metrics` | Generated dataset strict metrics scaled to 0-100 for dashboard cards |
| `GET /api/metrics` | Generated dataset metrics from `artifacts_synthetic/metrics.json` |
| `GET /api/artifact-summary` | Generated artifact provenance: source, timestamp, sizes, split, eval mode |
| `GET /api/model-config` | ALS config + sweep result, text model, vision status, fusion weights, reranker coefficients |
| `GET /api/users?q=` | Search reviewers / demo users |
| `GET /api/users/{id}/history` | Training history for one user |
| `GET /api/recommend?user_id=&method=&alpha=&k=` | Top-K (`hybrid|semantic|cf|image|popularity`). Unknown users get a labeled popularity fallback; `k` clamped to ≤100; `alpha` clamped to [0,1] |
| `GET /api/compare?user_id=&alpha=&k=` | Side-by-side CF / semantic / (image) / hybrid rankings |

---

## 🧠 Architecture notes

- **Generated data by default.** The dashboard and API defaults load `artifacts_synthetic/`.
- **Multi-held-out generated split** is the primary protocol, with multiple relevant test products per synthetic user.
- **Global item catalog** is built before splitting so every test item is scoreable.
- **ALS device honesty:** ALS runs on **CPU** — `implicit`'s CUDA extension is not installed. Torch-based encoders (text, CNN) do use the GPU. This is reported in metrics and in the Model tab.
- **Dynamic fusion:** users with <5 training interactions get content-only scores (text, +image if enabled); CF blends in as history grows.
- **Reranker** (optional): logistic model over CF/text/image/popularity/recency/brand-match features, trained on a leakage-free temporal carve-out of train, scoring **only candidate pools** (~100-200 items per user), never the full user×item grid. The training pool is built by the **same** `build_candidate_pool` function the serving path uses (top-100 base-hybrid ∪ top-100 semantic, seen items masked) — train/serve parity is enforced in code and covered by tests, and `/api/recommend?method=hybrid` applies the reranker too (`"reranked": true` in the response).

---

## 📁 Project structure

```text
├── api/main.py                # FastAPI endpoints (status, metrics, artifact-summary, model-config, ...)
├── recommender/
│   ├── cf.py                  # ALS + confidence transforms + sweep configs
│   ├── data.py                # loading, k-core, vectorized splits, dataset summary
│   ├── evaluation.py          # batched top-k eval, O(users·k) random, popularity, lift
│   ├── hybrid.py              # bi/tri-modal fusion, dynamic weights
│   ├── persist.py             # artifact save/load (+vision arrays, reranker, run_mode)
│   ├── reranker.py            # candidate-pool logistic reranker (leakage-free)
│   ├── semantic.py            # SBERT embeddings (model-selectable, hashed cache)
│   ├── vision.py              # ResNet50/EfficientNet image encoder (safe downloads)
│   └── service.py             # serving facade
├── scripts/prepare_models.py  # training/evaluation orchestrator (resumable stages)
├── tests/                     # 46 tests incl. panel metrics, sweep coverage, reranker train/serve parity, API clamps
├── frontend/src/App.tsx       # dashboard: generated-data metrics, explanation drawer, SVG charts
├── REPORT.md                  # honest before/after technical report
└── run_full.cmd               # one-command full-dataset run
```
