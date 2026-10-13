@echo off
cd /d "%~dp0"
REM Full-dataset training + evaluation (all 183,699 test users).
REM Regenerates artifacts/metrics.json INCLUDING the panel-friendly
REM (0-100) section: panel_metrics, panel_metrics_by_model, panel_detail.
REM Vision uses the local image_cache_full_vision embeddings (no downloads);
REM pass --artifacts-dir to write somewhere other than artifacts/.
.venv\Scripts\python.exe -X utf8 -u scripts\prepare_models.py --full-eval --als-sweep --als-sweep-subset 150000 --als-sweep-users 2000 --eval-users 20000 --batch-size 1024 --use-vision --image-cache-dir image_cache_full_vision --artifacts-dir artifacts >> full_run.log 2>&1
