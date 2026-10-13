from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / "validation_cache"
REVIEWS_PATH = ROOT / "Electronics_5.json"
META_PATH = ROOT / "meta_Electronics.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
ARTIFACTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)


def set_artifacts_dir(path) -> Path:
    """Redirect artifact output to a custom directory (e.g. --artifacts-dir smoke_out).

    Lets smoke/dev runs keep their results OUT of the production artifacts/ dir,
    so a smoke run can never clobber full-dataset serving artifacts. Call once,
    right after argument parsing, before any save/load.
    """
    global ARTIFACTS, METRICS_PATH
    ARTIFACTS = Path(path)
    METRICS_PATH = ARTIFACTS / "metrics.json"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
