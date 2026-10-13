"""CNN-based visual product encoder.

Downloads product images from ``imUrl`` into a configurable cache, encodes
them with a pretrained torchvision CNN (ResNet-50 by default, EfficientNet-B0
optional) on CUDA when available, and exposes:

- ``build_image_embeddings``  item asin -> embedding
- ``visual_user_profiles``    rating-weighted mean of history item embeddings
- ``cosine_batch``            (n_users, n_items) cosine similarity matrix

Missing or broken images never raise: they are counted and skipped, and the
corresponding items simply have zero vectors (excluded from profile weighting
and from the "covered" count reported in metrics).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

VISION_MODELS = ("resnet50", "efficientnet_b0")

# User-Agent: some image CDNs reject the default urllib agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HybridRec/0.4)"}


def vision_model_dim(model_name: str) -> int:
    return {"resnet50": 2048, "efficientnet_b0": 1280}.get(model_name, 0)


def _build_extractor(model_name: str, device: str):
    import torch
    import torchvision.models as tvm

    weights = {
        "resnet50": (tvm.ResNet50_Weights.IMAGENET1K_V2, tvm.resnet50),
        "efficientnet_b0": (tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, tvm.efficientnet_b0),
    }
    if model_name not in weights:
        raise ValueError(f"Unsupported vision model {model_name}; choose from {VISION_MODELS}")
    weights_cls, builder = weights[model_name]
    net = builder(weights=weights_cls)
    net.eval()
    net.to(device)
    # Everything after the avgpool is discarded -> (batch, embedding_dim) vectors.
    modules = list(net.children())[:-2] if model_name == "resnet50" else list(net.children())[:-2]
    extractor = torch.nn.Sequential(*modules)
    transform = weights_cls.DEFAULT.transforms()
    return extractor, transform


def download_image(
    url: str,
    cache_dir: Path,
    timeout: float = 8.0,
    max_bytes: int = 8 * 1024 * 1024,
) -> Path | None:
    """Download one image into cache_dir; return the local path or None."""
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    ext = ".jpg"
    for candidate in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        if candidate in url.lower():
            ext = candidate
            break
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"{key}{ext}"
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
        if not data or len(data) > max_bytes:
            return None
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path
    except Exception:
        return None


def encode_images_batch(
    image_paths: list[Path | None],
    model_name: str = "resnet50",
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode downloaded images; rows with None produce zero vectors."""
    import torch
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = vision_model_dim(model_name)
    out = np.zeros((len(image_paths), dim), dtype=np.float32)
    extractor, transform = _build_extractor(model_name, device)

    valid_rows = [i for i, p in enumerate(image_paths) if p is not None]
    with torch.no_grad():
        for start in range(0, len(valid_rows), batch_size):
            chunk_rows = valid_rows[start : start + batch_size]
            batch, kept = [], []
            for row in chunk_rows:
                try:
                    img = Image.open(image_paths[row]).convert("RGB")
                    batch.append(transform(img))
                    kept.append(row)
                except Exception:
                    continue  # corrupt/missing file: leave zero vector
            if not batch:
                continue
            tensor = torch.stack(batch).to(device)
            feats = extractor(tensor)
            if feats.dim() > 2:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            out[np.asarray(kept)] = feats.detach().cpu().numpy().astype(np.float32)
    return out


def build_image_embeddings(
    catalog: pd.DataFrame,
    model_name: str = "resnet50",
    cache_dir: Path | None = None,
    limit: int | None = None,
    timeout: float = 8.0,
    batch_size: int = 64,
    use_cache_file: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Download + encode catalog images. Returns (asin -> vec, report).

    report includes counts of attempted, embedded, missing URLs and failures.
    """
    import torch

    if cache_dir is None:
        cache_dir = Path.cwd() / "image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    catalog = catalog.drop_duplicates("asin").reset_index(drop=True)
    if limit is not None:
        catalog = catalog.head(int(limit))

    urls = catalog["imUrl"].tolist() if "imUrl" in catalog.columns else [""] * len(catalog)
    asins = catalog["asin"].tolist()

    # Cache key covers the FULL ordered ASIN list + model name + per-URL list,
    # so any catalog change (order, membership, imUrl) invalidates correctly.
    url_key = "|".join(str(u) for u in urls)
    digest = hashlib.md5(
        f"{model_name}|{len(asins)}|{'|'.join(asins)}|{url_key}".encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_dir / f"image_emb_{model_name}_{digest}.npy"
    if use_cache_file and cache_file.exists():
        arr = np.load(cache_file)
        if arr.shape[0] == len(asins):
            vecs = {asin: arr[i] for i, asin in enumerate(asins)}
            report = {
                "model": model_name,
                "attempted": int(len(asins)),
                "embedded": int((arr.any(axis=1)).sum()),
                "missing_or_failed": int((~arr.any(axis=1)).sum()),
                "device": device,
                "cache_file": str(cache_file),
            }
            return vecs, report

    missing = failed = 0
    paths: list[Path | None] = []
    for url in tqdm(urls, desc="Downloading product images"):
        local = download_image(url, cache_dir, timeout=timeout)
        if local is None:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                failed += 1
            else:
                missing += 1
            paths.append(None)
        else:
            paths.append(local)

    arr = encode_images_batch(paths, model_name=model_name, device=device, batch_size=batch_size)
    if use_cache_file:
        np.save(cache_file, arr)

    vecs = {asin: arr[i] for i, asin in enumerate(asins)}
    report = {
        "model": model_name,
        "attempted": int(len(asins)),
        "embedded": int((arr.any(axis=1)).sum()),
        "missing_or_failed": int(missing + failed),
        "missing_url": int(missing),
        "failed_download": int(failed),
        "device": device,
        "cache_file": str(cache_file),
    }
    return vecs, report


def visual_user_profiles(
    train_df: pd.DataFrame,
    item_vecs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rating-weighted mean image vector per user (zero-vector users dropped)."""
    user_vecs: dict[str, np.ndarray] = {}
    for user, group in train_df.groupby("reviewerID"):
        vecs = []
        weights = []
        for rec in group.itertuples(index=False):
            vec = item_vecs.get(getattr(rec, "asin"))
            if vec is None or not np.asarray(vec).any():
                continue
            vecs.append(np.asarray(vec, dtype=np.float32))
            weights.append(max(float(getattr(rec, "overall", 3.0)), 0.5))
        if vecs:
            user_vecs[user] = np.average(np.vstack(vecs), axis=0, weights=np.asarray(weights))
    return user_vecs


def cosine_batch(user_mat: np.ndarray, item_mat: np.ndarray) -> np.ndarray:
    """(n_users, n_items) cosine similarity; zero rows produce zero scores."""
    user_mat = np.asarray(user_mat, dtype=np.float32)
    item_mat = np.asarray(item_mat, dtype=np.float32)
    u_norm = np.linalg.norm(user_mat, axis=1, keepdims=True)
    i_norm = np.linalg.norm(item_mat, axis=1, keepdims=True)
    u_unit = user_mat / np.maximum(u_norm, 1e-12)
    i_unit = item_mat / np.maximum(i_norm, 1e-12)
    return (u_unit @ i_unit.T).astype(np.float32)
