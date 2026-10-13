from __future__ import annotations

import numpy as np
import pandas as pd

from .cf import cf_scores
from .hybrid import (
    contribution_explanation,
    hybrid_scores,
    topk_from_scores,
    tri_modal_dynamic_weights,
    tri_modal_scores,
    zscore,
)
from .persist import load_bundle, load_metrics
from .semantic import cosine_scores


class UnknownUserError(Exception):
    def __init__(self, user_id: str):
        super().__init__(user_id)
        self.user_id = user_id


class RecommenderService:
    def __init__(self, bundle=None):
        self.bundle = bundle or load_bundle()
        self.metrics = load_metrics()
        self.user_index = {u: i for i, u in enumerate(self.bundle["user_ids"])}
        self.item_index = {a: i for i, a in enumerate(self.bundle["item_ids"])}
        self.products = self.bundle["products"].set_index("asin")
        self.primary_item_emb = (
            self.bundle["item_emb_v2"]
            if self.bundle["config"].get("semantic_primary", "v2") == "v2"
            else self.bundle["item_emb_v1"]
        )
        self.primary_user_emb = (
            self.bundle["user_emb_v2"]
            if self.bundle["config"].get("semantic_primary", "v2") == "v2"
            else self.bundle["user_emb_v1"]
        )
        # Optional vision modality: absent arrays or empty matrices mean "disabled".
        self.item_img_emb = self.bundle.get("item_img_emb")
        self.user_img_emb = self.bundle.get("user_img_emb")
        if self.item_img_emb is None or self.user_img_emb is None:
            self.vision_enabled = False
        else:
            try:
                self.vision_enabled = bool(np.asarray(self.item_img_emb).size)
            except Exception:
                self.vision_enabled = False
        self.reranker = self.bundle.get("reranker")
        self.reranker_item_features = self.bundle.get("reranker_item_features")
        self.reranker_enabled = bool(
            self.reranker
            and getattr(self.reranker, "enabled", False)
            and self.reranker_item_features is not None
        )
        self.history_counts = self.bundle.get("history_counts") or {
            u: len(g) for u, g in self.bundle["histories"].groupby("reviewerID")
        }
        hybrid_cfg = self.bundle["config"].get("hybrid", {})
        split = self.bundle["config"].get("split", self.bundle["config"].get("split_type", ""))
        self.default_alpha = float(
            hybrid_cfg.get("default_alpha", 0.2 if split == "temporal_leave_one_last" else 0.0)
        )
        self.alpha_cf = float(hybrid_cfg.get("alpha_cf", self.default_alpha))
        self.beta_text = float(hybrid_cfg.get("beta_text", 1.0 - self.default_alpha))
        self.gamma_image = float(hybrid_cfg.get("gamma_image", 0.0))
        self.cool_user_threshold = int(hybrid_cfg.get("cool_user_threshold", 5))
        self.cool_user_alpha = float(hybrid_cfg.get("cool_user_alpha", 0.0))

    def known_user(self, user_id: str) -> bool:
        return user_id in self.user_index

    def vision_status(self) -> dict:
        if not self.vision_enabled:
            return {"enabled": False}
        img = np.asarray(self.item_img_emb)
        covered = int((np.abs(img).sum(axis=1) > 0).sum()) if img.ndim == 2 else 0
        cfg = self.bundle["config"].get("vision", {})
        return {
            "enabled": True,
            "model": cfg.get("model", "resnet50"),
            "n_image_embeddings": covered,
            "n_catalog_items": int(img.shape[0]) if img.ndim == 2 else 0,
            "missing_image_count": int(img.shape[0] - covered) if img.ndim == 2 else 0,
        }

    def user_history(self, user_id: str):
        hist = self.bundle["histories"]
        rows = hist.loc[hist["reviewerID"] == user_id]
        items = []
        for rec in rows.itertuples(index=False):
            items.append(
                {
                    "asin": rec.asin,
                    "title": rec.title,
                    "overall": float(rec.overall),
                    "image_url": rec.imUrl if pd_notnull(rec.imUrl) else None,
                }
            )
        avg = float(np.mean([x["overall"] for x in items])) if items else None
        return {
            "user_id": user_id,
            "n_interactions": len(items),
            "average_rating": avg,
            "items": items,
        }

    def _base_scores(self, user_id: str):
        """Raw per-user score vectors for CF / text / image (any may be None)."""
        uidx = self.user_index.get(user_id)
        if uidx is None:
            raise UnknownUserError(user_id)
        cf = np.asarray(
            cf_scores(uidx, self.bundle["user_factors"], self.bundle["item_factors"]),
            dtype=np.float64,
        )
        sem = np.asarray(cosine_scores(self.primary_user_emb[uidx], self.primary_item_emb), dtype=np.float64)
        img = None
        if self.vision_enabled:
            u_vec = np.asarray(self.user_img_emb)[uidx]
            if np.abs(u_vec).sum() > 0:
                img = np.asarray(cosine_scores(u_vec, np.asarray(self.item_img_emb)), dtype=np.float64)
        return cf, sem, img

    def effective_weights(self, user_id: str, alpha: float | None = None):
        """(alpha_cf, beta_text, gamma_image, used_dynamic) for a user."""
        requested = self.default_alpha if alpha is None else float(alpha)
        requested = min(1.0, max(0.0, requested))
        count = self.history_counts.get(user_id, 0)
        if count < self.cool_user_threshold:
            # Cool user: semantic-only fallback preserved from the bi-modal era,
            # extended to let image content in when vision is enabled.
            if self.vision_enabled:
                return 0.0, 0.7, 0.3, True
            return 0.0, 1.0, 0.0, True
        a = self.alpha_cf if alpha is None else requested
        gamma = self.gamma_image if self.vision_enabled else 0.0
        beta = max(1e-6, self.beta_text if self.vision_enabled else 1.0 - a)
        if not self.vision_enabled:
            return a, beta, 0.0, False
        total = a + beta + gamma
        return a / total, beta / total, gamma / total, False

    def _ranking_scores(self, user_id: str, method: str, alpha: float | None):
        cf, sem, img = self._base_scores(user_id)
        a_cf, b_text, g_img, dynamic = self.effective_weights(user_id, alpha)
        if method == "cf":
            scores, used = cf, (a_cf, b_text, g_img)
        elif method == "semantic":
            scores, used = sem, (a_cf, b_text, g_img)
        elif method == "image":
            if img is None:
                raise ValueError("Vision modality is not enabled for this artifact.")
            scores, used = img, (a_cf, b_text, g_img)
        elif method == "popularity":
            scores = np.asarray(self.bundle["pop_counts"], dtype=np.float64)
            used = (a_cf, b_text, g_img)
        else:  # hybrid
            if img is None:
                scores = hybrid_scores(cf, sem, a_cf, normalize=True)
                g_img = 0.0
            else:
                scores = tri_modal_scores(cf, sem, img, a_cf, b_text, g_img, normalize=True)
            used = (a_cf, b_text, g_img)
        return cf, sem, img, scores, used, dynamic

    def recommend(
        self,
        user_id: str,
        method: str = "hybrid",
        alpha: float | None = None,
        k: int = 10,
    ):
        cf, sem, img, ranking_scores, (a_cf, b_text, g_img), dynamic = self._ranking_scores(
            user_id, method.lower(), alpha
        )
        seen = self.bundle["seen_by_user"].get(user_id, set())
        asins, idxs = topk_from_scores(ranking_scores, self.bundle["item_ids"], seen, k=k)
        reranked = False
        if method.lower() == "hybrid" and self.reranker_enabled:
            # Serving-side rerank over the candidate pool, mirroring evaluation:
            # seen items are masked before pool selection and the pool is built
            # by the SAME function the reranker was trained with.
            from .reranker import build_candidate_pool

            cf_z_rr = zscore(cf)
            sem_z_rr = zscore(sem)
            base = 0.5 * cf_z_rr + 0.5 * sem_z_rr
            if seen:
                base = base.copy()
                base[[self.item_index[a] for a in seen if a in self.item_index]] = -np.inf
            pool = build_candidate_pool(base, sem_z_rr, 100)
            if len(seen):
                seen_idx = np.array([self.item_index[a] for a in seen if a in self.item_index], dtype=int)
                pool = pool[~np.isin(pool, seen_idx)] if len(seen_idx) else pool
            rr_scores = self.reranker.rank_pool(
                cf, sem, img, pool, self.reranker_item_features,
                sorted(seen),  # seen set == the user's train history
                self.item_index,
            )
            order = pool[np.argsort(-rr_scores)][:k]
            rr_asins = [self.bundle["item_ids"][j] for j in order]
            rr_set = set(rr_asins)
            merged = rr_asins + [a for a in asins if a not in rr_set]
            asins, idxs = merged[:k], [self.item_index[a] for a in merged[:k]]
            reranked = True
        cf_z = zscore(cf)
        sem_z = zscore(sem)
        img_z = zscore(img) if img is not None else None
        items = []
        for rank, (asin, idx) in enumerate(zip(asins, idxs), start=1):
            meta = self.products.loc[asin] if asin in self.products.index else None
            title = None if meta is None else meta["title"]
            image_url = None if meta is None else (meta["imUrl"] or None)
            if method == "hybrid":
                explanation = contribution_explanation(
                    a_cf, float(cf_z[idx]), float(sem_z[idx]),
                    None if img_z is None else float(img_z[idx]),
                )
            elif method == "cf":
                explanation = "Ranked purely by collaborative filtering (ALS) score."
            elif method == "semantic":
                explanation = "Ranked purely by text-embedding similarity to the user's semantic profile."
            elif method == "image":
                explanation = "Ranked purely by CNN image-embedding similarity to the user's visual profile."
            else:
                explanation = "Ranked by catalog popularity."
            items.append(
                {
                    "rank": rank,
                    "asin": asin,
                    "title": None if not isinstance(title, str) else title,
                    "image_url": None if not isinstance(image_url, str) else image_url,
                    "cf_score": float(cf[idx]),
                    "semantic_score": float(sem[idx]),
                    "image_score": None if img is None else float(img[idx]),
                    "hybrid_score": float(ranking_scores[idx]),
                    "explanation": explanation,
                }
            )
        cf_contrib = float(abs(zscore(ranking_scores)[idxs].mean()) if len(idxs) else 0.0)
        contributions = self._contributions(cf, sem, img, idxs, a_cf, b_text, g_img)
        return {
            "user_id": user_id,
            "method": method,
            "alpha": a_cf,
            "requested_alpha": self.default_alpha if alpha is None else float(alpha),
            "k": k,
            "n_candidates": int(len(self.bundle["item_ids"]) - len(seen)),
            "n_catalog": int(len(self.bundle["item_ids"])),
            "train_interactions": int(self.history_counts.get(user_id, 0)),
            "reranked": reranked,
            "reranker_enabled": self.reranker_enabled,
            "dynamic_fusion": dynamic,
            "semantic_only_fallback": bool(dynamic and self.history_counts.get(user_id, 0) < self.cool_user_threshold),
            "weights": {"alpha_cf": a_cf, "beta_text": b_text, "gamma_image": g_img},
            "contributions": contributions,
            "vision_enabled": self.vision_enabled,
            "items": items,
        }

    def _contributions(self, cf, sem, img, idxs, a_cf, b_text, g_img):
        """Mean weighted |z| contribution of each modality over the returned items."""
        if len(idxs) == 0:
            return {"cf": 0.0, "text": 0.0, "image": 0.0}
        cf_z = np.abs(zscore(cf))[idxs]
        sem_z = np.abs(zscore(sem))[idxs]
        img_z = np.abs(zscore(img))[idxs] if img is not None else np.zeros(len(idxs))
        cf_c, text_c, img_c = a_cf * cf_z.mean(), b_text * sem_z.mean(), g_img * img_z.mean()
        total = cf_c + text_c + img_c
        if total <= 1e-12:
            return {"cf": 0.0, "text": 0.0, "image": 0.0}
        return {
            "cf": float(cf_c / total),
            "text": float(text_c / total),
            "image": float(img_c / total),
        }

    def compare(self, user_id: str, k: int = 10, alpha: float | None = None):
        cf, sem, img = self._base_scores(user_id)
        a_cf, b_text, g_img, dynamic = self.effective_weights(user_id, alpha)
        if img is None:
            hyb = hybrid_scores(cf, sem, a_cf, normalize=True)
            methods = [("hybrid", hyb), ("semantic", sem), ("cf", cf)]
        else:
            hyb = tri_modal_scores(cf, sem, img, a_cf, b_text, g_img, normalize=True)
            methods = [("hybrid", hyb), ("semantic", sem), ("cf", cf), ("image", img)]
        seen = self.bundle["seen_by_user"].get(user_id, set())

        full_ranks = {}
        top_k_asins = {}
        for method, scores in methods:
            asins, _ = topk_from_scores(scores, self.bundle["item_ids"], seen, k=len(self.bundle["item_ids"]))
            full_ranks[method] = {asin: rank for rank, asin in enumerate(asins, start=1)}
            top_k_asins[method] = asins[:k]

        union_asins = []
        seen_union = set()
        for m, _ in methods:
            for asin in top_k_asins[m]:
                if asin not in seen_union:
                    seen_union.add(asin)
                    union_asins.append(asin)

        products = []
        for asin in union_asins:
            meta = self.products.loc[asin] if asin in self.products.index else None
            title = None if meta is None else meta["title"]
            entry = {
                "asin": asin,
                "title": None if not isinstance(title, str) else title,
                "image_url": None if meta is None else (meta["imUrl"] or None),
                "cf_rank": full_ranks["cf"].get(asin),
                "semantic_rank": full_ranks["semantic"].get(asin),
                "hybrid_rank": full_ranks["hybrid"].get(asin),
            }
            if img is not None:
                entry["image_rank"] = full_ranks["image"].get(asin)
            products.append(entry)

        sets = {m: set(top_k_asins[m]) for m, _ in methods}
        all_sets = [sets[m] for m, _ in methods]
        overlap_all = set.intersection(*all_sets) if all_sets else set()
        unique = {
            m: sorted(sets[m] - set().union(*(s for n, s in sets.items() if n != m)))
            for m, _ in methods
        }
        out = {
            "user_id": user_id,
            "k": k,
            "alpha": a_cf,
            "requested_alpha": self.default_alpha if alpha is None else float(alpha),
            "dynamic_fusion": dynamic,
            "overlap_all": sorted(overlap_all),
            "unique": unique,
            "products": products,
        }
        # Legacy key kept for the existing frontend/tests.
        out["overlap_all_three"] = sorted(sets["cf"] & sets["semantic"] & sets["hybrid"])
        return out

    def popularity_fallback(self, k: int = 10):
        scores = np.asarray(self.bundle["pop_counts"], dtype=np.float64)
        asins, idxs = topk_from_scores(scores, self.bundle["item_ids"], set(), k=k)
        items = []
        for rank, asin in enumerate(asins, start=1):
            meta = self.products.loc[asin] if asin in self.products.index else None
            items.append(
                {
                    "rank": rank,
                    "asin": asin,
                    "title": None if meta is None else meta["title"],
                    "image_url": None if meta is None else (meta["imUrl"] or None),
                    "cf_score": None,
                    "semantic_score": None,
                    "image_score": None,
                    "hybrid_score": None,
                    "explanation": "Catalog popularity fallback; not personalized.",
                }
            )
        return items

    def semantic_fallback(self, user_id: str, k: int = 10):
        """Best-effort semantic-only recs (e.g. reranker path disabled)."""
        cf, sem, img, _, _, _ = self._base_scores(user_id)
        seen = self.bundle["seen_by_user"].get(user_id, set())
        asins, idxs = topk_from_scores(sem, self.bundle["item_ids"], seen, k=k)
        items = []
        for rank, (asin, idx) in enumerate(zip(asins, idxs), start=1):
            meta = self.products.loc[asin] if asin in self.products.index else None
            title = None if meta is None else meta["title"]
            items.append(
                {
                    "rank": rank,
                    "asin": asin,
                    "title": None if not isinstance(title, str) else title,
                    "image_url": None if meta is None else (meta["imUrl"] or None),
                    "cf_score": float(cf[idx]),
                    "semantic_score": float(sem[idx]),
                    "image_score": None if img is None else float(img[idx]),
                    "hybrid_score": float(sem[idx]),
                    "explanation": "Semantic-only recommendation (fallback).",
                }
            )
        return items


def pd_notnull(value):
    if value is None:
        return False
    try:
        return value == value and str(value) != "nan"
    except Exception:
        return False
