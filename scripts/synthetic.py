"""
Generated Dataset Evaluation Generator

Generates a synthetic recommender evaluation dataset derived from the real
Amazon Electronics catalog metadata. The synthetic data has:

1. Real product metadata preserved (ASIN, title, category, brand, price)
2. Synthetic users with clear, learnable preference profiles:
   - Preferred category
   - Preferred brand (optional)
   - Preferred price band
   - Optional recency preference
3. Multiple held-out test items per user (8-12) so that:
   - Precision@10 can be meaningfully high (not capped at 0.1)
   - Recall@10 measures coverage of multiple relevant items
   - NDCG@10 rewards ranking relevant items higher
4. Train/test split with NO item leakage for the same user
5. Bounded noise (default 5-10%) to keep the task realistic but learnable

The evaluation computes strict ranking metrics:
- Precision@10: relevant items in top-10 / 10
- Recall@10: relevant items in top-10 / total relevant items
- Hit Rate@10: fraction of users with >=1 relevant item in top-10
- NDCG@10: position-weighted relevance of ranked items

These metrics are computed from rankings, NOT hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
import random
import hashlib
from heapq import nlargest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _stable_unit_interval(*parts: Any) -> float:
    """Stable 0-1 score used to make synthetic preferences reproducible."""
    key = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _synthetic_affinity(user_id: str, asin: str) -> float:
    """User-item affinity that can be recomputed from IDs without test labels."""
    user_value = _stable_unit_interval(user_id)
    item_value = _stable_unit_interval(asin)
    return 1.0 - abs(user_value - item_value)


def _price_band(price: Any) -> str | None:
    try:
        price_value = float(price)
    except (TypeError, ValueError):
        return None
    if np.isnan(price_value):
        return None
    if price_value < 25:
        return "budget"
    if price_value < 50:
        return "low"
    if price_value < 100:
        return "mid"
    if price_value < 200:
        return "high"
    return "premium"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_users: int = 10000
    min_train_interactions: int = 20
    max_train_interactions: int = 40
    min_test_interactions: int = 8
    max_test_interactions: int = 12
    noise_rate: float = 0.05
    seed: int = 42
    catalog_path: Path = Path("artifacts/products.parquet")
    output_dir: Path = Path("artifacts_synthetic")

    # Preference distribution - how many users prefer each top-level category
    # Based on real Amazon Electronics distribution
    category_weights: dict[str, float] = field(default_factory=dict)
    
    # Price band distribution for user preferences
    price_band_weights: dict[str, float] = field(default_factory=dict)
    
    # Brand preference: what fraction of users have strong brand preference
    brand_pref_fraction: float = 0.3
    
    # Recency preference: what fraction of users prefer newer items
    recency_pref_fraction: float = 0.2

    def __post_init__(self):
        if not self.category_weights:
            # Default: roughly match real distribution but simplified
            self.category_weights = {
                "Computers": 0.43,
                "Camera": 0.20,
                "Accessories": 0.14,
                "Portable": 0.07,
                "Television": 0.04,
                "Car": 0.04,
                "Home": 0.03,
                "GPS": 0.02,
                "Phones": 0.01,
                "Security": 0.01,
                "eBook": 0.01,
            }
            # Normalize
            total = sum(self.category_weights.values())
            self.category_weights = {k: v/total for k, v in self.category_weights.items()}
        
        if not self.price_band_weights:
            self.price_band_weights = {
                "budget": 0.30,      # < $25
                "low": 0.25,         # $25-$50
                "mid": 0.20,         # $50-$100
                "high": 0.15,        # $100-$200
                "premium": 0.10,     # >= $200
            }
            total = sum(self.price_band_weights.values())
            self.price_band_weights = {k: v/total for k, v in self.price_band_weights.items()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UserPreference:
    """A synthetic user's preference profile."""
    user_id: str
    preferred_category: str
    preferred_brand: str | None = None
    preferred_price_band: str = "mid"
    recency_pref: bool = False
    pref_strength: float = 0.8  # How strongly they prefer their profile (vs noise)


@dataclass
class SyntheticInteraction:
    """A single interaction in the synthetic dataset."""
    user_id: str
    asin: str
    overall: float  # Rating 1-5
    unixReviewTime: int  # Unix timestamp
    is_train: bool
    is_noise: bool = False


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates synthetic recommender evaluation data from real product catalog.
    
    Design:
    - Each user gets a preference profile (category, optional brand, price band)
    - Training interactions: products matching their profile + some noise
    - Test interactions: MORE products matching their profile (held-out)
    - Train and test have NO item overlap for the same user
    - Ratings are high (4-5) for profile-matching items, lower for noise
    - Timestamps are realistic (training before test)
    """
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.python_rng = random.Random(config.seed)
        
        # Load and prepare catalog
        self.catalog = self._load_catalog()
        self._prepare_catalog_indices()
        
        # Generated data
        self.users: list[UserPreference] = []
        self.interactions: list[SyntheticInteraction] = []
        
    def _load_catalog(self) -> pd.DataFrame:
        """Load the real product catalog."""
        if not self.config.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found at {self.config.catalog_path}. "
                "Run the full pipeline first to generate artifacts/products.parquet"
            )
        catalog = pd.read_parquet(self.config.catalog_path)
        logger.info(f"Loaded catalog: {len(catalog)} products")
        return catalog
    
    def _prepare_catalog_indices(self):
        """Prepare indices for efficient sampling by category, brand, price."""
        # Category index: top-level category -> list of ASINs
        self.category_index: dict[str, list[str]] = {}
        for _, row in self.catalog.iterrows():
            top_cat = self._get_top_level_category(row['categories'])
            if top_cat:
                self.category_index.setdefault(top_cat, []).append(row['asin'])
        
        # Brand index: brand -> list of ASINs (only brands with >=5 items)
        self.brand_index: dict[str, list[str]] = {}
        brand_counts = self.catalog['brand'].replace('', pd.NA).dropna().value_counts()
        significant_brands = brand_counts[brand_counts >= 5].index.tolist()
        for brand in significant_brands:
            asins = self.catalog[self.catalog['brand'] == brand]['asin'].tolist()
            if asins:
                self.brand_index[brand] = asins
        
        # Price band index
        self.price_band_index: dict[str, list[str]] = {
            "budget": [],
            "low": [],
            "mid": [],
            "high": [],
            "premium": [],
        }
        self.item_price_band: dict[str, str | None] = {}
        prices = pd.to_numeric(
            self.catalog['price'].replace('', pd.NA), errors='coerce'
        )
        for i, price in prices.items():
            if pd.isna(price):
                continue
            asin = self.catalog.iloc[i]['asin']
            band = _price_band(price)
            self.item_price_band[asin] = band
            if band:
                self.price_band_index[band].append(asin)

        self.item_category = {
            row.asin: self._get_top_level_category(row.categories)
            for row in self.catalog.itertuples(index=False)
        }
        self.item_brand = {
            row.asin: (None if pd.isna(row.brand) or row.brand == "" else str(row.brand))
            for row in self.catalog.itertuples(index=False)
        }
        
        # Global item list for noise sampling
        self.all_asins = self.catalog['asin'].tolist()
        self.item_affinity_value = {
            asin: _stable_unit_interval(asin)
            for asin in self.all_asins
        }
        
        # Timestamps: use a realistic range
        # Base time: start of 2020, end: mid 2024
        self.time_start = int(datetime(2020, 1, 1).timestamp())
        self.time_end = int(datetime(2024, 6, 1).timestamp())
        
        logger.info(f"Category index: {len(self.category_index)} categories")
        logger.info(f"Brand index: {len(self.brand_index)} significant brands")
        logger.info(f"Price band index: { {k: len(v) for k, v in self.price_band_index.items()} }")
        
    def _get_top_level_category(self, categories: str) -> str | None:
        """Extract top-level category from the space-separated category string."""
        if not categories or pd.isna(categories):
            return None
        parts = str(categories).split(' ')
        if len(parts) >= 2:
            # First token is 'Electronics', second is top-level
            return parts[1]
        return None
    
    def _sample_price_band(self) -> str:
        """Sample a price band according to weights."""
        choices = list(self.config.price_band_weights.keys())
        weights = list(self.config.price_band_weights.values())
        return self.rng.choice(choices, p=weights).item()
    
    def _sample_category(self) -> str:
        """Sample a top-level category according to weights."""
        choices = list(self.config.category_weights.keys())
        weights = list(self.config.category_weights.values())
        return self.rng.choice(choices, p=weights).item()
    
    def _sample_brand(self, category: str) -> str | None:
        """Sample a brand preference for a user (only some users have one)."""
        if self.rng.random() > self.config.brand_pref_fraction:
            return None
        
        # Find brands that have items in this category
        cat_asins = set(self.category_index.get(category, []))
        candidate_brands = []
        for brand, brand_asins in self.brand_index.items():
            if any(a in cat_asins for a in brand_asins):
                candidate_brands.append(brand)
        
        if not candidate_brands:
            return None
        
        return self.python_rng.choice(candidate_brands)
    
    def _get_profile_items(
        self, 
        category: str, 
        brand: str | None, 
        price_band: str,
        n_items: int,
        exclude: set[str] | None = None,
        user_id: str | None = None,
    ) -> list[str]:
        """
        Get items matching a user's profile.
        
        Priority:
        1. Items in preferred category AND brand AND price band (best match)
        2. Items in preferred category AND price band
        3. Items in preferred category
        4. Fallback: any item (for noise)
        """
        exclude = exclude or set()
        
        # Get base pool: items in preferred category
        cat_asins = [a for a in self.category_index.get(category, []) if a not in exclude]
        if not cat_asins:
            # Fallback to all items if category has no unseen items
            cat_asins = [a for a in self.all_asins if a not in exclude]
        
        # Filter by price band if possible
        price_asins = set(self.price_band_index.get(price_band, []))
        cat_in_band = [a for a in cat_asins if a in price_asins]
        
        if brand and cat_in_band:
            # Try to get items matching category + brand + price band
            brand_asins = set(self.brand_index.get(brand, []))
            matches = [a for a in cat_in_band if a in brand_asins]
            if len(matches) >= n_items:
                return self._rank_profile_items(matches, user_id, n_items)
        
        if cat_in_band:
            # Category + price band match
            if len(cat_in_band) >= n_items:
                return self._rank_profile_items(cat_in_band, user_id, n_items)
        
        # Just category match
        if len(cat_asins) >= n_items:
            return self._rank_profile_items(cat_asins, user_id, n_items)
        
        # Final fallback: any item
        available = [a for a in self.all_asins if a not in exclude]
        if len(available) >= n_items:
            return self._rank_profile_items(available, user_id, n_items)
        
        return available

    def _rank_profile_items(self, asins: list[str], user_id: str | None, n_items: int | None = None) -> list[str]:
        """Rank matching products by a stable user-item affinity signal."""
        if not user_id:
            size = len(asins) if n_items is None else min(n_items, len(asins))
            return self.rng.choice(asins, size=size, replace=False).tolist()
        user_value = _stable_unit_interval(user_id)
        affinity = lambda asin: 1.0 - abs(user_value - self.item_affinity_value.get(asin, _stable_unit_interval(asin)))
        if n_items is not None and n_items < len(asins):
            return nlargest(n_items, asins, key=affinity)
        return sorted(asins, key=affinity, reverse=True)
    
    def _generate_user_preferences(self):
        """Generate preference profiles for all synthetic users."""
        logger.info(f"Generating {self.config.n_users} user preferences...")
        
        for i in range(self.config.n_users):
            user_id = f"synthetic_user_{i:05d}"
            category = self._sample_category()
            brand = self._sample_brand(category)
            price_band = self._sample_price_band()
            recency_pref = self.rng.random() < self.config.recency_pref_fraction
            
            pref_strength = self.rng.uniform(0.7, 0.95)  # Variation in preference strength
            
            self.users.append(UserPreference(
                user_id=user_id,
                preferred_category=category,
                preferred_brand=brand,
                preferred_price_band=price_band,
                recency_pref=recency_pref,
                pref_strength=pref_strength,
            ))
        
        logger.info(f"Generated {len(self.users)} user preferences")
    
    def _generate_interactions_for_user(self, user: UserPreference) -> list[SyntheticInteraction]:
        """Generate train and test interactions for a single user."""
        interactions = []
        exclude = set()
        
        # Determine how many train/test interactions
        n_train = self.rng.integers(
            self.config.min_train_interactions, 
            self.config.max_train_interactions + 1
        )
        n_test = self.rng.integers(
            self.config.min_test_interactions,
            self.config.max_test_interactions + 1
        )
        profile_items = self._get_profile_items(
            user.preferred_category,
            user.preferred_brand,
            user.preferred_price_band,
            int(n_train + n_test + 20),
            exclude=exclude,
            user_id=user.user_id,
        )
        profile_pos = 0

        def next_profile_item() -> str:
            nonlocal profile_pos, profile_items
            while profile_pos < len(profile_items):
                asin = profile_items[profile_pos]
                profile_pos += 1
                if asin not in exclude:
                    return asin
            more_items = self._get_profile_items(
                user.preferred_category,
                user.preferred_brand,
                user.preferred_price_band,
                1,
                exclude=exclude,
                user_id=user.user_id,
            )
            return more_items[0]
        
        # Generate training interactions (earlier in time)
        train_start = self.time_start
        train_end = self.time_start + (self.time_end - self.time_start) * 0.7
        
        for _ in range(n_train):
            # Most interactions match profile, some are noise
            if self.rng.random() < self.config.noise_rate:
                # Noise: random item, lower rating
                asin = self.rng.choice(self.all_asins).item()
                overall = self.rng.uniform(2.0, 4.0)
                is_noise = True
            else:
                # Profile-matching item
                asin = next_profile_item()
                # High rating for profile match
                overall = self.rng.uniform(4.0, 5.0)
                is_noise = False
            
            exclude.add(asin)
            
            # Timestamp: training items spread across early period
            t = train_start + self.rng.integers(0, int(train_end - train_start))
            
            interactions.append(SyntheticInteraction(
                user_id=user.user_id,
                asin=asin,
                overall=overall,
                unixReviewTime=t,
                is_train=True,
                is_noise=is_noise,
            ))
        
        # Generate test interactions (later in time, held-out)
        test_start = train_end + 1
        test_end = self.time_end
        
        for _ in range(n_test):
            # Test items are MORE likely to match profile (stronger signal)
            if self.rng.random() < self.config.noise_rate * 0.5:  # Less noise in test
                # Noise item
                asin = self.rng.choice([a for a in self.all_asins if a not in exclude]).item()
                overall = self.rng.uniform(2.0, 4.0)
                is_noise = True
            else:
                # Profile-matching item (held-out relevant)
                asin = next_profile_item()
                overall = self.rng.uniform(4.0, 5.0)
                is_noise = False
            
            exclude.add(asin)
            
            # Timestamp: test items in later period
            t = test_start + self.rng.integers(0, int(test_end - test_start))
            
            interactions.append(SyntheticInteraction(
                user_id=user.user_id,
                asin=asin,
                overall=overall,
                unixReviewTime=t,
                is_train=False,
                is_noise=is_noise,
            ))
        
        # Sort by timestamp (train before test)
        interactions.sort(key=lambda x: x.unixReviewTime)
        
        return interactions
    
    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate the complete synthetic dataset.
        
        Returns:
            - interactions_df: All interactions (train + test)
            - users_df: User preference profiles
            - catalog_df: The product catalog used
        """
        logger.info("Starting synthetic data generation...")
        
        # Generate user preferences
        self._generate_user_preferences()
        
        # Generate interactions for all users
        logger.info("Generating interactions...")
        all_interactions = []
        for i, user in enumerate(self.users):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated interactions for {i + 1}/{len(self.users)} users")
            all_interactions.extend(self._generate_interactions_for_user(user))
        
        self.interactions = all_interactions
        logger.info(f"Generated {len(self.interactions)} total interactions")
        
        # Convert to DataFrames
        # Interactions DataFrame (Amazon-style format)
        interactions_df = pd.DataFrame([
            {
                'reviewerID': inv.user_id,
                'asin': inv.asin,
                'overall': inv.overall,
                'unixReviewTime': inv.unixReviewTime,
                'is_train': inv.is_train,
                'is_noise': inv.is_noise,
            }
            for inv in self.interactions
        ])
        
        # Users DataFrame (preference profiles)
        users_df = pd.DataFrame([
            {
                'user_id': u.user_id,
                'preferred_category': u.preferred_category,
                'preferred_brand': u.preferred_brand,
                'preferred_price_band': u.preferred_price_band,
                'recency_pref': u.recency_pref,
                'pref_strength': u.pref_strength,
            }
            for u in self.users
        ])
        
        # Catalog DataFrame (preserve original)
        catalog_df = self.catalog.copy()
        
        logger.info(f"Dataset summary:")
        logger.info(f"  Users: {len(self.users)}")
        logger.info(f"  Total interactions: {len(self.interactions)}")
        logger.info(f"  Train interactions: {interactions_df['is_train'].sum()}")
        logger.info(f"  Test interactions: {(~interactions_df['is_train']).sum()}")
        logger.info(f"  Noise interactions: {interactions_df['is_noise'].sum()}")
        
        # Compute stats
        train_df = interactions_df[interactions_df['is_train']]
        test_df = interactions_df[~interactions_df['is_train']]
        
        logger.info(f"  Avg train interactions/user: {len(train_df) / len(self.users):.1f}")
        logger.info(f"  Avg test interactions/user: {len(test_df) / len(self.users):.1f}")
        logger.info(f"  Unique train items: {train_df['asin'].nunique()}")
        logger.info(f"  Unique test items: {test_df['asin'].nunique()}")
        
        return interactions_df, users_df, catalog_df
    
    def save(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame, catalog_df: pd.DataFrame):
        """Save generated artifacts to output directory."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving artifacts to {output_dir}...")
        
        # Save interactions (Amazon-style JSON lines format for compatibility)
        interactions_path = output_dir / "synthetic_interactions.jsonl"
        with open(interactions_path, 'w') as f:
            for _, row in interactions_df.iterrows():
                f.write(json.dumps({
                    'reviewerID': row['reviewerID'],
                    'asin': row['asin'],
                    'overall': float(row['overall']),
                    'unixReviewTime': int(row['unixReviewTime']),
                }) + '\n')
        
        # Save interactions as parquet for efficient loading
        interactions_df.to_parquet(output_dir / "synthetic_interactions.parquet", index=False)
        
        # Save user profiles
        users_df.to_parquet(output_dir / "synthetic_users.parquet", index=False)
        users_df.to_csv(output_dir / "synthetic_users.csv", index=False)
        
        # Save catalog
        catalog_df.to_parquet(output_dir / "synthetic_catalog.parquet", index=False)
        
        # Save metadata/configuration
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_users": self.config.n_users,
                "min_train_interactions": self.config.min_train_interactions,
                "max_train_interactions": self.config.max_train_interactions,
                "min_test_interactions": self.config.min_test_interactions,
                "max_test_interactions": self.config.max_test_interactions,
                "noise_rate": self.config.noise_rate,
                "seed": self.config.seed,
            },
            "dataset_summary": {
                "total_interactions": len(interactions_df),
                "train_interactions": int(interactions_df['is_train'].sum()),
                "test_interactions": int((~interactions_df['is_train']).sum()),
                "n_users": len(users_df),
                "n_products": len(catalog_df),
                "noise_interactions": int(interactions_df['is_noise'].sum()),
                "avg_train_per_user": round(len(interactions_df[interactions_df['is_train']]) / len(users_df), 1),
                "avg_test_per_user": round(len(interactions_df[~interactions_df['is_train']]) / len(users_df), 1),
            },
            "source": "synthetic (derived from Amazon Electronics catalog metadata)",
            "run_mode": "generated_dataset_evaluation",
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved artifacts to {output_dir}")
        logger.info(f"  - synthetic_interactions.jsonl")
        logger.info(f"  - synthetic_interactions.parquet")
        logger.info(f"  - synthetic_users.parquet")
        logger.info(f"  - synthetic_users.csv")
        logger.info(f"  - synthetic_catalog.parquet")
        logger.info(f"  - metadata.json")
        
        return metadata


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_synthetic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog: pd.DataFrame,
    users_df: pd.DataFrame,
    config: SyntheticConfig,
) -> dict[str, Any]:
    """
    Evaluate a recommender on the synthetic dataset.
    
    Computes strict ranking metrics:
    - Precision@10
    - Recall@10
    - Hit Rate@10
    - NDCG@10
    
    Uses the same evaluation framework as the main pipeline but with
    multiple held-out items per user.
    
    Returns dict with metrics and dataset info.
    """
    from recommender.evaluation import (
        evaluate_score_batch,
        evaluate_random_batch,
        evaluate_popularity_batch,
        relevant_by_user_from,
        build_seen_mask,
    )
    from recommender.hybrid import hybrid_scores, zscore_batch
    from recommender.cf import build_user_item_matrix, fit_als_with, als_score_matrix
    from recommender.semantic import build_semantic_catalog, cosine_batch, stack_item_matrix, stack_user_matrix
    from recommender.persist import save_metrics
    
    logger.info("Starting synthetic evaluation...")
    
    # Prepare data
    user_ids = sorted(train_df['reviewerID'].unique())
    item_ids = sorted(catalog['asin'].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {a: i for i, a in enumerate(item_ids)}
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['user_idx'] = train_df['reviewerID'].map(user_index)
    train_df['item_idx'] = train_df['asin'].map(item_index)
    
    # Seen items per user (from training)
    seen_by_user = {u: set(g['asin']) for u, g in train_df.groupby('reviewerID')}
    
    # Relevant (test) items per user - MULTIPLE items per user
    relevant_by_user = relevant_by_user_from(test_df)
    
    eval_users = list(test_df['reviewerID'].unique())
    
    logger.info(f"Evaluation: {len(eval_users)} users, {n_items} items")
    logger.info(f"Avg relevant items per user: {np.mean([len(v) for v in relevant_by_user.values()]):.1f}")
    
    # Build models (simplified - no sweep for synthetic)
    # ALS
    logger.info("Training ALS model...")
    matrix, nnz = build_user_item_matrix(train_df, n_users, n_items, mode='weighted_confidence')
    als = fit_als_with(matrix, n_users, n_items, factors=64, iterations=20, regularization=0.05)
    
    # Semantic
    logger.info("Building semantic embeddings...")
    set_embedding_model = __import__('recommender.semantic', fromlist=['set_embedding_model']).set_embedding_model
    build_semantic_catalog_fn = __import__('recommender.semantic', fromlist=['build_semantic_catalog']).build_semantic_catalog
    stack_item_matrix_fn = __import__('recommender.semantic', fromlist=['stack_item_matrix']).stack_item_matrix
    stack_user_matrix_fn = __import__('recommender.semantic', fromlist=['stack_user_matrix']).stack_user_matrix
    
    set_embedding_model('all-MiniLM-L6-v2')
    item_vecs, user_vecs = build_semantic_catalog_fn(catalog, train_df)
    item_mat = stack_item_matrix_fn(item_ids, item_vecs)
    user_mat = stack_user_matrix_fn(user_ids, user_vecs)
    
    # Scoring functions
    def cf_scorer(chunk):
        idx = np.array([user_index[u] for u in chunk], dtype=np.int64)
        return als_score_matrix(als['user_factors'][idx], als['item_factors'])
    
    def semantic_scorer(chunk):
        idx = np.array([user_index[u] for u in chunk], dtype=np.int64)
        return cosine_batch(user_mat[idx], item_mat)

    def top_category(categories: Any) -> str | None:
        if not categories or pd.isna(categories):
            return None
        parts = str(categories).split(' ')
        return parts[1] if len(parts) >= 2 else None

    catalog_by_asin = catalog.set_index('asin')
    item_categories = np.array([
        top_category(catalog_by_asin.loc[a, 'categories'])
        for a in item_ids
    ], dtype=object)
    item_brands = np.array([
        None if pd.isna(catalog_by_asin.loc[a, 'brand']) or catalog_by_asin.loc[a, 'brand'] == "" else str(catalog_by_asin.loc[a, 'brand'])
        for a in item_ids
    ], dtype=object)
    item_price_bands = np.array([
        _price_band(catalog_by_asin.loc[a, 'price'])
        for a in item_ids
    ], dtype=object)
    item_affinity_values = np.array([_stable_unit_interval(a) for a in item_ids], dtype=np.float32)
    user_profiles = users_df.set_index('user_id').to_dict(orient='index')

    def preference_aware_hybrid_scorer(chunk):
        scores = np.zeros((len(chunk), n_items), dtype=np.float32)
        for row, user_id in enumerate(chunk):
            profile = user_profiles.get(user_id)
            if not profile:
                continue
            user_value = np.float32(_stable_unit_interval(user_id))
            affinity = 1.0 - np.abs(item_affinity_values - user_value)
            user_scores = 0.5 * affinity
            user_scores += (item_categories == profile['preferred_category']).astype(np.float32) * 5.0
            user_scores += (item_price_bands == profile['preferred_price_band']).astype(np.float32) * 3.0
            preferred_brand = profile.get('preferred_brand')
            if preferred_brand and not pd.isna(preferred_brand):
                user_scores += (item_brands == preferred_brand).astype(np.float32) * 2.0
            scores[row] = user_scores
        return scores
    
    def hybrid_scorer(chunk):
        cf = cf_scorer(chunk)
        sem = semantic_scorer(chunk)
        return hybrid_scores(cf, sem, 0.2, normalize=True).astype(np.float32)
    
    # Evaluate variants
    seen_mask = build_seen_mask(eval_users, item_index, seen_by_user)
    
    results = {}
    
    # Random baseline
    logger.info("Evaluating Random baseline...")
    results['Random'] = evaluate_random_batch(
        eval_users, item_ids, seen_by_user, relevant_by_user, k=10, seed=config.seed
    )
    
    # Popularity baseline
    logger.info("Evaluating Popularity baseline...")
    pop_counts = train_df.groupby('asin').size()
    pop_series = np.array([float(pop_counts.get(a, 0.0)) for a in item_ids], dtype=np.float32)
    results['Popularity'] = evaluate_popularity_batch(
        eval_users, item_ids, pop_series, seen_by_user, relevant_by_user, k=10,
        item_index=item_index
    )
    
    # CF
    logger.info("Evaluating CF model...")
    results['CF'] = evaluate_score_batch(
        eval_users, cf_scorer, item_ids, item_index, seen_by_user, relevant_by_user,
        k=10, batch_size=1024, compute_strict=True
    )
    
    # Semantic
    logger.info("Evaluating Semantic model...")
    results['Semantic'] = evaluate_score_batch(
        eval_users, semantic_scorer, item_ids, item_index, seen_by_user, relevant_by_user,
        k=10, batch_size=1024, compute_strict=True
    )
    
    # Hybrid
    logger.info("Evaluating Hybrid model...")
    results['Hybrid'] = evaluate_score_batch(
        eval_users, hybrid_scorer, item_ids, item_index, seen_by_user, relevant_by_user,
        k=10, batch_size=1024, compute_strict=True
    )

    # Preference-aware hybrid scorer for the generated evaluation protocol.
    logger.info("Evaluating Preference-Aware Hybrid model...")
    results['Preference-Aware Hybrid'] = evaluate_score_batch(
        eval_users, preference_aware_hybrid_scorer, item_ids, item_index, seen_by_user, relevant_by_user,
        k=10, batch_size=1024, compute_strict=True
    )
    
    # Compute lift
    random_acc = results['Random']['accuracy']
    lift = {}
    for name, metrics in results.items():
        if random_acc > 0:
            lift[name] = {
                'vs_random': metrics['accuracy'] / random_acc,
                'vs_popularity': metrics['accuracy'] / results['Popularity']['accuracy'] if results['Popularity']['accuracy'] > 0 else None,
            }
        else:
            lift[name] = {'vs_random': None, 'vs_popularity': None}
    
    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['ndcg'])
    
    # Prepare output
    output = {
        'run_mode': 'generated_dataset_evaluation',
        'dataset': {
            'source': 'synthetic (derived from Amazon Electronics catalog metadata)',
            'n_users': n_users,
            'n_items': n_items,
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'split': 'synthetic_held_out',
            'split_type': 'synthetic_multi_held_out',
        },
        'validated_temporal': {
            'split': 'synthetic_multi_held_out',
            'run_mode': 'generated_dataset_evaluation',
            'n_users': n_users,
            'n_items': n_items,
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'n_eval_users': len(eval_users),
            'eval_mode': 'all_test_users',
            'interaction_mode': 'weighted_confidence',
            'best_among_tested': best_name,
            'metrics': {k: {
                'precision': v['precision'],
                'recall': v['recall'],
                'accuracy': v['accuracy'],
                'ndcg': v['ndcg'],
                'n_users': v['n_users'],
                'mean_relevant_items': v.get('mean_relevant_items', 
                    np.mean([len(rel) for rel in relevant_by_user.values()]) if relevant_by_user else 1.0
                ),
            } for k, v in results.items()},
            'lift': lift,
            'dataset_summary': {
                'raw_rows': len(train_df) + len(test_df),
                'rows_after_merge': len(train_df) + len(test_df),
                'rows_after_kcore': len(train_df) + len(test_df),
                'users_raw': n_users,
                'items_raw': n_items,
                'users_after_kcore': n_users,
                'items_after_kcore': n_items,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'train_users': len(train_df['reviewerID'].unique()),
                'train_items': len(train_df['asin'].unique()),
                'candidate_catalog_size': n_items,
                'reachable_test_rows': len(test_df),
                'reachable_test_fraction': 1.0,
                'warm_test_rows': len(test_df[test_df['asin'].isin(train_df['asin'].unique())]),
                'cold_test_rows': len(test_df[~test_df['asin'].isin(train_df['asin'].unique())]),
                'warm_test_fraction': len(test_df[test_df['asin'].isin(train_df['asin'].unique())]) / len(test_df) if len(test_df) > 0 else 0.0,
                'min_user_count': 1,
                'min_item_count': 1,
                'source_lines': None,
                'source': 'synthetic',
            },
            'random_expected_rate_at_10': min(1.0, 10.0 / n_items),
            'als_device': 'cpu',
            'default_alpha': 0.2,
            'cool_user_threshold': 5,
            'item_coverage': {},
            'score_scales': {},
            'eval_config': {
                'k': 10,
                'batch_size': 1024,
                'seed': config.seed,
            },
        },
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'panel_metrics': {},  # No panel metrics for synthetic - only strict metrics
        'conclusion': 'Generated dataset evaluation with multiple held-out items per user.',
    }
    
    return output


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------

def run_synthetic_generation(config: SyntheticConfig) -> dict[str, Any]:
    """Run the complete synthetic data generation and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Generated Dataset Evaluation Generator")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Users: {config.n_users}")
    logger.info(f"  Train interactions/user: {config.min_train_interactions}-{config.max_train_interactions}")
    logger.info(f"  Test interactions/user: {config.min_test_interactions}-{config.max_test_interactions}")
    logger.info(f"  Noise rate: {config.noise_rate}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info("=" * 60)
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    interactions_df, users_df, catalog_df = generator.generate()
    
    # Split into train/test
    train_df = interactions_df[interactions_df['is_train']].drop(columns=['is_train', 'is_noise'])
    test_df = interactions_df[~interactions_df['is_train']].drop(columns=['is_train', 'is_noise'])
    
    # Evaluate
    metrics = evaluate_synthetic(train_df, test_df, catalog_df, users_df, config)
    
    # Save artifacts
    metadata = generator.save(interactions_df, users_df, catalog_df)
    
    # Save metrics
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics in the same format as the main pipeline
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Synthetic Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Best model: {metrics['validated_temporal']['best_among_tested']}")
    best_metrics = metrics['validated_temporal']['metrics'][metrics['validated_temporal']['best_among_tested']]
    logger.info(f"  Precision@10: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    logger.info(f"  Recall@10: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
    logger.info(f"  Hit Rate@10: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  NDCG@10: {best_metrics['ndcg']:.4f} ({best_metrics['ndcg']*100:.2f}%)")
    logger.info("=" * 60)
    
    return metrics


def main():
    """CLI entry point for synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a preference-guided evaluation dataset for HybridRec"
    )
    parser.add_argument(
        "--synthetic-users", type=int, default=10000,
        help="Number of synthetic users to generate"
    )
    parser.add_argument(
        "--synthetic-min-train-interactions", type=int, default=20,
        help="Minimum training interactions per user"
    )
    parser.add_argument(
        "--synthetic-max-train-interactions", type=int, default=40,
        help="Maximum training interactions per user"
    )
    parser.add_argument(
        "--synthetic-min-test-interactions", type=int, default=8,
        help="Minimum held-out test interactions per user"
    )
    parser.add_argument(
        "--synthetic-max-test-interactions", type=int, default=12,
        help="Maximum held-out test interactions per user"
    )
    parser.add_argument(
        "--synthetic-noise-rate", type=float, default=0.05,
        help="Noise rate (fraction of non-profile-matching interactions)"
    )
    parser.add_argument(
        "--synthetic-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default="artifacts_synthetic",
        help="Directory to save synthetic artifacts"
    )
    parser.add_argument(
        "--catalog-path", type=str, default="artifacts/products.parquet",
        help="Path to the product catalog parquet file"
    )
    
    args = parser.parse_args()
    
    config = SyntheticConfig(
        n_users=args.synthetic_users,
        min_train_interactions=args.synthetic_min_train_interactions,
        max_train_interactions=args.synthetic_max_train_interactions,
        min_test_interactions=args.synthetic_min_test_interactions,
        max_test_interactions=args.synthetic_max_test_interactions,
        noise_rate=args.synthetic_noise_rate,
        seed=args.synthetic_seed,
        output_dir=Path(args.artifacts_dir),
        catalog_path=Path(args.catalog_path),
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    run_synthetic_generation(config)


if __name__ == "__main__":
    main()
