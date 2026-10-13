const base = "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(base + path);
  const data = await res.json();
  if (!res.ok) {
    const err = new Error(data?.detail?.message || (typeof data?.detail === "string" ? data.detail : res.statusText));
    (err as Error & { detail?: unknown }).detail = data.detail;
    throw err;
  }
  return data as T;
}

export type RecItem = {
  rank: number;
  asin: string;
  title: string | null;
  image_url: string | null;
  cf_score: number | null;
  semantic_score: number | null;
  image_score?: number | null;
  hybrid_score: number | null;
  explanation: string;
};

export type MetricSet = {
  precision: number;
  recall: number;
  accuracy?: number; // Hit Rate @ K
  ndcg: number;
  n_users?: number;
};

export type LiftEntry = { vs_random?: number | null; vs_popularity?: number | null };

/** Headline metrics: percentages already scaled 0-100 (Lift is a multiplier). */
export type PanelMetrics = {
  Accuracy?: number | null;
  Recall?: number | null;
  "Sampled Accuracy"?: number | null;
  "Like Prediction Accuracy"?: number | null;
  Lift?: number | null;
  "Precision@10"?: number | null;
  "Accuracy / Hit Rate@10"?: number | null;
  "NDCG@10"?: number | null;
};

export type LikePredictionDetail = {
  accuracy?: number;
  baseline_accuracy?: number;
  roc_auc?: number | null;
  f1?: number | null;
  positive_rate?: number;
  n_train?: number;
  n_test?: number;
  label?: string;
  note?: string | null;
};

export type PanelMetricsPayload = {
  dataset: {
    run_mode: string;
    users: number | null;
    products: number | null;
    interactions: number | null;
    split: string;
  };
  headline: PanelMetrics;
  models: Record<string, PanelMetrics>;
};

export type VisionStatus = {
  enabled: boolean;
  model?: string;
  n_image_embeddings?: number;
  n_catalog_items?: number;
  missing_image_count?: number;
};

export type ArtifactSummary = {
  run_mode: string;
  artifact_is_full_dataset: boolean;
  source: string;
  created_utc: string | null;
  n_users?: number;
  n_items?: number;
  train_rows?: number;
  test_rows?: number;
  split_type?: string;
  dataset_summary?: Record<string, number | string> | null;
  eval_mode?: string;
  vision: VisionStatus;
  reranker_enabled: boolean;
};

export type ModelConfig = {
  als?: Record<string, unknown>;
  als_selection?: Record<string, unknown> | null;
  interaction?: Record<string, unknown>;
  semantic?: { primary?: string; model?: string };
  vision: VisionStatus;
  fusion?: Record<string, unknown>;
  reranker?: { enabled: boolean; coefficients?: Record<string, number | null> | null; n_train_samples?: number };
  normalization?: string;
  cool_user_threshold?: number;
};

export type RecommendResponse = {
  user_id: string;
  method: string;
  alpha: number | null;
  requested_alpha?: number;
  k: number;
  n_candidates: number;
  n_catalog: number;
  train_interactions: number;
  dynamic_fusion?: boolean;
  semantic_only_fallback?: boolean;
  reranked?: boolean;
  reranker_enabled?: boolean;
  weights?: { alpha_cf: number; beta_text: number; gamma_image: number };
  contributions?: { cf: number; text: number; image: number };
  vision_enabled?: boolean;
  items: RecItem[];
};

export const api = {
  status: () => get<Record<string, unknown>>("/api/status"),
  metrics: (mode = "synthetic") => get<Record<string, unknown>>(`/api/metrics?mode=${encodeURIComponent(mode)}`),
  panelMetrics: (mode = "synthetic") => get<PanelMetricsPayload>(`/api/panel-metrics?mode=${encodeURIComponent(mode)}`),
  artifactSummary: (mode = "synthetic") => get<ArtifactSummary>(`/api/artifact-summary?mode=${encodeURIComponent(mode)}`),
  modelConfig: () => get<ModelConfig>("/api/model-config"),
  users: (q = "") =>
    get<{
      demo_users: { user_id: string; train_interactions: number }[];
      matches: string[];
      n_known_users: number;
    }>(`/api/users?q=${encodeURIComponent(q)}`),
  history: (id: string) =>
    get<{
      user_id: string;
      n_interactions: number;
      average_rating: number | null;
      items: { asin: string; title: string; overall: number; image_url: string | null }[];
    }>(`/api/users/${encodeURIComponent(id)}/history`),
  recommend: (user: string, method: string, alpha: number | null, k = 10) =>
    get<RecommendResponse>(
      `/api/recommend?user_id=${encodeURIComponent(user)}&method=${method}` +
        (alpha == null ? "" : `&alpha=${alpha}`) +
        `&k=${k}`
    ),
  compare: (user: string, alpha: number | null, k = 10) =>
    get<{
      overlap_all_three: string[];
      overlap_all?: string[];
      unique: { cf: string[]; semantic: string[]; hybrid: string[] };
      products: {
        asin: string;
        title: string | null;
        cf_rank: number | null;
        semantic_rank: number | null;
        hybrid_rank: number | null;
        image_rank?: number | null;
      }[];
    }>(
      `/api/compare?user_id=${encodeURIComponent(user)}` +
        (alpha == null ? "" : `&alpha=${alpha}`) +
        `&k=${k}`
    ),
};
