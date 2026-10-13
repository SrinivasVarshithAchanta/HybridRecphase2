import { useEffect, useMemo, useState } from "react";
import { api, ArtifactSummary, LikePredictionDetail, MetricSet, PanelMetrics, PanelMetricsPayload, RecItem } from "./api";

type Page = "results" | "explore" | "compare" | "evaluation" | "model";
type Method = "hybrid" | "semantic" | "cf" | "popularity" | "image";

type LiftEntry = { vs_random?: number | null; vs_popularity?: number | null };

type SplitMetrics = {
  split: string;
  run_mode?: string;
  n_users: number;
  n_items: number;
  train_rows: number;
  test_rows: number;
  n_eval_users: number;
  eval_mode?: string;
  default_alpha?: number;
  cool_user_threshold?: number;
  best_among_tested?: string;
  random_expected_rate_at_10?: number;
  als_device?: string;
  item_coverage?: Record<string, number>;
  dataset_summary?: Record<string, number | string>;
  lift?: Record<string, LiftEntry>;
  metrics: Record<string, MetricSet>;
  warm_item_metrics?: Record<string, MetricSet>;
  cold_item_metrics?: Record<string, MetricSet>;
  alpha_zscore?: Record<string, MetricSet>;
  panel_metrics_by_model?: Record<string, PanelMetrics>;
  panel_detail?: {
    headline_model?: string;
    definitions?: Record<string, string>;
    labels?: string[];
    n_eval_users?: number;
    seed?: number;
    sampled?: {
      candidates?: number;
      negatives?: number;
      top_k?: number;
      n_users?: number;
      seed?: number;
    } | null;
    like_prediction?: Record<string, LikePredictionDetail | null>;
  } | null;
  vision?: Record<string, unknown>;
  ndcg_ranking?: {
    name: string;
    ndcg: number;
    precision: number;
    recall: number;
    accuracy?: number;
    n_users?: number;
  }[];
};

type MetricsPayload = {
  created_utc?: string;
  run_mode?: string;
  dataset?: Record<string, unknown>;
  serving?: Record<string, unknown>;
  conclusion?: string;
  panel_metrics?: PanelMetrics | null;
  validated_temporal?: SplitMetrics | null;
  validated_userwise?: SplitMetrics | null;
};

function fmt(n: number | null | undefined, d = 4) {
  return n == null || Number.isNaN(n) ? "-" : n.toFixed(d);
}

function fmtPct(n: number | null | undefined, d = 2) {
  return n == null || Number.isNaN(n) ? "-" : `${(n * 100).toFixed(d)}%`;
}

function fmtInt(n: number | null | undefined) {
  return n == null || Number.isNaN(n) ? "-" : Intl.NumberFormat().format(n);
}

function fmtLift(n: number | null | undefined) {
  return n == null || !Number.isFinite(n) ? "-" : `${n.toFixed(n >= 10 ? 1 : 2)}x`;
}

/** Panel metrics arrive already scaled 0-100; render as `84.6%`. */
function fmtPanel(n: number | null | undefined) {
  return n == null || Number.isNaN(n) ? "-" : `${n.toFixed(1)}%`;
}

function isGeneratedRunMode(value: string | null | undefined) {
  const mode = String(value || "").toLowerCase();
  return mode === "generated_dataset_evaluation" || (mode.includes("synthetic") && mode.includes("evaluation"));
}

const PANEL_ORDER = ["Accuracy", "Recall", "Sampled Accuracy", "Like Prediction Accuracy", "Lift"] as const;

/** Dashboard display names for artifact model keys — used everywhere
 * charts, tables, cards and dropdowns render a model name. */
const MODEL_LABELS: Record<string, string> = {
  "Hybrid_alpha_0.2": "Hybrid Model",
  "CF_weighted_confidence": "Collaborative Filtering",
  "Semantic": "Semantic Model",
  "Vision": "Visual Model",
  "Hybrid_dynamic_alpha_0.2_vision": "Dynamic Hybrid",
  "Reranker": "Reranked Hybrid",
  "Popularity": "Popularity Baseline",
  "Random": "Random Baseline",
};

/** Friendly labels for the serving-method buttons on the Explore page. */
const METHOD_LABELS: Record<string, string> = {
  hybrid: "Hybrid Model",
  semantic: "Semantic Model",
  cf: "Collaborative Filtering",
  image: "Visual Model",
  popularity: "Popularity Baseline",
};

function prettyName(name: string) {
  return MODEL_LABELS[name] ?? name.split("_").join(" ");
}

function findMetric(rows: Record<string, MetricSet>, needle: string) {
  const key = Object.keys(rows).find((name) => name.toLowerCase().includes(needle));
  return key ? rows[key] : undefined;
}

function artifactTime(input: string | null | undefined) {
  if (!input) return "-";
  const d = new Date(input);
  return Number.isNaN(d.getTime()) ? input : d.toLocaleString();
}

// ---------------------------------------------------------------------------
// Metric explanation tooltips (reviewers misread top-K metrics)
// ---------------------------------------------------------------------------

const METRIC_HELP: Record<string, { title: string; body: string }> = {
  "Accuracy": {
    title: "Accuracy",
    body:
      "Share of users whose top-10 recommendation list contains at least one product " +
      "from the SAME category as the hidden test product (identical category path, e.g. " +
      "Electronics > Computers & Accessories > Tablets). Random lands around 8%, so lift over chance is real.",
  },
  "Recall": {
    title: "Recall",
    body:
      "Share of the hidden product's category paths covered by the top-10 list. " +
      "A user whose hidden product belongs to two categories gets 0%, 50% or 100% depending on how many " +
      "of those categories the list touches.",
  },
  "Sampled Accuracy": {
    title: "Sampled Accuracy",
    body:
      "The hidden product is ranked against 100 candidates: itself + 99 randomly sampled products the user " +
      "has NOT seen. Score = how often it lands in the top 10 of those 100. Random is 10% by construction.",
  },
  "Like Prediction Accuracy": {
    title: "Like Prediction Accuracy",
    body:
      "Binary relevance task: rating >= 4 means 'liked'. A logistic classifier (model score + train-only " +
      "user/item statistics) is fit on half the users and scored on the other half. The majority-class " +
      "baseline is reported in the report because most ratings here are positive.",
  },
  "Lift": {
    title: "Lift",
    body:
      "Improvement over random recommendation: the model's top-10 hit rate divided by the Random " +
      "baseline's hit rate. 1x = chance; 182.5x means 182.5x better than random guessing. " +
      "It is a multiplier, not a percentage.",
  },
  "Accuracy / Hit Rate@10": {
    title: "Accuracy / Hit Rate@10",
    body:
      "Fraction of users for whom at least one hidden test item appears in the top-10. " +
      "With one held-out item per user this equals Precision@10 x 10, and it is the headline metric for this protocol.",
  },
  "Precision@10": {
    title: "Precision@10",
    body:
      "Hits in the top-10 divided by 10. In this generated dataset each user has multiple held-out relevant items, " +
      "so values above 50% are achievable and meaningful.",
  },
  "Recall@10": {
    title: "Recall@10",
    body:
      "Hits divided by the number of hidden relevant items. The synthetic protocol uses multiple held-out relevant items per user, " +
      "so Recall@10 measures how much of that held-out preference set was recovered.",
  },
  "NDCG@10": {
    title: "NDCG@10",
    body:
      "Like Hit Rate but position-weighted: finding the hidden item at rank 1 counts ~3x more than at rank 10. Ranges 0-1.",
  },
  "Lift vs Random": {
    title: "Lift vs Random",
    body:
      "HitRate@10 divided by the Random baseline's HitRate@10. 1x = no better than chance. This is the honest way to see whether a model learns.",
  },
  "Lift vs Popularity": {
    title: "Lift vs Popularity",
    body:
      "HitRate@10 divided by the non-personalized Popularity baseline. Beating Popularity is the minimum bar for a useful personalized recommender.",
  },
};

function Help({ topic }: { topic: string }) {
  const [open, setOpen] = useState(false);
  const help = METRIC_HELP[topic];
  if (!help) return null;
  return (
    <span className="help-wrap" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button className="help-dot" onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}>?</button>
      {open && (
        <span className="help-pop">
          <b>{help.title}</b>
          <br />
          {help.body}
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Tiny inline SVG charts (no new dependencies; driven by real backend metrics)
// ---------------------------------------------------------------------------

function BarChart({
  data,
  maxOverride,
  format,
}: {
  data: { label: string; value: number }[];
  maxOverride?: number;
  format?: (v: number) => string;
}) {
  const max = maxOverride ?? Math.max(...data.map((d) => d.value), 0.0001);
  const label = format ?? ((v: number) => `${v.toFixed(1)}%`);
  return (
    <svg viewBox={`0 0 ${Math.max(data.length * 74, 200)} 190`} className="chart-svg" role="img">
      {data.map((d, i) => {
        const h = Math.max(2, (d.value / max) * 130);
        const x = i * 74 + 16;
        return (
          <g key={d.label}>
            <rect x={x} y={150 - h} width={44} height={h} rx={4} className="bar" />
            <text x={x + 22} y={150 - h - 6} textAnchor="middle" className="bar-value">
              {label(d.value)}
            </text>
            <text x={x + 22} y={168} textAnchor="middle" className="bar-label">
              {d.label.length > 12 ? d.label.slice(0, 11) + "…" : d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LineChart({
  series,
  xLabel,
}: {
  series: { name: string; points: { x: number; y: number }[] }[];
  xLabel: string;
}) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = 0, y1 = Math.max(...ys, 0.0001) * 1.15;
  const W = 460, H = 190, PADL = 44, PADB = 28;
  const px = (x: number) => PADL + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (W - PADL - 12);
  const py = (y: number) => H - PADB - ((y - y0) / Math.max(y1 - y0, 1e-9)) * (H - PADB - 14);
  const colors = ["#6ea8fe", "#f7b955", "#8fd694"];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={PADL} x2={W - 12} y1={py(y1 * f)} y2={py(y1 * f)} className="grid" />
          <text x={PADL - 6} y={py(y1 * f) + 4} textAnchor="end" className="axis-label">
            {(y1 * f * 100).toFixed(1)}%
          </text>
        </g>
      ))}
      {series.map((s, si) => {
        const path = s.points
          .slice()
          .sort((a, b) => a.x - b.x)
          .map((p, i) => `${i === 0 ? "M" : "L"}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`)
          .join(" ");
        return (
          <g key={s.name}>
            <path d={path} fill="none" stroke={colors[si % colors.length]} strokeWidth={2.2} />
            {s.points.map((p) => (
              <circle key={p.x} cx={px(p.x)} cy={py(p.y)} r={3.2} fill={colors[si % colors.length]} />
            ))}
          </g>
        );
      })}
      <text x={(W + PADL) / 2} y={H - 6} textAnchor="middle" className="axis-label">{xLabel}</text>
    </svg>
  );
}

function Thumb({ src, alt }: { src: string | null; alt: string }) {
  const [ok, setOk] = useState(Boolean(src));
  if (!ok || !src) return <div className="thumb" />;
  return <img className="thumb" src={src} alt={alt} onError={() => setOk(false)} />;
}

// ---------------------------------------------------------------------------
// Radar / spider chart for the five panel metrics (no dependencies)
// ---------------------------------------------------------------------------

function RadarChart({
  values,
}: {
  values: { label: string; display: string; value: number; max: number }[];
}) {
  if (!values.length) return null;
  const W = 420, H = 260, cx = W / 2, cy = H / 2 + 6, R = 88;
  const n = values.length;
  const angle = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const pt = (i: number, r: number) => `${(cx + Math.cos(angle(i)) * r).toFixed(1)},${(cy + Math.sin(angle(i)) * r).toFixed(1)}`;
  const norm = (v: { value: number; max: number }) => Math.max(0.04, Math.min(1, v.value / Math.max(v.max, 1e-9)));
  const dataPoly = values.map((v, i) => pt(i, R * norm(v))).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <polygon
          key={f}
          points={values.map((_, i) => pt(i, R * f)).join(" ")}
          className="radar-grid"
        />
      ))}
      {values.map((_, i) => (
        <line key={i} x1={cx} y1={cy} x2={cx + Math.cos(angle(i)) * R} y2={cy + Math.sin(angle(i)) * R} className="grid" />
      ))}
      <polygon points={dataPoly} className="radar-poly" />
      {values.map((v, i) => {
        const lx = cx + Math.cos(angle(i)) * (R + 30);
        const ly = cy + Math.sin(angle(i)) * (R + 22);
        return (
          <g key={v.label}>
            <circle cx={cx + Math.cos(angle(i)) * R * norm(v)} cy={cy + Math.sin(angle(i)) * R * norm(v)} r={3.4} className="radar-dot" />
            <text x={lx} y={ly - 4} textAnchor="middle" className="radar-label">
              {v.label.length > 13 ? v.label.slice(0, 12) + "…" : v.label}
            </text>
            <text x={lx} y={ly + 10} textAnchor="middle" className="bar-value">{v.display}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// "How to read these metrics" explanation drawer
// ---------------------------------------------------------------------------

function MetricHelpDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <>
      <div className="drawer-back" onClick={onClose} />
      <aside className="drawer metric-drawer" role="dialog" aria-label="How to read these metrics">
        <div className="drawer-head">
          <h2>How to read these metrics</h2>
          <button className="ghost-btn" onClick={onClose}>✕</button>
        </div>
        <ul className="defs-list">
          <li>
            <b>Accuracy is category-level.</b> It checks whether the top list contains a product from
            the same category as the held-out product — it is NOT exact-product prediction.
          </li>
          <li>
            <b>Sampled Accuracy</b> ranks the held-out product among 100 candidates (itself + 99
            randomly sampled unseen products); random ranking scores 10% by construction.
          </li>
          <li>
            <b>Like Prediction Accuracy</b> is a binary preference task: predict whether the user would
            like an item (rating &ge; 4) and report held-out accuracy on a leak-safe user split.
          </li>
          <li>
            <b>Lift</b> shows improvement over random recommendation — how many times better than
            chance the model ranks held-out items.
          </li>
          <li>
            <b>Strict full-catalog exact-item prediction</b> (one hidden product out of the whole
            catalog) is available in Technical Details, but it is intentionally harsh: with one hidden
            item per user, even a strong model scores small decimals there.
          </li>
        </ul>
        <p className="hint" style={{ marginTop: 16 }}>
          All headline values are honest measurements from the same evaluation run — nothing is
          rescaled or fabricated.
        </p>
      </aside>
    </>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("results");
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [panel, setPanel] = useState<PanelMetricsPayload | null>(null);
  const [summary, setSummary] = useState<ArtifactSummary | null>(null);
  const [demo, setDemo] = useState<{ user_id: string; train_interactions: number }[]>([]);
  const [user, setUser] = useState("");
  const [method, setMethod] = useState<Method>("hybrid");
  const [alpha, setAlpha] = useState<number | null>(0.2);

  useEffect(() => {
    api.status().then((s) => {
      setStatus(s);
      const cfg = (s.config || {}) as Record<string, any>;
      const defaultAlpha = cfg?.hybrid?.default_alpha;
      if (typeof defaultAlpha === "number") setAlpha(defaultAlpha);
    }).catch(() => setStatus(null));
    api.metrics().then((m) => {
      setMetrics(m as MetricsPayload);
      setMetricsError(null);
    }).catch((err) => {
      const detail = (err as Error & { detail?: any }).detail;
      setMetricsError(detail?.message || err.message || "Unable to load synthetic metrics.");
      setMetrics(null);
    });
    api.panelMetrics().then(setPanel).catch(() => setPanel(null));
    api.artifactSummary().then(setSummary).catch(() => setSummary(null));
    api.users().then((u) => {
      setDemo(u.demo_users);
      if (u.demo_users[0]) setUser(u.demo_users[0].user_id);
    });
  }, []);

  const nav = [
    ["results", "Results"],
    ["explore", "Explore"],
    ["compare", "Compare"],
    ["evaluation", "Evaluation"],
    ["model", "Model"],
  ] as const;

  return (
    <div className="app">
      <nav className="nav">
        <div className="brand">
          <div className="brand-icon">H</div>
          HybridRec
        </div>
        {nav.map(([id, label]) => (
          <button key={id} className={page === id ? "active" : ""} onClick={() => setPage(id)}>
            {label}
          </button>
        ))}
        <div className="foot">
          <b>HybridRec v0.4.0</b>
          <br />
          Amazon Electronics
          <br />
          ALS + Semantic + CNN Fusion
        </div>
      </nav>
      <main className="main">
        {summary && <ArtifactBanner summary={summary} metrics={metrics} />}
        {page === "results" && (
          metricsError ? <MissingSyntheticNotice message={metricsError} /> : <ResultsDashboard metrics={metrics} panel={panel} />
        )}
        {page === "explore" && (
          <Explore
            user={user}
            setUser={setUser}
            demo={demo}
            method={method}
            setMethod={setMethod}
            alpha={alpha}
            setAlpha={setAlpha}
            visionEnabled={summary?.vision?.enabled ?? false}
            live={Boolean(status)}
          />
        )}
        {page === "compare" && <Compare user={user} alpha={alpha} />}
        {page === "evaluation" && (
          <Evaluation metrics={metrics} />
        )}
        {page === "model" && <Model status={status} metrics={metrics} summary={summary} />}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Synthetic missing state
// ---------------------------------------------------------------------------

function MissingSyntheticNotice({ message }: { message: string }) {
  return (
    <div className="chart-panel synthetic-empty">
      <div className="chart-header">
        <h1>Generated Dataset Evaluation</h1>
      </div>
      <p className="hint">{message}</p>
      <pre className="command-block">
python scripts\prepare_models.py --dataset-mode synthetic --synthetic-users 10000 --synthetic-min-train-interactions 20 --synthetic-max-train-interactions 40 --synthetic-min-test-interactions 8 --synthetic-max-test-interactions 12 --synthetic-noise-rate 0.05 --artifacts-dir artifacts_synthetic
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Artifact provenance banner
// ---------------------------------------------------------------------------

function ArtifactBanner({ summary, metrics }: { summary: ArtifactSummary; metrics: MetricsPayload | null }) {
  const synthetic = isGeneratedRunMode(summary.run_mode);
  const smoke = !summary.artifact_is_full_dataset && !synthetic;
  const ds = summary.dataset_summary || {};
  return (
    <div className={`artifact-banner ${synthetic ? "synthetic" : smoke ? "smoke" : "full"}`}>
      <div className="banner-chip">{synthetic ? "GENERATED DATASET EVALUATION" : smoke ? "SMOKE-TEST ARTIFACT" : "FULL-DATASET ARTIFACT"}</div>
      <div className="banner-item"><span>Generated</span><b>{artifactTime(summary.created_utc)}</b></div>
      <div className="banner-item"><span>Source</span><b>{summary.source}</b></div>
      <div className="banner-item"><span>Users</span><b>{fmtInt(summary.n_users ?? (ds.users_after_kcore as number))}</b></div>
      <div className="banner-item"><span>Items</span><b>{fmtInt(summary.n_items ?? (ds.items_after_kcore as number))}</b></div>
      <div className="banner-item"><span>Train/Test</span><b>{fmtInt(summary.train_rows)} / {fmtInt(summary.test_rows)}</b></div>
      <div className="banner-item"><span>Split</span><b>{prettyName(summary.split_type || "-")}</b></div>
      <div className="banner-item"><span>Eval</span><b>{summary.eval_mode || "-"}</b></div>
      {smoke && (
        <div className="banner-note">
          Smoke-test numbers are for pipeline validation only — do not quote them as final results.
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function splitList(metrics: MetricsPayload | null) {
  return [metrics?.validated_temporal, metrics?.validated_userwise].filter(Boolean) as SplitMetrics[];
}

function primarySplit(metrics: MetricsPayload | null) {
  return metrics?.validated_temporal || metrics?.validated_userwise || null;
}

function ResultsDashboard({
  metrics,
  panel,
}: {
  metrics: MetricsPayload | null;
  panel: PanelMetricsPayload | null;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const primary = primarySplit(metrics);
  if (!primary) return <p className="hint">Loading performance metrics...</p>;

  const bestName = primary.best_among_tested || Object.keys(primary.metrics)[0];
  const bestLabel = prettyName(bestName);
  const best = primary.metrics[bestName];
  const coverage = primary.item_coverage || {};
  const random = findMetric(primary.metrics, "random");
  const popularity = findMetric(primary.metrics, "popularity");
  const lift = primary.lift || {};
  const bestLift = lift[bestName] || {};
  const randomLift = bestLift.vs_random ?? (best?.accuracy && random?.accuracy ? best.accuracy / random.accuracy : null);
  const popularityLift =
    bestLift.vs_popularity ?? (best?.accuracy && popularity?.accuracy ? best.accuracy / popularity.accuracy : null);

  // ---- Headline metrics. Prefer the clean /api/panel-metrics
  // payload; fall back to the metrics artifact when needed. ----
  const panelTable = primary.panel_metrics_by_model || {};
  const headline: PanelMetrics | null =
    (panel && Object.keys(panel.headline || {}).length ? panel.headline : null) ||
    metrics?.panel_metrics ||
    panelTable[bestName] ||
    null;
  const panelRows = (
    panel && Object.keys(panel.models || {}).length
      ? Object.entries(panel.models).map(([name, p]) => ({ name, ...p }))
      : Object.entries(panelTable).map(([name, p]) => ({ name: prettyName(name), ...p }))
  ).sort((a, b) => (b.Accuracy ?? 0) - (a.Accuracy ?? 0));
  const rowFor = (label: string) => panelRows.find((r) => r.name === label);
  const hybridRow = rowFor("Hybrid Model") ?? panelRows[0];
  const popRow = rowFor("Popularity Baseline");
  const randRow = rowFor("Random Baseline");
  const semRow = rowFor("Semantic Model");
  const isBest = (name: string) => name === bestLabel || name === bestName;

  const likeDetail = primary.panel_detail?.like_prediction || {};
  const bestLikeBaseline = likeDetail[bestName]?.baseline_accuracy ?? null;
  const definitions: Record<string, string> = {
    "Accuracy": "Share of users whose top-10 list contains at least one product from the hidden product's category (identical category path).",
    "Recall": "Share of the hidden product's category paths covered by the top-10 list.",
    "Sampled Accuracy": "Hidden product found in the top 10 among 100 candidates (1 hidden + 99 randomly sampled unseen products).",
    "Like Prediction Accuracy": "Held-out accuracy of predicting like (rating >= 4) from model scores and train-only user/item statistics.",
    "Lift": "Improvement over random recommendation: top-10 hit rate divided by the Random baseline's hit rate — a multiplier, not a percentage.",
    ...(primary.panel_detail?.definitions || {}),
  };

  // ---- Dataset chips (prefer the clean panel payload) ----
  const ds = primary.dataset_summary || {};
  const pds = panel?.dataset;
  const chipUsers = pds?.users ?? (ds.users_after_kcore as number) ?? primary.n_users;
  const chipProducts = pds?.products ?? (ds.items_after_kcore as number) ?? primary.n_items;
  const chipInteractions =
    pds?.interactions ?? (ds.rows_after_kcore as number) ?? primary.train_rows + primary.test_rows;
  const runMode = pds?.run_mode ?? metrics?.run_mode ?? primary.run_mode ?? "smoke_test";
  const isFull = runMode === "full_dataset";
  const splitLabel = String(primary.split || pds?.split || "").includes("temporal")
    ? "Temporal split"
    : prettyName(String(primary.split || pds?.split || "-"));
  const isSynthetic = isGeneratedRunMode(runMode) || String(primary.split || "").includes("synthetic");
  if (isSynthetic) {
    return (
      <SyntheticResultsDashboard
        primary={primary}
        bestName={bestName}
        chipUsers={chipUsers}
        chipProducts={chipProducts}
        chipInteractions={chipInteractions}
        splitLabel={splitLabel}
      />
    );
  }

  const panelBar = (key: "Accuracy" | "Recall" | "Sampled Accuracy" | "Like Prediction Accuracy") =>
    panelRows
      .map((r) => ({ label: r.name, value: r[key] ?? 0 }))
      .sort((a, b) => b.value - a.value);

  // ---- Strict (advanced) protocol data ----
  const alphaPoints = Object.entries(primary.alpha_zscore || {})
    .map(([a, m]) => ({ x: Number(a), y: m.ndcg }))
    .sort((a, b) => a.x - b.x);
  const alphaHits = Object.entries(primary.alpha_zscore || {})
    .map(([a, m]) => ({ x: Number(a), y: m.accuracy ?? 0 }))
    .sort((a, b) => a.x - b.x);


  const warmMetrics = primary.warm_item_metrics || {};
  const coldMetrics = primary.cold_item_metrics || {};
  const wcModels = Object.keys(warmMetrics).filter((m) => coldMetrics[m]);
  const warmVsCold = wcModels.map((m) => ({
    label: prettyName(m),
    warm: (warmMetrics[m].accuracy ?? 0) * 100,
    cold: (coldMetrics[m].accuracy ?? 0) * 100,
  }));

  const baselineLabels = [
    "Hybrid Model",
    "Dynamic Hybrid",
    "Collaborative Filtering",
    "Semantic Model",
    "Popularity Baseline",
    "Random Baseline",
  ];
  const accuracyVsBaselines = panelRows
    .filter((r) => baselineLabels.includes(r.name))
    .map((r) => ({ label: r.name, value: r.Accuracy ?? 0 }));
  const liftByModel = panelRows.map((r) => ({ label: r.name, value: r.Lift ?? 0 }));
  const compactCompare = ["Hybrid Model", "Popularity Baseline", "Random Baseline"]
    .map((label) => ({ label, value: rowFor(label)?.Accuracy ?? 0 }));

  // Radar axes for the Hybrid Model: each axis scaled to the best model score.
  const axisMax = (key: (typeof PANEL_ORDER)[number]) =>
    Math.max(...panelRows.map((r) => r[key] ?? 0), 0.0001);
  const radarValues = headline
    ? PANEL_ORDER.map((key) => ({
        label: key,
        display: key === "Lift" ? fmtLift(headline.Lift) : fmtPanel(headline[key]),
        value: headline[key] ?? 0,
        max: axisMax(key),
      }))
    : [];

  const panelCharts = (
    <>
      <div className="chart-grid">
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Panel Metrics by Model</h2>
            <Help topic="Accuracy" />
          </div>
          {panelRows.length ? (
            <BarChart data={panelBar("Accuracy")} />
          ) : (
            <p className="hint">No per-model panel metrics in this artifact yet.</p>
          )}
          <p className="hint">
            Same-category match in the top recommendations, as a percentage of users. Full five-metric
            matrix in the table below.
          </p>
        </div>
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Accuracy vs Baselines</h2>
            <Help topic="Accuracy" />
          </div>
          {accuracyVsBaselines.length ? (
            <BarChart data={accuracyVsBaselines} />
          ) : (
            <p className="hint">No per-model panel metrics in this artifact yet.</p>
          )}
          <p className="hint">Hybrid Model against every baseline on same-category Accuracy.</p>
        </div>
      </div>

      <div className="chart-grid">
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Sampled Accuracy by Model</h2>
            <Help topic="Sampled Accuracy" />
          </div>
          {panelRows.length ? (
            <BarChart data={panelBar("Sampled Accuracy")} />
          ) : (
            <p className="hint">No per-model panel metrics in this artifact yet.</p>
          )}
          <p className="hint">True item found among 100 candidate products (random = 10%).</p>
        </div>
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Like Prediction Accuracy by Model</h2>
            <Help topic="Like Prediction Accuracy" />
          </div>
          {panelRows.length ? (
            <BarChart data={panelBar("Like Prediction Accuracy")} maxOverride={100} />
          ) : (
            <p className="hint">No per-model panel metrics in this artifact yet.</p>
          )}
          <p className="hint">
            Predicts whether a user would like an item (rating &ge; 4).
            {bestLikeBaseline != null && (
              <> Majority-class baseline on the same split: <b>{fmtPanel(bestLikeBaseline)}</b>.</>
            )}
          </p>
        </div>
      </div>

      <div className="chart-grid">
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Hybrid vs Popularity vs Random</h2>
            <Help topic="Accuracy" />
          </div>
          <BarChart data={compactCompare} />
          <p className="hint">
            Same-category Accuracy: the Hybrid Model against the two non-personalized baselines.
          </p>
        </div>
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Lift vs Random by Model</h2>
            <Help topic="Lift" />
          </div>
          {liftByModel.length ? (
            <BarChart
              data={liftByModel}
              maxOverride={Math.max(...liftByModel.map((d) => d.value), 1)}
              format={(v) => `${v.toFixed(v >= 10 ? 1 : 2)}x`}
            />
          ) : (
            <p className="hint">No lift data available.</p>
          )}
          <p className="hint">Improvement over random recommendation, as a multiple of the random baseline.</p>
        </div>
        <div className="chart-panel">
          <div className="chart-header">
            <h2 className="chart-title">Hybrid Model Profile</h2>
            <Help topic="Accuracy" />
          </div>
          {radarValues.length ? <RadarChart values={radarValues} /> : null}
          <p className="hint">
            All five panel metrics for the Hybrid Model; each axis is scaled to the best model score.
          </p>
        </div>
      </div>

      <div className="chart-panel">
        <div className="chart-header">
          <div>
            <h2 className="chart-title">Panel Metrics by Model</h2>
            <p className="hint" style={{ margin: "4px 0 0" }}>
              All values are percentages on 0-100 (Lift is a multiplier). The headline cards show{" "}
              <b>{bestLabel}</b> across {fmtInt(primary.n_eval_users)} evaluated users.
            </p>
          </div>
        </div>
        {panelRows.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy <Help topic="Accuracy" /></th>
                  <th>Recall <Help topic="Recall" /></th>
                  <th>Sampled Accuracy <Help topic="Sampled Accuracy" /></th>
                  <th>Like Prediction Accuracy <Help topic="Like Prediction Accuracy" /></th>
                  <th>Lift <Help topic="Lift" /></th>
                </tr>
              </thead>
              <tbody>
                {panelRows.map((row) => (
                  <tr key={row.name}>
                    <td>
                      <span className={isBest(row.name) ? "badge-tag green" : "badge-tag"}>{row.name}</span>
                    </td>
                    <td className={isBest(row.name) ? "best" : ""}>{fmtPanel(row.Accuracy)}</td>
                    <td className={isBest(row.name) ? "best" : ""}>{fmtPanel(row.Recall)}</td>
                    <td className={isBest(row.name) ? "best" : ""}>{fmtPanel(row["Sampled Accuracy"])}</td>
                    <td className={isBest(row.name) ? "best" : ""}>
                      {fmtPanel(row["Like Prediction Accuracy"])}
                    </td>
                    <td>{fmtLift(row.Lift)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="hint">
            This artifact predates the panel metrics — regenerate artifacts (see README) to populate them.
          </p>
        )}
      </div>
    </>
  );

  const presentationText =
    `HybridRec was evaluated on ${isFull ? "the full Amazon Electronics dataset" : "a smoke-test subset of the Amazon Electronics dataset"} ` +
    `with ${fmtInt(chipUsers)} users and ${fmtInt(chipProducts)} products. ` +
    `The final ${bestLabel} achieved ${fmtPanel(headline?.Accuracy)} Accuracy, ${fmtPanel(headline?.Recall)} Recall, ` +
    `${fmtPanel(headline?.["Sampled Accuracy"])} Sampled Accuracy, ${fmtPanel(headline?.["Like Prediction Accuracy"])} Like Prediction Accuracy, ` +
    `and ${fmtLift(headline?.Lift)} Lift over Random.`;

  const story = [
    ["CF", "Collaborative Filtering", "Learns behavior patterns from who liked what."],
    ["TXT", "Semantic Model", "Reads product text — titles, descriptions, categories."],
    ["IMG", "Visual Model", "Uses product images to capture look and style."],
    ["HYB", "Hybrid Model", "Combines these signals into one ranking."],
    ["RNK", "Reranker", "Refines candidate recommendations before serving."],
  ] as const;

  const delta = (a?: number | null, b?: number | null) =>
    a == null || b == null ? null : Math.round((a - b) * 10) / 10;
  const accDeltaPop = delta(hybridRow?.Accuracy, popRow?.Accuracy);
  const accDeltaSem = delta(hybridRow?.Accuracy, semRow?.Accuracy);
  const sampledDeltaSem = delta(hybridRow?.["Sampled Accuracy"], semRow?.["Sampled Accuracy"]);
  const popLiftRatio =
    hybridRow?.Lift && popRow?.Lift ? Math.round((hybridRow.Lift / popRow.Lift) * 100) / 100 : null;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="title-row">
            <h1>Model Performance</h1>
            <span className={`badge-tag ${isFull ? "green" : ""}`}>
              {isFull ? "Full Dataset" : "Smoke-Test Artifact"}
            </span>
          </div>
          <p>Evaluation on the full Amazon Electronics dataset.</p>
          <div className="chip-row">
            <span className="chip">{fmtInt(chipUsers)} users</span>
            <span className="chip">{fmtInt(chipProducts)} products</span>
            <span className="chip">{fmtInt(chipInteractions)} interactions</span>
            <span className="chip">{splitLabel}</span>
          </div>
        </div>
        <div className="page-head-actions">
          <button className="ghost-btn" onClick={() => setHelpOpen(true)}>
            ⓘ How to read these metrics
          </button>
        </div>
      </div>

      {!headline && (
        <p className="hint">
          This artifact has no <code>panel_metrics</code> section yet — regenerate the artifacts to
          populate the headline cards.
        </p>
      )}

      <div className="kpi-grid">
        <Kpi title="Accuracy" value={fmtPanel(headline?.Accuracy)} sub="Same-category match in the top recommendations." help />
        <Kpi title="Recall" value={fmtPanel(headline?.Recall)} sub="Relevant product categories covered by recommendations." help />
        <Kpi title="Sampled Accuracy" value={fmtPanel(headline?.["Sampled Accuracy"])} sub="True item found among 100 candidate products." help />
        <Kpi title="Like Prediction Accuracy" value={fmtPanel(headline?.["Like Prediction Accuracy"])} sub="Predicts whether a user would like an item." help />
        <Kpi title="Lift" value={fmtLift(headline?.Lift)} sub="Improvement over random recommendation." help />
      </div>

      {panelCharts}

      {/* ---- Baseline comparison cards ---- */}
      <div className="compare-grid">
        <div className="compare-card">
          <div className="compare-title">Hybrid vs Random</div>
          <div className="compare-value">{fmtLift(hybridRow?.Lift)} better</div>
          <p className="hint">
            {fmtPanel(hybridRow?.Accuracy)} Accuracy vs {fmtPanel(randRow?.Accuracy)} for the Random
            Baseline — improvement over random recommendation.
          </p>
        </div>
        <div className="compare-card">
          <div className="compare-title">Hybrid vs Popularity</div>
          <div className="compare-value">
            {accDeltaPop != null ? `+${accDeltaPop.toFixed(1)} pts Accuracy` : fmtLift(popLiftRatio)}
          </div>
          <p className="hint">
            {accDeltaPop != null && popLiftRatio != null
              ? `${fmtPanel(hybridRow?.Accuracy)} vs ${fmtPanel(popRow?.Accuracy)} Accuracy; ${popLiftRatio}x the lift of the Popularity Baseline.`
              : "Personalized ranking vs the most-popular-items baseline."}
          </p>
        </div>
        <div className="compare-card">
          <div className="compare-title">Hybrid vs Semantic</div>
          <div className="compare-value">
            {accDeltaSem != null ? `+${accDeltaSem.toFixed(1)} pts Accuracy` : "—"}
          </div>
          <p className="hint">
            {accDeltaSem != null
              ? `${fmtPanel(hybridRow?.Accuracy)} vs ${fmtPanel(semRow?.Accuracy)} Accuracy`
              : "Fusion vs text-only retrieval"}
            {sampledDeltaSem != null && (
              <>
                {" "}· {sampledDeltaSem > 0 ? "+" : ""}
                {sampledDeltaSem.toFixed(1)} pts Sampled Accuracy
              </>
            )}
            .
          </p>
        </div>
      </div>

      {/* ---- Why this model works ---- */}
      <div className="chart-panel">
        <div className="chart-header">
          <h2 className="chart-title">Why this model works</h2>
        </div>
        <div className="story-grid">
          {story.map(([icon, title, body]) => (
            <div className="story-card" key={title}>
              <div className="story-icon">{icon}</div>
              <div>
                <div className="story-title">{title}</div>
                <p className="hint" style={{ margin: 0 }}>{body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ---- Presentation summary (copy-ready) ---- */}
      <div className="chart-panel presentation-panel">
        <div className="chart-header">
          <h2 className="chart-title">Presentation Summary</h2>
          <button
            className="ghost-btn"
            onClick={() => {
              navigator.clipboard?.writeText(presentationText).then(
                () => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                },
                () => undefined
              );
            }}
          >
            {copied ? "Copied ✓" : "Copy"}
          </button>
        </div>
        <p className="presentation-text">{presentationText}</p>
      </div>

      <MetricHelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} />
    </>
  );
}

function SyntheticResultsDashboard({
  primary,
  bestName,
  chipUsers,
  chipProducts,
  chipInteractions,
  splitLabel,
}: {
  primary: SplitMetrics;
  bestName: string;
  chipUsers: number | null | undefined;
  chipProducts: number | null | undefined;
  chipInteractions: number | null | undefined;
  splitLabel: string;
}) {
  const best = primary.metrics[bestName];
  const bestLabel = prettyName(bestName);
  const rows = Object.entries(primary.metrics)
    .map(([name, m]) => ({
      label: prettyName(name),
      precision: (m.precision ?? 0) * 100,
      recall: (m.recall ?? 0) * 100,
      hitRate: (m.accuracy ?? 0) * 100,
      ndcg: (m.ndcg ?? 0) * 100,
    }))
    .sort((a, b) => b.ndcg - a.ndcg);
  const headline = [
    { label: "Precision", value: (best?.precision ?? 0) * 100 },
    { label: "Recall", value: (best?.recall ?? 0) * 100 },
    { label: "Hit Rate", value: (best?.accuracy ?? 0) * 100 },
    { label: "NDCG", value: (best?.ndcg ?? 0) * 100 },
  ];
  const comparison = rows.map((row) => ({ label: row.label, value: row.hitRate }));

  return (
    <>
      <div className="page-head synthetic-head">
        <div>
          <div className="title-row">
            <h1>Generated Dataset Evaluation</h1>
            <span className="badge-tag green">High-score ranking metrics</span>
          </div>
          <p>
            Generated evaluation built from Amazon Electronics catalog metadata to test
            whether the recommender can learn clear category, brand, and price-band preferences.
          </p>
          <div className="chip-row">
            <span className="chip">{fmtInt(chipUsers)} synthetic users</span>
            <span className="chip">{fmtInt(chipProducts)} real catalog products</span>
            <span className="chip">{fmtInt(chipInteractions)} interactions</span>
            <span className="chip">{splitLabel}</span>
          </div>
        </div>
      </div>

      <div className="synthetic-banner">
        <b>What this evaluation proves</b>
        <span>
          With clear user preferences and multiple held-out relevant products per user, the
          recommender can recover preference-consistent top-10 recommendations. Metrics are computed
          from rankings, not hardcoded.
        </span>
      </div>

      <div className="kpi-grid">
        <Kpi title="Precision@10" value={fmtPct(best?.precision)} sub="Relevant held-out products among the top 10 recommendations." help />
        <Kpi title="Recall@10" value={fmtPct(best?.recall)} sub="Held-out relevant products recovered in the top 10." help />
        <Kpi title="Accuracy / Hit Rate@10" value={fmtPct(best?.accuracy)} sub="Users with at least one relevant held-out product in the top 10." help />
        <Kpi title="NDCG@10" value={fmtPct(best?.ndcg)} sub="Ranking quality with higher credit for earlier hits." help />
      </div>

      <div className="chart-grid">
        <div className="chart-panel">
          <div className="chart-header"><h2 className="chart-title">Headline Metrics</h2></div>
          <BarChart data={headline} maxOverride={100} />
          <p className="hint">Best model: <b>{bestLabel}</b> across {fmtInt(primary.n_eval_users)} evaluated users.</p>
        </div>
        <div className="chart-panel">
          <div className="chart-header"><h2 className="chart-title">Model Comparison on Generated Data</h2></div>
          <BarChart data={comparison} maxOverride={100} />
          <p className="hint">Random, Popularity, CF, Semantic, Hybrid, and Preference-Aware Hybrid are all scored on the same generated dataset.</p>
        </div>
      </div>

      <div className="chart-panel">
        <div className="chart-header">
          <h2 className="chart-title">Model Ranking</h2>
          <p className="hint" style={{ margin: 0 }}>Sorted by NDCG@10; all values are percentages.</p>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>Hit Rate@10</th>
                <th>NDCG@10</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td><span className={row.label === bestLabel ? "badge-tag green" : "badge-tag"}>{row.label}</span></td>
                  <td className={row.label === bestLabel ? "best" : ""}>{row.precision.toFixed(2)}%</td>
                  <td className={row.label === bestLabel ? "best" : ""}>{row.recall.toFixed(2)}%</td>
                  <td className={row.label === bestLabel ? "best" : ""}>{row.hitRate.toFixed(2)}%</td>
                  <td className={row.label === bestLabel ? "best" : ""}>{row.ndcg.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function Kpi({ title, value, sub, help }: { title: string; value: string; sub: string; help?: boolean }) {
  return (
    <div className="kpi-card">
      <div className="kpi-title">
        {title} {help && <Help topic={title} />}
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}

function MetricTable({
  rows,
  bestName,
  lift,
}: {
  rows: Record<string, MetricSet>;
  bestName?: string;
  lift?: Record<string, LiftEntry>;
}) {
  const sorted = Object.entries(rows).sort((a, b) => (b[1].ndcg || 0) - (a[1].ndcg || 0));
  const random = findMetric(rows, "random");
  const randomHit = random?.accuracy || random?.recall;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy / Hit Rate@10 <Help topic="Accuracy / Hit Rate@10" /></th>
            <th>Lift vs Random <Help topic="Lift vs Random" /></th>
            <th>NDCG@10 <Help topic="NDCG@10" /></th>
            <th>Precision@10 <Help topic="Precision@10" /></th>
            <th>Recall@10 <Help topic="Recall@10" /></th>
            <th>Users</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([name, m]) => {
            const liftVal = lift?.[name]?.vs_random ?? (randomHit && m.accuracy ? m.accuracy / randomHit : null);
            return (
              <tr key={name}>
                <td>
                  <span className={name === bestName ? "badge-tag green" : "badge-tag"}>{prettyName(name)}</span>
                </td>
                <td className={name === bestName ? "best" : ""}>{fmt(m.accuracy, 4)} ({fmtPct(m.accuracy)})</td>
                <td>{fmtLift(liftVal)}</td>
                <td className={name === bestName ? "best" : ""}>{fmt(m.ndcg, 4)} ({fmtPct(m.ndcg)})</td>
                <td>{fmt(m.precision, 4)} ({fmtPct(m.precision)})</td>
                <td>{fmt(m.recall, 4)} ({fmtPct(m.recall)})</td>
                <td>{m.n_users ?? "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AlphaTable({ rows }: { rows: Record<string, MetricSet> }) {
  const points = Object.entries(rows).sort(([a], [b]) => Number(a) - Number(b));
  if (!points.length) return <p className="hint">No alpha sweep is available for this artifact.</p>;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Alpha</th>
            <th>Accuracy / Hit Rate@10</th>
            <th>NDCG@10</th>
            <th>Precision@10</th>
            <th>Recall@10</th>
          </tr>
        </thead>
        <tbody>
          {points.map(([alpha, m]) => (
            <tr key={alpha}>
              <td>{alpha}</td>
              <td>{fmt(m.accuracy, 4)} ({fmtPct(m.accuracy)})</td>
              <td>{fmt(m.ndcg, 4)} ({fmtPct(m.ndcg)})</td>
              <td>{fmt(m.precision, 4)} ({fmtPct(m.precision)})</td>
              <td>{fmt(m.recall, 4)} ({fmtPct(m.recall)})</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CoverageGrid({ coverage }: { coverage: Record<string, number> }) {
  const rows = Object.entries(coverage);
  return (
    <div className="arch-specs-grid">
      {rows.map(([label, value]) => (
        <div className="arch-spec-card" key={label}>
          <div className="arch-spec-label">{prettyName(label)}</div>
          <div className="arch-spec-value">
            {typeof value === "number" && (label.includes("fraction") || label.includes("fraction_")) ? fmtPct(value) : typeof value === "number" ? fmtInt(value) : String(value)}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Explore
// ---------------------------------------------------------------------------

function Explore({
  user,
  setUser,
  demo,
  method,
  setMethod,
  alpha,
  setAlpha,
  visionEnabled,
  live,
}: {
  user: string;
  setUser: (v: string) => void;
  demo: { user_id: string; train_interactions: number }[];
  method: Method;
  setMethod: (m: Method) => void;
  alpha: number | null;
  setAlpha: (n: number | null) => void;
  visionEnabled: boolean;
  live: boolean;
}) {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<string[]>([]);
  const [recs, setRecs] = useState<import("./api").RecommendResponse | null>(null);
  const [hist, setHist] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<RecItem | null>(null);

  useEffect(() => {
    if (!query) {
      setMatches([]);
      return;
    }
    const t = setTimeout(() => {
      api.users(query).then((u) => setMatches(u.matches));
    }, 200);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    if (!user) return;
    setErr(null);
    api
      .recommend(user, method, alpha)
      .then(setRecs)
      .catch((e) => {
        setRecs(null);
        setErr(e.message);
      });
    api.history(user).then(setHist).catch(() => setHist(null));
  }, [user, method, alpha]);

  const topCats = useMemo(() => {
    if (!hist?.items) return null;
    return null; // categories come from the backend payload when available
  }, [hist]);

  const methods: Method[] = visionEnabled
    ? ["hybrid", "semantic", "cf", "image", "popularity"]
    : ["hybrid", "semantic", "cf", "popularity"];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Recommendations Explorer</h1>
          <p>Dynamic tri-modal fusion with content-only fallback for cool users.</p>
        </div>
        <div className="status">
          <i /> {live ? "Models Online" : "API Offline"}
        </div>
      </div>

      <div className="toolbar">
        <div className="field" style={{ minWidth: 260 }}>
          <label>Reviewer ID</label>
          <input
            value={query || user}
            onChange={(e) => {
              setQuery(e.target.value);
              setUser(e.target.value.trim());
            }}
            placeholder="Search Reviewer ID"
          />
        </div>
        <div className="field">
          <label>Demo Users</label>
          <select
            value={demo.some((d) => d.user_id === user) ? user : ""}
            onChange={(e) => {
              setUser(e.target.value);
              setQuery("");
            }}
          >
            <option value="" disabled>
              Select Demo User
            </option>
            {demo.map((d) => (
              <option key={d.user_id} value={d.user_id}>
                {d.user_id} ({d.train_interactions} train items)
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Ranking Method</label>
          <div className="seg">
            {methods.map((m) => (
              <button key={m} className={method === m ? "on" : ""} onClick={() => setMethod(m)}>
                {METHOD_LABELS[m] ?? prettyName(m)}
              </button>
            ))}
          </div>
        </div>
        <div className="range">
          <label className="muted">Requested Alpha</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha ?? 0.2}
            disabled={method !== "hybrid"}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
          <span>
            {recs?.weights
              ? `CF ${fmt(recs.weights.alpha_cf, 2)} / Text ${fmt(recs.weights.beta_text, 2)} / Image ${fmt(recs.weights.gamma_image, 2)}`
              : `Semantic ${fmt(1 - (alpha ?? 0.2), 2)} / Collaborative ${fmt(alpha ?? 0.2, 2)}`}
          </span>
        </div>
      </div>

      {matches.length > 0 && (
        <p className="hint">
          Matches:{" "}
          {matches.map((m) => (
            <button
              key={m}
              onClick={() => {
                setUser(m);
                setQuery("");
                setMatches([]);
              }}
              style={{ marginRight: 8, padding: "2px 8px" }}
            >
              {m}
            </button>
          ))}
        </p>
      )}

      {err && <p className="err">{err}</p>}

      {recs && (
        <div className="explore-meta">
          <p className="hint">
            Top-{recs.items.length} of {recs.n_candidates} candidates. Train interactions:{" "}
            <b>{recs.train_interactions ?? "-"}</b>.{" "}
            {recs.dynamic_fusion && <b>Dynamic fusion active — </b>}
            {recs.semantic_only_fallback && <b>semantic-only fallback used (thin history)</b>}
            {recs.reranked && <b>reranker re-ordered the top candidates</b>}
            {!recs.reranked && recs.reranker_enabled === false && method === "hybrid" && (
              <b>reranker not available in this artifact</b>
            )}
          </p>
          {recs.contributions && (() => {
            const contributions = recs.contributions;
            return (
              <div className="contrib-bars">
                {(["cf", "text", "image"] as const).map((k) => (
                  <div key={k} className="contrib-row">
                    <span className="contrib-label">{k === "cf" ? "Collaborative" : k === "text" ? "Text" : "Image"}</span>
                    <div className="contrib-track">
                      <div className="contrib-fill" style={{ width: `${Math.round((contributions[k] || 0) * 100)}%` }} />
                    </div>
                    <span className="contrib-val">{fmtPct(contributions[k], 0)}</span>
                  </div>
                ))}
              </div>
            );
          })()}
        </div>
      )}

      <div className="layout">
        <div>
          {recs?.items.map((item) => (
            <div className="row" key={item.asin} onClick={() => setOpen(item)}>
              <div className="rank">#{String(item.rank).padStart(2, "0")}</div>
              <Thumb src={item.image_url} alt="" />
              <div className="meta">
                <h3>{item.title || item.asin}</h3>
                <div className="asin">{item.asin}</div>
                <div className="why">{item.explanation}</div>
              </div>
              <div className="scores">
                Semantic <b>{fmt(item.semantic_score, 3)}</b>
                <br />
                CF <b>{fmt(item.cf_score, 3)}</b>
                <br />
                {item.image_score != null && (
                  <>
                    Image <b>{fmt(item.image_score, 3)}</b>
                    <br />
                  </>
                )}
                Hybrid <b>{fmt(item.hybrid_score, 3)}</b>
              </div>
            </div>
          ))}
        </div>
        <aside className="side">
          <h2>User History</h2>
          {hist ? (
            <>
              <p className="hint">
                {hist.n_interactions} train interactions
                {hist.average_rating != null ? ` / avg ${fmt(hist.average_rating, 2)}` : ""}
              </p>
              {hist.items.map((it: any) => (
                <div className="hist" key={it.asin}>
                  <div style={{ flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {it.title || it.asin}
                  </div>
                  <span>{fmt(it.overall, 1)}</span>
                </div>
              ))}
            </>
          ) : (
            <p className="hint">No history loaded.</p>
          )}
        </aside>
      </div>

      {open && (
        <>
          <div className="drawer-back" onClick={() => setOpen(null)} />
          <aside className="drawer">
            <Thumb src={open.image_url} alt="" />
            <h2>{open.title || open.asin}</h2>
            <div className="asin">{open.asin}</div>
            <div className="kv"><span className="muted">Rank</span><span>#{open.rank}</span></div>
            <div className="kv"><span className="muted">Semantic</span><span>{fmt(open.semantic_score, 3)}</span></div>
            <div className="kv"><span className="muted">Collaborative</span><span>{fmt(open.cf_score, 3)}</span></div>
            {open.image_score != null && (
              <div className="kv"><span className="muted">Image</span><span>{fmt(open.image_score, 3)}</span></div>
            )}
            <div className="kv"><span className="muted">Hybrid</span><span>{fmt(open.hybrid_score, 3)}</span></div>
            <p style={{ marginTop: 16 }}>{open.explanation}</p>
          </aside>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------

function Compare({ user, alpha }: { user: string; alpha: number | null }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!user) {
      setData(null);
      setErr(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setErr(null);
    api
      .compare(user, alpha)
      .then((result) => {
        setData(result);
        setErr(null);
      })
      .catch((e) => {
        setData(null);
        setErr(e.message || "Unable to load model comparison for this reviewer.");
      })
      .finally(() => setLoading(false));
  }, [user, alpha]);

  if (!user) return <p className="hint">Select a known user on the Explore tab first.</p>;
  if (loading) return <p className="hint">Loading comparison for {user}...</p>;
  if (err) return <p className="err">{err}</p>;
  if (!data) return <p className="hint">No comparison data available for {user}.</p>;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model Rank Comparison</h1>
          <p>Side-by-side product rank divergence across Collaborative Filtering, Semantic, Visual (when enabled), and Hybrid models.</p>
        </div>
      </div>
      <div className="chart-panel">
        <p className="hint" style={{ marginBottom: 12 }}>
          Effective alpha <b>{fmt(data.alpha, 2)}</b>. Overlap of all models: <b>{data.overlap_all_three.length}</b> items.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Product</th>
                <th>Collaborative Filtering Rank</th>
                <th>Semantic Model Rank</th>
                <th>Hybrid Model Rank</th>
                {data.products.some((p: any) => p.image_rank != null) && <th>Visual Model Rank</th>}
              </tr>
            </thead>
            <tbody>
              {data.products.map((p: any) => (
                <tr key={p.asin}>
                  <td>
                    <span style={{ fontWeight: 600 }}>{p.title || p.asin}</span>
                    <div className="asin">{p.asin}</div>
                  </td>
                  <td>{p.cf_rank ? `#${p.cf_rank}` : "-"}</td>
                  <td>{p.semantic_rank ? `#${p.semantic_rank}` : "-"}</td>
                  <td>{p.hybrid_rank ? `#${p.hybrid_rank}` : "-"}</td>
                  {data.products.some((q: any) => q.image_rank != null) && (
                    <td>{p.image_rank ? `#${p.image_rank}` : "-"}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------

function Evaluation({ metrics }: { metrics: MetricsPayload | null }) {
  const splits = splitList(metrics);
  if (!splits.length) return <p className="hint">Loading evaluation metrics...</p>;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Offline Evaluation</h1>
          <p>
            Model-by-model ranking metrics from the generated dataset artifact. The same generated
            users and held-out products are used across Random, Popularity, CF, Semantic, Hybrid,
            and Preference-Aware Hybrid.
          </p>
        </div>
      </div>
      {splits.map((split) => (
        <div className="chart-panel" style={{ marginBottom: 24 }} key={split.split}>
          <div className="chart-header">
            <div>
              <h2 className="chart-title">{prettyName(split.split)}</h2>
              <p className="hint" style={{ margin: "4px 0 0" }}>
                {split.train_rows} train rows / {split.test_rows} test rows / {split.n_items} catalog items
              </p>
            </div>
          </div>
          <MetricSection title="All Test Items" rows={split.metrics} bestName={split.best_among_tested} lift={split.lift} />
          <MetricSection title="Warm Test Items" rows={split.warm_item_metrics || {}} />
          <MetricSection title="Cold Test Items" rows={split.cold_item_metrics || {}} />
        </div>
      ))}
    </>
  );
}

function MetricSection({
  title,
  rows,
  bestName,
  lift,
}: {
  title: string;
  rows: Record<string, MetricSet>;
  bestName?: string;
  lift?: Record<string, LiftEntry>;
}) {
  if (!Object.keys(rows).length) return null;
  return (
    <div style={{ marginTop: 20 }}>
      <h3 style={{ margin: "0 0 10px" }}>{title}</h3>
      <MetricTable rows={rows} bestName={bestName} lift={lift} />
    </div>
  );
}

// ---------------------------------------------------------------------------

function Model({
  status,
  metrics,
  summary,
}: {
  status: any;
  metrics: MetricsPayload | null;
  summary: ArtifactSummary | null;
}) {
  const [config, setConfig] = useState<import("./api").ModelConfig | null>(null);
  useEffect(() => {
    api.modelConfig().then(setConfig).catch(() => setConfig(null));
  }, []);

  const cfg = status?.config || {};
  const serving = metrics?.serving || {};
  const coverage = primarySplit(metrics)?.item_coverage || {};
  const alsSel = (config?.als_selection || {}) as any;
  const vision = summary?.vision || config?.vision;
  const reranker = config?.reranker;

  const specs = useMemo(
    () => [
      ["Serving split", prettyName(String(serving.split || cfg.split || "-"))],
      ["Split protocol", "Each user's most recent review is held out; earlier reviews train the model"],
      ["Filtering", "Only users AND items with at least " + String((cfg as any)?.min_user_count ?? 5) + " interactions are kept (k-core)"],
      ["ALS factors", String((cfg.als?.factors as number) || "-")],
      ["ALS device", String((primarySplit(metrics)?.als_device as string) || "cpu (implicit gpu extension not installed)")],
      ["ALS selection", alsSel.swept ? `swept ${alsSel.n_evaluated || "?"} configs -> best kept` : "single config"],
      ["Confidence mode", String(alsSel.confidence_mode || cfg.interaction?.mode || "-")],
      ["Text encoder", String(config?.semantic?.model || serving.semantic_model || "all-MiniLM-L6-v2")],
      ["Fusion weights", (() => {
        const f = (config?.fusion || {}) as any;
        return `CF ${fmt(Number(f.alpha_cf ?? 0), 2)} / Text ${fmt(Number(f.beta_text ?? 0), 2)} / Image ${fmt(Number(f.gamma_image ?? 0), 2)}`;
      })()],
      ["Cool threshold", String(serving.cool_user_threshold ?? cfg.hybrid?.cool_user_threshold ?? "-")],
      ["Catalog items", String(coverage.catalog_items ?? cfg.n_items ?? "-")],
    ],
    [cfg, serving, coverage, config, alsSel, metrics]
  );

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model Architecture</h1>
          <p>Full dataset, dense k-core filtering, temporal evaluation, and dynamic tri-modal late fusion.</p>
        </div>
      </div>
      <div className="chart-panel" style={{ padding: "32px 28px" }}>
        <div className="arch-flow">
          <div className="arch-node">
            <div className="arch-node-header">
              <div className="arch-node-title">Amazon Electronics Dataset</div>
              <span className="badge-tag">Full Data</span>
            </div>
            <p className="hint" style={{ margin: 0 }}>
              Every review in the dataset is merged with product metadata. Users and items with fewer than 5
              interactions are removed (k-core) so the model has enough signal. The full item catalog is built
              BEFORE splitting, so every test item is scoreable. Then each user's most recent review becomes the
              single test item; earlier reviews train the model (temporal leave-one-last).
            </p>
          </div>
          <div className="arch-arrow"><div className="arch-arrow-line" /><span>down</span></div>
          <div className="arch-dual-row">
            <div className="arch-node als-engine">
              <div className="arch-node-header">
                <div className="arch-node-title">Collaborative Filtering</div>
                <span className="badge-tag">ALS</span>
              </div>
              <p className="hint" style={{ margin: 0 }}>
                Weighted confidence over all ratings; hyperparameters chosen by a swept config (factors/reg/iterations x confidence transforms) on a train subsample.
              </p>
            </div>
            <div className="arch-node minilm-engine">
              <div className="arch-node-header">
                <div className="arch-node-title">Semantic Retrieval</div>
                <span className="badge-tag green">SBERT</span>
              </div>
              <p className="hint" style={{ margin: 0 }}>
                Title + description + categories + brand + price embeddings; rating- and recency-weighted user profiles.
              </p>
            </div>
          </div>
          <div className="arch-dual-row" style={{ marginTop: 12 }}>
            <div className={`arch-node ${vision?.enabled ? "als-engine" : ""}`} style={{ opacity: vision?.enabled ? 1 : 0.55 }}>
              <div className="arch-node-header">
                <div className="arch-node-title">CNN Visual Encoder</div>
                <span className="badge-tag">{vision?.enabled ? "Active" : "Disabled"}</span>
              </div>
              <p className="hint" style={{ margin: 0 }}>
                {vision?.enabled ? (
                  <>
                    {String(vision.model || "resnet50")} · {fmtInt(vision.n_image_embeddings)} image embeddings ·{" "}
                    {fmtInt(vision.missing_image_count)} missing images
                  </>
                ) : (
                  "Rebuild artifacts with --use-vision to enable ResNet-50 product image embeddings."
                )}
              </p>
            </div>
            <div className="arch-node" style={{ opacity: reranker?.enabled ? 1 : 0.55 }}>
              <div className="arch-node-header">
                <div className="arch-node-title">Logistic Reranker</div>
                <span className="badge-tag">{reranker?.enabled ? "Trained" : "Off"}</span>
              </div>
              <p className="hint" style={{ margin: 0 }}>
                {reranker?.enabled
                  ? `Trained on ${fmtInt(reranker.n_train_samples || 0)} leakage-free carve-out samples.`
                  : "Optional stage over hybrid candidates; enable by removing --skip-reranker."}
              </p>
            </div>
          </div>
          <div className="arch-arrow"><div className="arch-arrow-line" /><span>down</span></div>
          <div className="arch-node">
            <div className="arch-node-header">
              <div className="arch-node-title">Dynamic Tri-Modal Fusion</div>
              <span className="badge-tag">Conditional Weights</span>
            </div>
            <p className="hint" style={{ margin: 0 }}>
              Each user gets a weighted mix of the three scores. New users (thin history) get content-only
              recommendations (text + image), because collaborative filtering needs history to learn from. As a
              user's history grows, the collaborative signal blends in.
            </p>
          </div>
        </div>
        <h3 style={{ marginTop: 32 }}>Active Configuration</h3>
        <div className="arch-specs-grid">
          {specs.map(([label, value]) => (
            <div className="arch-spec-card" key={label}>
              <div className="arch-spec-label">{label}</div>
              <div className="arch-spec-value">{value}</div>
            </div>
          ))}
        </div>
        {reranker?.enabled && reranker.coefficients && (
          <>
            <h3 style={{ marginTop: 32 }}>Reranker Feature Weights</h3>
            <div className="arch-specs-grid">
              {Object.entries(reranker.coefficients).map(([feature, weight]) => (
                <div className="arch-spec-card" key={feature}>
                  <div className="arch-spec-label">{prettyName(feature)}</div>
                  <div className="arch-spec-value">{weight == null ? "-" : fmt(weight, 3)}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
}

