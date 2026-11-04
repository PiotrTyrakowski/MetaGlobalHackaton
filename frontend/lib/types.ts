export type Suggestion = {
  column: string;
  level: "batch" | "recipe";
  kind: string;
  coverage: number;
  verdict: "use" | "collect" | "proxy" | "reject" | "not_tested";
  p?: number;
  strength?: number;
  scatter_explained?: number;
  pct_strength_per_unit?: number;
  levels?: number;
  spread_pct?: number;
  error_without_psi?: number;
  error_with_psi?: number;
  error_shuffled_psi?: number;
  gain?: number;
  batches_with_value?: number;
  batches_total?: number;
  missing_by?: { column: string; missing_values: string[]; batches_affected: number } | null;
  associated_with?: { column: string; how: string }[];
  reason?: string;
};

export type Suggestions = { use: string[]; collect: string[]; columns: Suggestion[]; method: Record<string, string> };

export type Validation = {
  summary: {
    recipes: number;
    batches: number;
    strength_results: number;
    cylinders: number;
    ages_days: number[];
    batches_per_recipe: [number, number];
    strength_28d_psi: [number, number] | null;
  };
  boxcrete_inputs: {
    input: string;
    source: string | null;
    coverage: number;
    plant_range: [number, number];
    meta_range: [number, number];
    outside_meta_range: number;
  }[];
  candidate_columns: { column: string; level: string; kind: string; file: string; coverage: number }[];
  issues: { level: "info" | "warning" | "error"; message: string }[];
};

export type DataArtifact = { validation: Validation; suggestions: Suggestions | null; mapping: Record<string, unknown> };

export type Settings = { gate_tau: number; shrinkage_sigma: number; mll_weight: number };

export type LeaderRow = {
  name: string;
  inputs: string[];
  settings: Settings;
  tuned: boolean;
  error_28d_psi: number;
  errors_by_age_psi: Record<string, number>;
  error_all_ages: number;
  error_all_ages_pct: number;
  folds_won_vs_baseline: string;
  folds_won_vs_meta_settings?: string;
  baseline: boolean;
  best: boolean;
  vs_baseline: number;
  vs_baseline_28d: number;
};

export type Training = {
  leaderboard: LeaderRow[];
  winner: { name: string; inputs: string[]; settings: Settings; error_28d_psi: number; vs_baseline: number; vs_baseline_28d: number };
  meta_default: Settings;
  tuning_helped: boolean;
  trials: { setup: string; source: string; error_all_ages: number | null; error_28d_psi: number | null }[];
  method: Record<string, unknown>;
  seconds: number;
  finished?: string;
};

export type Proposal = {
  id: string;
  replaces: string;
  co2_saving: number;
  gwp_kg_m3: number;
  recipe_gwp_kg_m3: number;
  target_28d_psi: number;
  predicted_28d_psi: number;
  sd_28d_psi: number;
  lower90_28d_psi: number;
  predicted_7d_psi: number;
  composition: Record<string, number>;
  composition_unit: string;
  extras: Record<string, number>;
};

export type Results = {
  model: string;
  recipes: { recipe: string; gwp_kg_m3: number; observed_28d_psi: number | null; predicted_28d_psi: number }[];
  proposals: Proposal[];
  method: { candidates: number; rule: string; co2: string };
};

export type ModelMeta = {
  version: string;
  trained: string;
  inputs: string[];
  setup: string;
  error_28d_psi: number;
  vs_meta_default: number;
  recipes: number;
  batches: number;
  active: boolean;
};
