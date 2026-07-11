/*
 * Fair-baseline AUROC comparison — the paper's HONEST position on discrimination.
 *
 * Source of record: asri/arxiv-submission/ASRI_Paper_Canon.tex
 *   - §`subsec:dy_comparison` / §`subsec:fair_baselines` and the abstract's demotion
 *     of the Diebold–Yilmaz head-to-head.
 *
 * The earlier "ASRI vs. Diebold–Yilmaz — ASRI WINS" framing OVERSTATED the result.
 * The paper explicitly DEMOTES the D-Y comparison. The honest picture:
 *
 *   1. ASRI's day-level discrimination (AUROC 0.866) is NOT a gain over simpler
 *      baselines: a standalone equity-volatility (VIX) series matches/slightly
 *      exceeds it (0.875), and ASRI is statistically indistinguishable from its own
 *      strongest single channel (Contagion Risk, 0.851) and from the first principal
 *      component of its sub-indices (PC1, 0.858).
 *   2. ASRI's only clear margin is over the Diebold–Yilmaz connectedness series
 *      (0.670) — but that is a CIRCULAR comparator (D-Y is constructed from ASRI's
 *      own four sub-indices) and is itself outperformed by an off-the-shelf Crypto
 *      Fear & Greed index (0.789).
 *   3. The value of aggregation is therefore INTERPRETIVE — channel attribution,
 *      lead-time, and regime structure in one auditable, reproducible composite —
 *      NOT superior discrimination.
 *   4. Every figure rests on only FOUR systemic crisis events (1,402 day-level obs):
 *      the binding statistical-power limit of the study.
 *
 * AUROC figures are fixed canonical study results (verified against the canon .tex,
 * 2026-06-30). They are not live-API quantities — safe to hardcode here.
 */

/** Role of a comparator in the fair-baseline picture, used purely for styling/copy. */
export type BaselineKind = "asri" | "external" | "internal" | "circular";

export interface BaselineRow {
  key: string;
  /** Short series name. */
  label: string;
  /** One-line description of what the series is. */
  sublabel: string;
  /** Day-level crisis-classification AUROC (positive = crisis within 30 days). */
  auroc: number;
  kind: BaselineKind;
  /** Short honest tag rendered as a pill on the row. */
  note?: string;
}

/* Ordered by AUROC, descending — so it is visible that ASRI does NOT top the list. */
export const BASELINE_AUROCS: BaselineRow[] = [
  {
    key: "vix",
    label: "VIX",
    sublabel: "Standalone equity-volatility series (off-the-shelf)",
    auroc: 0.875,
    kind: "external",
    note: "tops the list",
  },
  {
    key: "asri",
    label: "ASRI",
    sublabel: "The aggregated composite",
    auroc: 0.866,
    kind: "asri",
    note: "no clear gain over baselines",
  },
  {
    key: "pc1",
    label: "PC1",
    sublabel: "First principal component of ASRI's sub-indices",
    auroc: 0.858,
    kind: "internal",
    note: "tied with ASRI (n.s.)",
  },
  {
    key: "contagion",
    label: "Contagion Risk",
    sublabel: "ASRI's strongest single channel",
    auroc: 0.851,
    kind: "internal",
    note: "tied with ASRI (n.s.)",
  },
  {
    key: "fng",
    label: "Crypto Fear & Greed",
    sublabel: "Off-the-shelf sentiment index",
    auroc: 0.789,
    kind: "external",
    note: "beats D-Y off-the-shelf",
  },
  {
    key: "dy",
    label: "Diebold–Yilmaz",
    sublabel: "Connectedness, built from ASRI's own four sub-indices",
    auroc: 0.67,
    kind: "circular",
    note: "circular comparator",
  },
];

/** AUROC = 0.5 is no-skill; scale bar length to discrimination above chance. */
export const AUROC_CHANCE = 0.5;

export const BENCHMARK_CIRCULARITY_CAVEAT =
  "The Diebold–Yilmaz series is constructed from ASRI's own four sub-indices, so the apparent margin over it is circular rather than an external validation.";

export const BENCHMARK_POWER_CAVEAT =
  "Day-level classification (1,402 obs; positive = crisis within 30 days). Every AUROC ultimately rests on four systemic crisis events — the binding statistical-power limit of the study; bootstrap CIs do not overcome it.";

export const BENCHMARK_INTERPRETIVE_NOTE =
  "ASRI does not beat simple baselines on discrimination. Its contribution is interpretive: channel attribution, lead-time, and regime structure in one auditable, reproducible composite.";
