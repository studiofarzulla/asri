/*
 * ASRI vs. Diebold–Yilmaz (2012) connectedness — head-to-head classification benchmark.
 *
 * Source of record: asri/arxiv-submission/ASRI_Paper_Canon.tex
 *   - Table `tab:roc_metrics` and §`subsec:dy_comparison` (also stated in the abstract).
 *   - Day-level binary classification (1,402 daily observations; positive class =
 *     a crisis occurs within the 30-day forward window), scored against the same
 *     crisis labels. D-Y = Diebold–Yilmaz generalized-FEVD connectedness via a
 *     60-day rolling VAR(1), H = 10.
 *
 * Reported figures (DO NOT edit without re-checking the canon .tex — verified 2026-06-27):
 *   AUROC             ASRI 0.866 [0.844, 0.885]   vs   D-Y 0.670 [0.640, 0.701]
 *   AUPRC             ASRI 0.298                   vs   D-Y 0.121
 *   Precision@Youden  ASRI 35.2%                   vs   D-Y 14.9%
 *   (Recall: ASRI 75.8% vs D-Y 80.6% — D-Y alerts more often at a far higher
 *    false-alarm rate, so its precision and F1 fall well below ASRI's.)
 *
 * Binding caveat (from §subsec:limitations): every figure ultimately rests on only
 * FOUR systemic crisis events; bootstrap CIs do not overcome that power ceiling.
 */

export type BenchmarkFormat = "ratio" | "percent";

export interface BenchmarkMetric {
  key: string;
  label: string;
  description: string;
  /** ASRI value in native units (ratio in [0,1] or percent in [0,100]). */
  asri: number;
  /** Diebold–Yilmaz value in native units. */
  dy: number;
  format: BenchmarkFormat;
  /** 95% bootstrap CI for the ASRI figure, where reported in the canon. */
  asriCi?: [number, number];
  dyCi?: [number, number];
}

export const BENCHMARK_METRICS: BenchmarkMetric[] = [
  {
    key: "auroc",
    label: "AUROC",
    description: "Day-level discrimination (95% bootstrap CIs do not overlap)",
    asri: 0.866,
    dy: 0.67,
    format: "ratio",
    asriCi: [0.844, 0.885],
    dyCi: [0.64, 0.701],
  },
  {
    key: "auprc",
    label: "AUPRC",
    description: "Precision–recall area under ~9% crisis-day class imbalance",
    asri: 0.298,
    dy: 0.121,
    format: "ratio",
  },
  {
    key: "precision",
    label: "Precision @ Youden",
    description: "Precision at the Youden-J optimal threshold",
    asri: 35.2,
    dy: 14.9,
    format: "percent",
  },
];

export const BENCHMARK_CAVEAT =
  "Day-level classification (1,402 obs; positive = crisis within 30 days). All figures rest on four systemic crisis events — the binding statistical-power limit of the study.";
