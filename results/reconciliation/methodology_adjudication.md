## Methodology Adjudication (2026-02-14)

Canonical runtime profile: `paper_v2`.

Why this profile is selected:
- Reproduces current paper event-study table exactly on the same local ASRI series.
- Preserves 4/4 significant events with coherent peak/CAS trajectories.
- Uses explicit and bounded lead-time measurement (`max_lookback=30`) consistent with the reported 30-day pre-crisis warning framing.

Profile parameters:
- Estimation window: `(-90, -31)`
- Event window: `(-30, +10)`
- Significance level: `alpha=0.05`
- Lead method: `first_sigma_breach`
- Lead sigma threshold: `1.5`
- Lead lookback: `30` days

Reconciliation artifacts:
- `results/reconciliation/event_study_method_comparison.json`
- `results/reconciliation/event_study_primary.json`
- `results/tables/event_study_comparison.tex`
- `results/tables/event_study.tex`

Counterfactual profile retained for diagnostics:
- `counterfactual_60d` (60-day estimation, sustained-threshold lead definition).
