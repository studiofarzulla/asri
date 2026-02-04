# ASRI Paper - Future Improvements

These are potential enhancements identified during arXiv submission review (Feb 2026) that were deemed too large for the current submission but could strengthen future revisions.

## Benchmark Comparisons Beyond Diebold-Yilmaz

**Current state:** Paper only compares ASRI to Diebold-Yilmaz (2012) connectedness index.

**Suggested additions:**
- TV-DIG (Etesami et al.) - time-varying directed information/causal influence
- Factor + sparse VAR connectedness (Krampe & Margaritella) - decomposes common vs idiosyncratic drivers
- VCoES (Vulnerability Conditional ES) - tail-focused dependence metric
- Wavelet/transfer-entropy network methods - multi-scale information flow

**Why it matters:** Would clarify where ASRI's interpretable composite has advantages/disadvantages relative to causal/tail/network alternatives.

**Effort:** High - requires implementing additional baselines and running full validation suite.

---

## Confidence Intervals for Event Study Lead Times

**Current state:** Table 5 (event study results) shows point estimates for lead times without uncertainty measures. Bootstrap CIs exist in Table 7 but use different methodology.

**Suggested fix:** Add SE or CI columns to event study table, using same methodology as bootstrap analysis for consistency.

**Why it matters:** Allows readers to assess statistical significance of lead time differences across events.

**Effort:** Medium - requires modifying `scripts/run_event_study.py` to output CIs and regenerating table.

---

## Notes

- These were flagged by AI-generated review (Thesify) but verified as genuine gaps
- Other "issues" in the review were false alarms - the paper already addresses them in appendices
- See `arxiv-submission/` for the submission-ready version with fixes applied
