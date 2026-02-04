# ASRI arXiv Submission Metadata

## Title
```
ASRI: An Aggregated Systemic Risk Index for Cryptocurrency Markets
```

## Authors (in order)
```
Murad Farzulla, Andrew Maksakov
```

## Author Affiliations
| Author | Affiliations |
|--------|--------------|
| Murad Farzulla | King's College London; Dissensus AI |
| Andrew Maksakov | Dissensus AI |

## Corresponding Author
```
Murad Farzulla <murad@dissensus.ai>
```

## Abstract
```
Cryptocurrency markets have grown to represent over $3 trillion in capitalization, yet no unified index exists to monitor the systemic risks arising from the interconnection between decentralized finance (DeFi) protocols and traditional financial institutions. This paper introduces the Aggregated Systemic Risk Index (ASRI), a composite measure comprising four weighted sub-indices: Stablecoin Concentration Risk (30%), DeFi Liquidity Risk (25%), Contagion Risk (25%), and Regulatory Opacity Risk (20%). We derive theoretical foundations for each component, specify quantitative formulas incorporating data from DeFi Llama, Federal Reserve FRED, and on-chain analytics, and validate the framework against historical crisis events including the Terra/Luna collapse (May 2022), the Celsius/3AC contagion (June 2022), the FTX bankruptcy (November 2022), and the SVB banking crisis (March 2023). Event study analysis detects statistically significant abnormal signals for all four crises (t-statistics 5.47-32.64, all p < 0.01), though threshold-based operational detection identifies three of four events with an average lead time of 18 days. A three-regime Hidden Markov Model identifies distinct Low Risk, Moderate, and Elevated states with regime persistence exceeding 94%. Out-of-sample specificity testing on 2024-2025 data confirms zero false positives. The ASRI framework addresses a critical gap in existing risk monitoring by capturing DeFi-specific vulnerabilities---composability risk, flash loan exposure, and tokenized real-world asset linkages---that traditional systemic risk measures cannot accommodate.
```

---

## arXiv Categories

### Primary Category
```
q-fin.CP
```
**Quantitative Finance - Computational Finance**

### Cross-List
```
q-fin.GN
```
**Quantitative Finance - General Finance**

---

## ACM Computing Classification System (CCS 2012)

### Primary Class
```
J.4 [Social and Behavioral Sciences]: Economics
```

### Secondary Classes
```
G.3 [Probability and Statistics]: Time series analysis
H.4.2 [Information Systems Applications]: Types of Systems—Decision support
I.6.4 [Simulation and Modeling]: Model Validation and Analysis
```

### ACM CCS Concepts (new format)
```
• Applied computing → Economics
• Mathematics of computing → Time series analysis
• Information systems → Decision support systems
• Computing methodologies → Model verification and validation
```

### Full ACM CCS String (for LaTeX)
```latex
\begin{CCSXML}
<ccs2012>
<concept>
<concept_id>10010405.10010489</concept_id>
<concept_desc>Applied computing~Economics</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003648.10003671</concept_id>
<concept_desc>Mathematics of computing~Time series analysis</concept_desc>
<concept_significance>300</concept_significance>
</concept>
<concept>
<concept_id>10002951.10003317.10003347.10003350</concept_id>
<concept_desc>Information systems~Decision support systems</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
\end{CCSXML}
```

---

## MSC (Mathematics Subject Classification) 2020

```
91G40, 91G70, 91B84, 62M10
```

| Code | Description |
|------|-------------|
| 91G40 | Credit risk |
| 91G70 | Statistical methods in finance |
| 91B84 | Economic time series analysis |
| 62M10 | Time series, auto-correlation, regression |

---

## JEL Classification (Economics)

```
G01, G15, G23, G28
```

| Code | Description |
|------|-------------|
| G01 | Financial Crises |
| G15 | International Financial Markets |
| G23 | Non-bank Financial Institutions; Financial Instruments; Institutional Investors |
| G28 | Government Policy and Regulation |

---

## Keywords
```
systemic risk, cryptocurrency, decentralized finance, stablecoin stability, contagion risk, DeFi-TradFi interconnection, risk monitoring, event study, regime detection, Hidden Markov Model
```

---

## Comments Field
```
79 pages, 6 figures, 9 tables. Live dashboard: https://asri.dissensus.ai. Code: https://github.com/studiofarzulla/asri
```

---

## Journal Reference (optional)
```
Preprint. Previously published on Zenodo: DOI 10.5281/zenodo.17918239
```

---

## Report Number (optional)
```
DISSENSUS-2026-001
```

---

## DOI (existing)
```
10.5281/zenodo.17918239
```

---

## License
```
CC BY 4.0 (Creative Commons Attribution 4.0 International)
```

---

## Submission Checklist

- [x] Tarball created: `ASRI_arxiv.tar.gz` (247KB)
- [x] Main tex file: `ASRI_Paper.tex`
- [x] Bibliography: `ASRI_Paper.bbl` + `references.bib`
- [x] Figures: 5 PDFs in `figures/`
- [x] Tables: 9 tex files in `tables/`
- [x] Compiles cleanly (79 pages)
- [x] No undefined references
- [x] No undefined citations

---

## Quick Copy-Paste for arXiv Form

**Title:**
ASRI: An Aggregated Systemic Risk Index for Cryptocurrency Markets

**Authors:**
Murad Farzulla, Andrew Maksakov

**Primary category:**
q-fin.CP

**Cross-list:**
q-fin.GN

**MSC:**
91G40; 91G70; 91B84

**ACM:**
J.4; G.3; H.4.2

**Comments:**
79 pages, 6 figures, 9 tables. Live dashboard: https://asri.dissensus.ai
