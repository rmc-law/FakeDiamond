# Univariate ROI analysis
This module performs the mass-univariate Region of Interest (ROI) analysis reported in Figure 3 and Figure 4. It utilizes the Eelbrain toolkit to compute temporal cluster-based permutation tests on source-localized activity.

## Execution Order
1. Statistical tests
    - `eelbrain_permutation_temporal_biATL.py`: Runs the primary 2x2x2 mixed-effects ANOVA (Composition × Concreteness × Hemisphere) in the bilateral ATLs.
    - `eelbrain_permutation_temporal.py`: Runs the analyses for secondary ROIs (IFC, PTL) and the denotation contrasts.

> Note: These scripts save the statistical results (clusters and p-values) to disk for plotting.

2. Generate gigures:
    - `fig3_composition_biATL.py`: Generates the time-course plots for the ATL composition effect (Fig 3).
    - `fig4_denotation.py`: Generates the time-course plots for the subsective vs. privative contrast (Fig 4).
    - figSupp*_*.py: Generates the supplementary time-courses and Bayes Factor evidence plots.

Notes:
- Statistical framework: Mass-univariate mixed-effects ANOVAs were fitted at each time point.
- Correction: Significance was estimated using cluster-based permutation tests (5,000 randomisations).
- Bayes Factors: Evidence for the null vs. alternative hypothesis was quantified using Bayes Factors (BF10).
- Configuration: The file `fake_diamond.rc` handles environment variables and resource paths used across these scripts.