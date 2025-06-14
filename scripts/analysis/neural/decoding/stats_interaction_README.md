# Decoding Concreteness: Mixed-Effects Analysis in ROI Time Windows

This repository contains an R script for analyzing concreteness decoding scores in early and late time windows across two types of phrases (privative and subsective), using linear mixed-effects models.

The analysis:
- Iterates over specified ROIs
- Loads pre-averaged score data
- Fits appropriate mixed-effects models (by finding appropriate random-effects structure)
- Computes ANOVAs and post-hoc comparisons
- Saves plots and textual summaries

---

## Files

- `stats_interaction_test.R` — main R script
- `renv.lock` — locked dependency file for package versions
- `renv/` — project-local package library created by `{renv}`
- `output/` — folder where result plots and summaries will be saved

## To run the script
```
module load R/4.3.1 
Rscript stats_interaction_test.R
```