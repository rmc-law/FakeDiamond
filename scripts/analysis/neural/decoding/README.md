# Multivariate Decoding & Statistics (Python + R)
This module contains the code for the time-resolved decoding and temporal generalization analyses (Figure 5 and Figure 6). The pipeline involves training classifiers on the cluster (HPC), computing cluster-based statistics in Python, and performing interaction analyses using Linear Mixed Effects Models (LMEM) in R.

## Execution Order
1. Run decoding (HPC Cluster)
    - `decoding.py`: The core script. Trains L2-regularized Logistic Regression classifiers (scikit-learn) on source-localized data.
    - `decoding_batch.sh` & `decoding_job.sh`: Submits the decoding jobs for all subjects and time points.
    > Usage: sbatch decoding_batch.sh (adjust for your scheduler).

2. Statistical Inference (Python)
    - `stats_decoding_diagonal.py`: Performs cluster-based permutation tests on the diagonal (time-resolved decoding).
    - `stats_decoding_timegen.py`: Performs stats on the full temporal generalization matrix (off-diagonal stability).
    - `stats_corr_beh-decod.py`: Correlates decoding performance with behavioral reaction times.

3. Interaction Analysis (Python ➔ R)
    - To test the "Phrase Type × Time Window" interaction reported in Figure 6E and Figure 6F, we use a hybrid workflow:
    - Export data: Run `stats_interaction_save_csv.py` to extract mean AUC scores from "early" (300-500ms) and "late" (500-800ms) windows and save them as CSVs.
    - Run LMEM: Run `stats_interaction_test.R` to fit the Linear Mixed Effects Models (lme4) using the exported CSVs.
    > Note: This script uses renv to ensure the exact R package versions are used.

4. Generate Figures
    - `fig5_denotation+concreteness.py`: Plots time-courses and generalization matrices for single-word/phrase decoding.
    - `fig6_concreteness_xcond.py`: Plots the cross-decoding results (training on words, testing on phrases).


### Usage of `decoding_batch.sh`
`./decoding_batch.sh` runs a parallel job for each subject separately. To run the batch script, you need to supply the desired analysis and classifier.

Usage:
```
./decoding_batch concreteness logistic ROI single micro_ave generalise
```

Possible `analysis_name` options: 
- lexicality: whether word 1 is word vs. letter-string
- composition: whether word 2 is a single noun vs. in a phrase
- concreteness: whether word 2 is concrete vs. abstract
- denotation: whether word 2 is preceded by subsective vs. privative
- specificity
- specificity_word
- denotation_cross_condition

Possible `classifier_name` options:
- logistic
- svc
- naive_bayes

`data_type` options:
- MEEG
- MEG
- ROI (decoding in each ROI separately)

`window` (single, sliding): `single` performs decoding at each time point; `sliding` takes a sliding-window approach with temporal smoothing.

`micro_ave`: use micro-averaging to create pseudotrials to increase SNR (recommended)

`generalise`: whether to perform temporal generalisation analyses

The batch script will then check if that analysis is done for all subjects, and run the analysis if not. 