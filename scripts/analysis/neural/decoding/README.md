# Decoding analysis


## Usage of `decoding_batch.sh`
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


## Plotting decoding scores and generalisation matrices

- `plot_decoding_diagonal.py` for plotting diagonal scores.
- `plot_decoding_timegen.py` for plotting generalisation matrices.
- `plot_decoding.py` is a config/utils file.