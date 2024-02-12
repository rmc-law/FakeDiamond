# Decoding analysis

## Summary of decoding pipeline
1. `cd /imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding`
2. `./decoding_batch.sh analysis_name classifier_name sensor_type`

### Usage of `decoding_batch.sh`
`./decoding_batch.sh` runs a parallel job for each subject separately. To run the batch script, you need to supply the desired analysis and classifier.

Possible `analysis_name` options: 
- lexicality: whether word 1 is word vs. letter-string
- composition: whether word 2 is a single noun vs. in a phrase
- concreteness: whether word 2 is concrete vs. abstract
- denotation: whether word 2 is preceded by subsective vs. privative
- specificity

Possible `classifier_name` options:
- logistic
- svc
- naive_bayes

`sensor_type` options:
- MEEG
- MEG
- to add: source (for source space decoding)

The batch script will then check if that analysis is done for all subjects, and run the analysis if not. 