# Behavioural analysis
R scripts for analysing reaction times and accuracy using mixed-effects models.  

### Reproducing the R environment:
This project uses `renv`. To install all required R packages:
1. Open R in this project folder.
2. Install renv: `install.packages("renv")`
3. Restore the environment: `renv::restore()`

### Execution order:
1. `1_group_data.R`: Aggregates individual subject logs.
2. `2_generate_summaries.R`: Produces descriptive statistics.
3. `3_stats_accuracy.R`: Fits the binomial GLMM for accuracy.
4. `4_stats_rt.R`: Fits the LMM for log-transformed RTs.

### To run the scripts
```
# Load the required R module
module load R/4.3.1 # or whatever version you have

# Execute the script
Rscript 1_group_data.R
Rscript 2_generate_summaries.R
Rscript 3_stats_accuracy.R
Rscript 4_stats_rt.R
```