#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoding_job
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/%j_decoding_subject_specific_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/%j_decoding_subject_specific_error.log


# Set the subject ID passed as an argument
subject="$subject"
analysis="$analysis" 
classifier="$classifier"

echo "Decoding $analysis for sub-$subject." 

conda activate mne1.4.2

python decoding.py --subject "$subject" --analysis "$analysis" --classifier "$classifier"
