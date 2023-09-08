#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=maxfilter_job
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/meg/job_log/%j_maxfilter_subject_run_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/meg/job_log/%j_maxfilter_subject_run_error.log


# Set the subject ID passed as an argument
subject="$subject"
run="$run" 

echo "Applying maxwell filter to sub-$subject run-$run raw." 

conda activate mne1.4.2
# Run the main recon script for the specific subject
python maxfilter.py "$subject" "$run"
