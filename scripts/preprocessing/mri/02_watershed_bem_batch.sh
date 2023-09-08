#!/bin/bash

# Set the partition and other SBATCH specifications
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=batch_recon
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/job_error.log


# read subject_ids from config
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subject_ids=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))

# set the path to the recon script
watershed_script="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/watershed_bem.sh"

# loop through each subject ID and submit a job
for subject_id in "${subject_ids[@]}"; do
    bem_dir="/imaging/hauk/rl05/fake_diamond/data/mri/sub-$subject_id/bem"
    if [ ! -d "$bem_dir" ]; then
        echo "sub-$subject_id BEM directory missing. Processing."
        sbatch --export=subject_id="$subject_id",watershed_script="$watershed_script" watershed_bem_job.sh
    else
        echo "sub-$subject_id BEM directory exists. Skipping."
    fi
done