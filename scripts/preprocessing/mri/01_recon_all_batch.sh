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
recon_script="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/recon_all.sh"

# loop through each subject ID and submit a job
for subject_id in "${subject_ids[@]}"; do

    recon_dir="/imaging/hauk/rl05/fake_diamond/data/mri/sub-$subject_id/"
    if [ ! -d "$recon_dir" ]; then
        echo "sub-$subject_id MRI directory missing. Processing."
        sbatch --export=subject_id="$subject_id",recon_script="$recon_script" recon_all_job.sh
    else
        echo "sub-$subject_id MRI directory exists. Skipping."
    fi
done
