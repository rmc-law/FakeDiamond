#!/bin/bash

# Set the partition and other SBATCH specifications
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=recon_all_batch
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/job_error.log


# Read subject_ids from the Python script
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subject_ids=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))

# Set the path to the recon script
recon_script="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/recon_all.sh"

# Loop through each subject ID and submit a job
for subject_id in "${subject_ids[@]}"; do
    echo "sub-$subject_id:"
    sbatch --export=subject_id="$subject_id",recon_script="$recon_script" submit_subject_recon_job.sh
done
