#!/bin/bash

# Set the partition and other SBATCH specifications for individual subject jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=recon_all_subject
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/%j_subject_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/job_log/%j_subject_error.log


# Set the subject ID passed as an argument
subject_id="$subject_id"
echo "sub-$subject_id"

recon_script="$recon_script" 

# Run the main recon script for the specific subject
bash "$recon_script" "$subject_id"
