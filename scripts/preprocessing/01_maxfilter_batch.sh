#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=maxfilter_batch
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/meg/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/meg/job_log/job_error.log


# read in subjects 
subjects=($(python -c "from config import subject_ids; print(' '.join(subject_ids))"))

# loop over subjects
for subject in "${subjects[@]}"; do
    
    # loop over experiment runs for the current subject
    for run in {1..5}; do
        
        # check if maxwell filtered raw already exists
        preprocessed_data_path="/imaging/hauk/rl05/fake_diamond/data/preprocessed"  # Replace with the actual path
        raw_dir="/imaging/hauk/rl05/fake_diamond/data/raw/sub-${subject}"
        raw_sss_fname="$preprocessed_data_path/sub-${subject}/maxfiltered/run-${run}/run${run}_sss_raw.fif"
        
        if [ -d "$raw_dir" ]; then
            if [ ! -e "$raw_sss_fname" ]; then
                echo "Maxwell filtered data do not exist. Processing sub-$subject run-$run."
                sbatch --export=subject="$subject",run="$run" maxfilter_job.sh
            else
                echo "Maxwell filtered data exist for sub-$subject run-$run. Skipped."
            fi
        else
            echo "Raw directory does not exist for sub-$subject. Skipped."
        fi
        
    done
    
done
