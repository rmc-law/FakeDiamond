#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=decoding_group
#SBATCH --output=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_output.log
#SBATCH --error=/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/job_log/job_error.log


# specify decoding analysis and classifier
analysis="$1"
classifier="$2"
data_type="$3"
generalise="$4"

# read in subjects 
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subjects=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))


# loop over subjects
for subject in "${subjects[@]}"; do

    if [ "$generalise" = "generalise" ]; then
    
        timegen_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/timegen/${classifier}/${data_type}/sub-${subject}"
        
        if [ ! -e "$timegen_output_dir" ]; then
            echo "Timegen output of $analysis does not exist for sub-$subject. Decoding."
            sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type",generalise="$generalise" decoding_job.sh
        else
            echo "Timegen output of $analysis exists for sub-$subject. Skipping."
        fi
        
    else
    
        timedecod_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/diagonal/${classifier}/${data_type}/sub-${subject}"
        
        if [ ! -e "$timedecod_output_dir" ]; then
            echo "Diagonal decod output of $analysis does not exist for sub-$subject. Decoding."
            sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type" decoding_job.sh
        else
            echo "Diagonal decod output of $analysis exists for sub-$subject. Skipping."
        fi
        
    fi
    
done