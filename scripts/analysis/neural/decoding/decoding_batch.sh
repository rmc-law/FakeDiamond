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

# read in subjects 
script_dir="/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/"
subjects=($(python -c "import sys; sys.path.append('$script_dir'); from config import subject_ids; print(' '.join(subject_ids))"))


# loop over subjects
for subject in "${subjects[@]}"; do

    # check if the particular decoding analysis is done for a subject
    subject_analysis_output_dir="/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output/${analysis}/${classifier}}/${data_type}/sub-${subject}"
    
    if [ ! -e "$subject_analysis_output_dir" ]; then
        echo "Output of $analysis decoding does not exist for sub-$subject. Decoding."
        sbatch --export=subject="$subject",analysis="$analysis",classifier="$classifier",data_type="$data_type" decoding_job.sh
    else
        echo "Output of $analysis decoding exists for sub-$subject."
    fi

done