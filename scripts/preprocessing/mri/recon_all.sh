#!/bin/bash

# Set input directory and FreeSurfer home directory
module load freesurfer
source "$FREESURFER_HOME/FreeSurferEnv.sh"
echo "Freesurfer home: $FREESURFER_HOME"

conda activate mne1.4.2

subject="sub-$1"

# set subjects_dir
SUBJECTS_DIR="/imaging/hauk/rl05/fake_diamond/data/mri"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"


# run recon-all
input_dir="$SUBJECTS_DIR/T1w/$subject"
output_dir="$SUBJECTS_DIR/$subject"
log_dir="$SUBJECTS_DIR/logs/$subject"

if [ -d "$output_dir" ]; then
    echo "Recon-all appears to have been completed for $subject."
else
    # check if nifti file exists 
    nifti_file=$(find "$input_dir" -name "*nii*")
    if [ -z "$nifti_file" ]; then 
        echo "Nifti files not found. Have you converted dicoms already?"
    else
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
    
        log_file="$log_dir/recon-all.log"
    
        echo "Running recon-all for $subject."
        recon-all -s $subject -i "$nifti_file" -all > "$log_file" 2>&1

        # Check if reconstruction completed successfully
        if [ $? -eq 0 ]; then
            echo "recon-all done for $subject."
        else
            echo "recon-all failed for $subject. See $log_file for details."
        fi
    fi
fi