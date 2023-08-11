#!/bin/bash

# Set input directory and FreeSurfer home directory
module load freesurfer
source "$FREESURFER_HOME/FreeSurferEnv.sh"
echo "Freesurfer home: $FREESURFER_HOME"

conda activate mne1.4.2

# Function to run recon-all and redirect output to a log file
run_recon_all() {
    local subject_id="$1"
    local input_dir="$2"
    local output_dir="$3"
    local log_dir="$input_dir/log"
    
    # check if nifti file exists 
    nifti_file=$(find "$input_dir" -name "*nii*")
    if [ -z "$nifti_file" ]; then 
        echo "Nifti files not found. Have you converted dicoms already?"
    else
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
    
        log_file="$log_dir/recon-all.log"
    
        if [ -d "$output_dir" ]; then
            echo "Output directory already contains recon-all results for $subject_id."
        else
            echo "Starting cortical surface reconstruction for $subject_id..."
            recon-all -s recon -i "$nifti_file" -all > "$log_file" 2>&1
    
            # Check if reconstruction completed successfully
            if [ $? -eq 0 ]; then
                echo "Cortical surface reconstruction completed for $subject_id."
                
                # Run mne make_scalp_surfaces
                mne make_scalp_surfaces -s recon -o "$output_dir" > "$mne_scalp_log" 2>&1
    
                # Run mne watershed_bem
                mne watershed_bem -s recon -o "$output_dir" > "$mne_watershed_log" 2>&1
    
                echo "BEM surfaces generated for $subject_id."
    
            else
                echo "Cortical surface reconstruction failed for $subject_id. See $log_file for details."
            fi
        fi
    fi
}

subject_id="$1"

# set subjects_dir
SUBJECTS_DIR="/imaging/hauk/rl05/fake_diamond/data/sub-$subject_id/anat"
echo "$SUBJECTS_DIR"

# set output_dir
input_dir="$SUBJECTS_DIR"
output_dir="$SUBJECTS_DIR/recon"

# check if recon files already exists
if [ -d "$output_dir" ]; then
    echo "Output directory already contains recon-all results for $subject_id."
else
    # call the run_recon_all function
    run_recon_all "$subject_id" "$input_dir" "$output_dir"
