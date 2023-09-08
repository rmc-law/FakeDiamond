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


# run watershed bem using mne to make bem meshes
input_dir="$SUBJECTS_DIR/$subject"
output_dir="$SUBJECTS_DIR/$subject/bem"
log_dir="$SUBJECTS_DIR/logs/$subject"

if [ -d "$output_dir" ]; then
    echo "BEM directory exists for $subject."
else

    log_file="$log_dir/watershed_bem.log"

    echo "Running watershed_bem for $subject."
    mne watershed_bem -s $subject -o "$output_dir" > "$log_file" 2>&1

    # check if watershed_bem is completed succesfully
    if [ $? -eq 0 ]; then
        echo "watershed_bem done for $subject."
    else
        echo "watershed_bem failed for $subject. See $log_file for details."
    fi
fi


