#!/bin/bash
# shrink inner skull if BEM surfaces touch
# usage: shrink_innerskull.sh <subject number> (e.g., 05)
# note: mne_watershed_shrink.sh must be in same directory
# note: check whether symbolic link was properly established


export FSVER='6.0.0'

export FSDIR=${FSROOT}/${FSVER}

export FREESURFER_HOME=/imaging/local/software/freesurfer/${FSVER}/`arch`
export SUBJECTS_DIR=/imaging/hauk/rl05/fake_diamond/data/mri
# export FUNCTIONALS_DIR=/imaging/`whoami`/sessions
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${FREESURFER_HOME}/lib_flat

echo $FREESURFER_HOME

source $FREESURFER_HOME/FreeSurferEnv.sh

export MNE_ROOT=/imaging/local/software/mne/mne_2.7.3/x86_64/MNE-2.7.3-3268-Linux-x86_64
export MNE_BIN_PATH=$MNE_ROOT/bin

export PATH=${PATH}:${MNE_BIN_PATH}
# source $MNE_ROOT/bin/mne_setup

export SUBJECT=$1
echo "Shrinking innerskull of sub-$SUBJECT"

rm -fR ${SUBJECTS_DIR}/${SUBJECT}/bem/watershed/*

# shrinking script
/imaging/hauk/rl05/fake_diamond/scripts/preprocessing/mri/mne_watershed_shrink.sh --subject "sub-${SUBJECT}" --overwrite

ln -sf ${SUBJECTS_DIR}/sub-${SUBJECT}/bem/watershed/sub-${SUBJECT}_inner_skull_surface ${SUBJECTS_DIR}/sub-${SUBJECT}/bem/inner_skull.surf
