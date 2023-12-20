#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:37:18 2023

@author: rl05

Set up MNE-BIDS format and convert dicoms to nift for each subject
"""

# import sys
import os
import os.path as op
from glob import glob

from mne import set_log_level, find_events, pick_events
from mne.io import read_raw_fif
from mne_bids import (BIDSPath, write_raw_bids, print_dir_tree)

import dicom2nifti 

import config

set_log_level(verbose='WARNING')

bids_root = config.bids_root 

subject = input('sub to set up: ')
meg_id = config.map_subjects_meg[subject][0]
meg_date = config.map_subjects_meg[subject][1]
mri_id = config.map_subjects_mri[subject]
bids_path = BIDSPath(subject=subject, root=bids_root)
t1w_dir = f'/imaging/hauk/rl05/fake_diamond/data/mri/T1w/sub-{subject}'

if not op.exists(t1w_dir):
    print(f'sub-{subject} nifti does not exist.')

if mri_id == '':
    print(f'sub-{subject} not found in config.map_subjects_mri. Add manually?')

def find_meg_raw(directory, run=None):
    possible_naming = ['block*', 'run*']
    for naming in possible_naming:
        try:
            meg_path = glob(op.join(directory, f'{naming}{run}*'))
            if meg_path:
                print(f'Found run-{run} raw: ', meg_path[0])
                return meg_path[0]
        except Exception as e:
            print(f'Error searching for MEG raw: {e}')
    print(f'Can\'t seem to find MEG raw for sub-{subject}. Try looking manually.')
    return None

def create_bids_directory():
    cbu_data_path = op.join(config.cbu_repo_meg, meg_id, meg_date)
    for run in range(1,config.runs+1):
        bids_path = BIDSPath(
            subject=subject,
            session='01',
            task='semantic',
            run=str(run),
            root=bids_root
        )
        raw_fname = find_meg_raw(cbu_data_path, run=run)
        raw = read_raw_fif(raw_fname, preload=False)
        events = find_events(raw, stim_channel='STI101', min_duration=0.005)
        events = pick_events(events, include=list(config.event_id_semantic.values())) # exclude button presses and photodiodes
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            events=events,
            event_id=config.event_id_semantic,
        )
        preprocessed_data_dir = op.join(config.project_repo, 'data', 'preprocessed', f'sub-{subject}')
        info_fname = op.join(preprocessed_data_dir, 'info', f'run{run}-info.fif')
        if not op.exists(info_fname):
            os.makedirs(op.join(preprocessed_data_dir, 'info'), exist_ok=True)
        raw.info.save(info_fname)
    print(print_dir_tree(bids_path.directory))

def save_info(info, directory=None): 
    info.save(directory)
    return None

def check_nifti_exists(directory):
    nifti_files = glob(op.join(directory, '*nii*'))
    return len(nifti_files) > 0

def find_dicom_images(directory):
    possible_mri_endings = ['*anat-T1w', '*32chn','*T1w*']
    for ending in possible_mri_endings:
        try:
            dicoms_path = glob(op.join(directory, f'{mri_id}*/*', ending))
            if dicoms_path:
                print(f'Found dicom images for MRI ID {mri_id}: {dicoms_path[0]}')
                return dicoms_path[0]
        except Exception as e:
            print(f'Error searching for dicom images: {e}')
    raise FileNotFoundError(f'Can\'t seem to find dicom images for MRI ID {mri_id}. Try looking manually.')

def convert_dicom2nifti(t1w_dir=''):
    if not op.isdir(t1w_dir):    
        os.mkdir(t1w_dir)
    dicoms_path = find_dicom_images('/mridata/cbu')
    dicom2nifti.convert_directory(dicoms_path, t1w_dir) # converts all dicom to nifti

if __name__ == "__main__":
    # create bids format for meg data
    if op.isdir(op.join(bids_root, f'sub-{subject}')):
        print('BIDS directory already exists.')
    else:
        print('Creating BIDS directory.')
        create_bids_directory()
        print('BIDS directory created.')

    # convert dicom to nifti for surface reconstruction
    if check_nifti_exists(t1w_dir):
        print(f'Nifti files exist in {t1w_dir}.')
    else:
        print('Nifti files do not exist. Converting dicoms now.')
        convert_dicom2nifti(t1w_dir)
        print(f'Nifti saved in {t1w_dir}')
