#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:56:30 2023

@author: rl05

Convert subject dicom images to nifti files for MRI reconstructions
"""

import os
import os.path as op
from glob import glob

import dicom2nifti

import config


bids_root = config.bids_root
print('BIDS root:', bids_root)

subjects = config.subject_ids
print('Subjects:', subjects)

def check_nifti_exists(directory):
    nifti_files = glob(op.join(directory, '*nii*'))
    return len(nifti_files) > 0

def find_dicom_images(directory):
    try: 
        dicoms_path = glob(op.join(directory, f'{mri_id}*/*', '*anat-T1w'))
        if len(dicoms_path) > 0: 
            return dicoms_path[0]
        dicoms_path = glob(op.join(directory, f'{mri_id}*/*', '*32chn'))
        if len(dicoms_path) > 0: 
            return dicoms_path[0]
        return None
    except:
        print(f'Can\'t seem to find sub-{subject}\'s T1w images. Try manually look?')
        return None

print()
print('Converting subject dicom images to nifti files...')
for subject in subjects:
    bids_path_anat = op.join('/imaging/hauk/rl05', config.project, 'data', f'sub-{subject}', 'anat')
    if not op.isdir(bids_path_anat):    
        os.mkdir(bids_path_anat)
    mri_id = config.map_subjects_mri[subject]
    if check_nifti_exists(bids_path_anat):
        print(f'sub-{subject}: nifti files already exist in subject BIDS dataset.')
    else:
        print(f'sub-{subject}: nifti files do not exist. Converting dicoms now...')
        dicoms_path = find_dicom_images('/mridata/cbu')
        dicom2nifti.convert_directory(dicoms_path, bids_path_anat) # converts all dicom to nifti
        print('Done.')

print()

print('Dicoms converted to niftis for all subjects.')
