#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:21:50 2023

@author: rl05
"""

import os
import os.path as op

from mne import (setup_source_space, write_source_spaces, read_source_spaces, 
                 compute_source_morph)

import config 

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subjects = config.subject_ids

src_to = read_source_spaces(op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src.fif'))

# for processing single subject
subject = input('subject to process: ')


subject = f'sub-{subject}'
print(subject)
print('======')

src_fname = op.join(subjects_dir, subject, f'{subject}_oct6_src.fif')
morph_fname = op.join(subjects_dir, subject, f'{subject}')

if op.isfile(src_fname):
    print(f'{subject} source spaces exist.')
    src = read_source_spaces(src_fname)
    
else:
    if op.exists(op.join(subjects_dir, subject)):
        print (f'Setting up source spaces for {subject}.')
        src = setup_source_space(subject, spacing=config.src_spacing) 
        write_source_spaces(src_fname, src, overwrite=True, verbose=False)
    else:
        print(f'{subject} recon perhaps not yet done. Skipping for now.')

if src:
    if not op.isfile(morph_fname):
        print (f'{subject} source morph does not exist. Computing.')
        morph = compute_source_morph(src, 
                                     subject_from=subject, 
                                     subject_to='fsaverage', 
                                     src_to=src_to
                                     )
        
        morph.save(morph_fname, overwrite=True)