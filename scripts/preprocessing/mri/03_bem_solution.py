#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:25:48 2023

@author: rl05
"""

import os
import os.path as op
from mne import make_bem_model, make_bem_solution, write_bem_solution
from mne.viz import plot_bem

import config 

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subjects = config.subject_ids


for subject in subjects:
    
    subject = f'sub-{subject}'
    print(subject)
    print('======')

    # create two noise covariance matrices, one for just MEG, one for EEG & MEG
    ch_types = ['MEEG', 'MEG']
    conductivities = [[0.3, 0.006, 0.3], [0.3]]

    for ch_type, conductivity in zip(ch_types, conductivities):

        bem_fname = op.join(subjects_dir, subject, 'bem', f'{subject}_{ch_type}-bem-sol.fif')
        
        if op.exists(bem_fname):

            print(f'bem solution {ch_type} exists.')
            pass

        else:
            
            if op.exists(op.join(subjects_dir, subject)):

                print(f'bem solution {ch_type} does not exist. Computing.')
                os.makedirs(op.join(subjects_dir, subject, 'bem'), exist_ok=True)

                surfaces = make_bem_model(
                    subject, 
                    ico=4, 
                    conductivity=conductivity, 
                    verbose=True
                    )
                
                bem = make_bem_solution(surfaces)

                write_bem_solution(bem_fname, bem)
                
            else:
                print(f'{subject} recon perhaps not yet done. Skipping for now.')

            del surfaces, bem
        
        del bem_fname


    # plot bem to visualise
    plot_bem_kwargs = dict(
        subject=subject,
        subjects_dir=subjects_dir,
        brain_surfaces="white",
        orientation="coronal",
        slices=[50, 100, 150, 200],
    )

    fig_bem = plot_bem(**plot_bem_kwargs, show=False)
    fig_bem.savefig(op.join(subjects_dir, subject, 'bem', f'{subject}_bem.png'))