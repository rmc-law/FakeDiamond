#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:47:55 2024

@author: rl05
"""


import sys
import os
import os.path as op
import numpy as np

from mne import read_epochs, EvokedArray, compute_source_morph, read_source_spaces
from mne.minimum_norm import read_inverse_operator, apply_inverse

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 


subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

analyses = [
    # 'denotation','concreteness','composition',
            'denotation_cross_condition_test_on_subsective',
            'denotation_cross_condition_test_on_privative']


fsaverage_src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_to = read_source_spaces(fsaverage_src_fname) 

for analysis in analyses:
        
    for subject in subjects:
        
        subject = f'sub-{subject}'
        coef_dir = op.join(decoding_dir, f'output/{analysis}/diagonal/logistic/MEEG/{subject}')
        coef_fname = op.join(coef_dir, 'scores_coef_MEEG.npy')
        
        coef_projection_fname = op.join(coef_dir, f'source_projection_coef_{analysis}-lh.stc')
        if op.exists(coef_projection_fname):
            print(subject, 'done. skipping.')
            pass
        else:
            if (subject == 'sub-12') or (subject == 'sub-13'):
                continue
            print(subject, 'processing.')
            epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
            epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
            epochs = read_epochs(epoch_fname, preload=False, verbose=False)
            if epochs.info['bads'] != []:
                epochs.info['bads'] = []
            inv_fname = op.join(epoch_path, f'{subject}_MEEG_inv.fif')
            inv = read_inverse_operator(inv_fname, verbose=False)
            
            snr = 3.0
            lambda2 = 1.0 / snr ** 2
            
            if op.exists(coef_fname):
                print('load backprojected coef.')
                coef = np.load(coef_fname)
            
            evoked_time_decod = EvokedArray(coef, epochs.info, tmin=epochs.times[0])
            
            stc = apply_inverse(evoked_time_decod, inv, lambda2=lambda2, method='MNE',
                                return_residual=False)
            morph = compute_source_morph(
                src=stc,
                subject_from=subject,
                subject_to='fsaverage_src', 
                src_to=src_to
                )        
            stc_morph = morph.apply(stc)
            
            assert len(stc_morph.vertices[0])+len(stc_morph.vertices[1]) == src_to[0]['nuse']+src_to[1]['nuse']
            
            del stc
            stc_morph.save(op.join(coef_dir, f'source_projection_coef_{analysis}'), 
                           overwrite=True)
            del stc_morph
