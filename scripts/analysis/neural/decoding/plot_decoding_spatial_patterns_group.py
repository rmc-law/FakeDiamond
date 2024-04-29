#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:57:52 2024

@author: rl05
"""


import sys
import os
import os.path as op
import numpy as np

from mne import read_epochs, EvokedArray
import matplotlib.pyplot as plt

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 
import helper


subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')

analysis = input('analysis: ')
classifier = input('classifier: ')
sensors= input('sensors: ')
data_type = input('MEEG or ROI: ')

figures_dir = op.join(decoding_dir, f'figures/{analysis}')
if not op.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

# to get epochs info
epochs = read_epochs(op.join(config.project_repo, 'data/preprocessed/sub-01/epoch/sub-01_epo.fif'), preload=True, verbose=False)

coef_all = []
for subject in subjects:
    
    subject = f'sub-{subject}'
    coef_dir = op.join(decoding_dir, f'output/{analysis}/diagonal/{classifier}/{sensors}/{subject}')
    coef_file = op.join(coef_dir, f'scores_coef_{sensors}.npy')
    if op.exists(coef_file):
        coef = np.load(coef_file)
        if coef.shape[0] != 370:
            pass
        else:
            coef_all.append(coef)
    
coef_all = np.stack(coef_all, axis=0)
print(coef_all.shape)
avg, _ = helper.calculate_avg_sem(coef_all)
print(avg.shape)

# plot group-level spatial patterns
evoked_time_decod = EvokedArray(avg, epochs.info, tmin=epochs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
times = np.arange(-0.2, 1.4, 0.2)
fig_evokeds = evoked_time_decod.plot_joint(
    times=times, title='patterns', show=False, **joint_kwargs
)
if sensors == 'MEEG':
    ch_types = ['eeg', 'mag', 'grad']
elif sensors == 'MEG':
    ch_types = ['mag', 'grad']
for fig_evoked, ch_type in zip(fig_evokeds, ch_types):
    fig_evoked.savefig(op.join(figures_dir, f'fig_patterns_group_{classifier}_{sensors}_{ch_type}.png'))
    plt.close(fig_evoked)