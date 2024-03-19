#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:49:58 2024

@author: rl05
"""

import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_context("talk")

analysis_path = '/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/regression'
output_path = op.join(analysis_path, 'output')
rois = ['anteriortemporal-lh','posteriortemporal-lh','inferiorfrontal-lh',
        'temporoparietal-lh','lateraloccipital-lh']

analyses = ['composition', 'denotation', 'concreteness', 'specificity','length']


for analysis in analyses:
    fig, axs = plt.subplots(len(rois), 1, sharey=True, sharex=True, figsize=(8, len(rois) * 2))

    for i, roi in enumerate(rois):
        
        chisq = pd.read_csv(op.join(output_path, f'chisq_({analysis})_{roi}_(n=36)_word2only.csv'))
        chisq_roi = chisq[chisq['roi'] == roi]
        
        times = np.unique(chisq_roi['timepoint']) / 250
        axs[i].plot(times, chisq_roi['chisq'])
        axs[i].set_title(f'{roi}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Chi-square')
    
    plt.tight_layout()
    plt.savefig(op.join(analysis_path, f'figures/chisq_{analysis}_(n=36)_word2only.png'))
    plt.close(fig)