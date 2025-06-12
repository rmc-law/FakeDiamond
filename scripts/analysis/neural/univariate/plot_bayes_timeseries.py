#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:46:46 2024

@author: rl05

Plot Bayes factor time series for supplementary
"""

import sys
import os
import os.path as op
import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 
from config_plotting import *

mpl.rc_file('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/univariate/fake_diamond.rc')

FIG_DIR = '/imaging/hauk/rl05/fake_diamond/figures/bayes/'
os.makedirs(FIG_DIR, exist_ok=True)
analyses = ['composition','denotation']

for analysis in analyses:
    for hemi in ['lh','rh']:
        bf_dir = f'/imaging/hauk/rl05/fake_diamond/results/neural/bayes/{analysis}'
        bf_ts_fname = os.path.join(bf_dir, f'bayes_factor_time_series_{hemi}.csv')
        if not os.path.isfile(bf_ts_fname):
            continue
        bf_ts_fig_fname = os.path.join(FIG_DIR, f'bayes_factor_timeseries_{analysis}_{hemi}.png')

        df = pd.read_csv(bf_ts_fname)
        times = np.linspace(0.0, 0.8, 200)
        plt.figure(figsize=(5.5, 3.))

        # Base plot: thinner lines for all points
        plt.plot(times, df['BF10_A'], color='purple', linewidth=1.25, label='BF10: Concreteness effect')
        plt.plot(times, df['BF01_A'], color='purple', linestyle='--', linewidth=1.0, label='BF01: Concreteness effect')
        plt.plot(times, df['BF10_B'], color='black', linewidth=1.25, label='BF10: Composition effect')
        plt.plot(times, df['BF01_B'], color='black', linestyle='--', linewidth=1.0, label='BF01: Composition effect')

        # Plot interpretability thresholds
        plt.axhline(1, color='gray', linestyle=':', linewidth=1.0)
        plt.axhline(3, color='red', linestyle='--', linewidth=0.8)
        plt.axhline(1/3, color='red', linestyle='--', linewidth=0.8)

        plt.xlabel('Time (s)')
        plt.ylabel('Bayes Factor\n(log scale)')
        plt.title(f'anteriortemporal-{hemi}')
        plt.yscale('log')
        # plt.grid(True, linestyle=':', alpha=0.5)
        if analysis == 'composition' and hemi == 'lh': # just put legend on the first figure
            plt.legend(loc='best', fontsize='x-small')
        plt.tight_layout()

        if bf_ts_fig_fname:
            plt.savefig(bf_ts_fig_fname, dpi=300)
            print(f'Saved plot to {bf_ts_fig_fname}')
            plt.close()
        else:
            plt.show()
