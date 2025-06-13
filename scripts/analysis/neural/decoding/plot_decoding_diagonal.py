#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates and saves plots and statistical results of neural decoding.

This script is interactive and performs the following steps:
1.  Loads specified decoding scores for a group of subjects.
2.  Plots the time-series of decoding accuracy.
3.  Performs permutation cluster tests and saves the p-values to disk.
4.  For cross-condition analyses, it builds a dataframe, runs an ANOVA,
    saves the results, and creates a corresponding bar plot.
5.  Saves all figures to the project's 'figures' directory.

@author: rl05
"""

# --- 1. IMPORTS ---
import sys
import os
import os.path as op
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from mne.stats import permutation_cluster_1samp_test

# Add custom script paths
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config
from plot_decoding import *

mpl.rc_file('fake_diamond.rc')

# ─────────────────────────────────────────────────────────────────────────────
# Parse CLI arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot time-generalisation decoding results")
parser.add_argument('--to_plot', type=str, default='denotation+concreteness')
parser.add_argument('--data_type', type=str, default='ROI')
args = parser.parse_args()
to_plot = args.to_plot
data_type = args.data_type


# ─────────────────────────────────────────────────────────────────────────────
# Helper function
# ─────────────────────────────────────────────────────────────────────────────
def add_time_window_annotation(axis, x, y, width, height, label, facecolor, alpha, **kwargs):
    """Adds a labeled rectangle patch to an axes object."""
    rect = patches.Rectangle((x, y), width, height, linewidth=0., facecolor=facecolor, alpha=alpha)
    axis.add_patch(rect)
    text_x = x + width / 2
    text_y = y + height / 2
    axis.text(text_x, text_y, label, ha='center', va='center', **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Map user-friendly input names to the backend analysis names
ANALYSIS_MAPPING = {
    'denotation+concreteness': ['denotation', 'concreteness'],
    'composition': ['composition'],
    'specificity': ['specificity'],
    'specificity_word': ['specificity_word'],
    'concreteness_xcond': ['concreteness_trainWord_testSub', 'concreteness_trainWord_testPri'],
    'concreteness_xcond_general': ['concreteness_general_testSub', 'concreteness_general_testPri'],
    'concreteness_xcond_full': [
        'concreteness_trainSub_testSub', 'concreteness_trainSub_testPri',
        'concreteness_trainPri_testSub', 'concreteness_trainPri_testPri'
    ]
}

# Define time windows for statistical analysis [in samples]
# This cleans up the large if/elif block for time windows
ANOVA_TIME_WINDOWS = {
    'single': {
        100: {'early': slice(30, 50), 'late': slice(50, 80)},
        250: {'early': slice(75, 125), 'late': slice(125, 200)}
    },
    'sliding': {
        47: {'early': slice(14, 24), 'late': slice(24, 38)},
        50: {'early': slice(15, 25), 'late': slice(25, 40)},
        95: {'early': slice(29, 48), 'late': slice(48, 76)},
        97: {'early': slice(29, 49), 'late': slice(49, 78)},
        118: {'early': slice(36, 60), 'late': slice(60, 95)}
    }
}

layout_config = {
    'denotation+concreteness': (2, 1, (10, 6), True, True),
    'concreteness_xcond': (2, 1, (6, 6), False, True),
    'concreteness_xcond_general': (2, 1, (6, 6), False, True),
    'concreteness_xcond_full': (2, 2, (12, 6), True, True),
    # fallback/default
    'default': (1, 1, (10, 3), False, False),
}

# --- 3. USER INPUT & PATHS ---

classifier = 'logistic'
window = 'single'
micro_ave = True
if micro_ave:
    micro_averaging = 'micro_ave'
roi = input(f'For "{to_plot}", enter ROI: ') if data_type == 'ROI' else None
subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print(f'Subjects (n={len(subjects)}): {subjects}')

# Figure output directory
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}/diagonal/{classifier}/{data_type}/{window}/{micro_averaging}')
os.makedirs(figures_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load decoding scores
# ─────────────────────────────────────────────────────────────────────────────
analyses = ANALYSIS_MAPPING.get(to_plot, [])
scores_group = [
    read_decoding_scores(subjects, analysis, classifier, data_type, window=window, roi=roi, micro_ave=micro_ave)
    for analysis in analyses
]

# Infer sampling frequency from data shape
sfreq = int(scores_group[0].shape[1] / 1.6) if to_plot in ['denotation+concreteness', 'specificity'] else int(scores_group[0].shape[1] / 0.8)
print('Inferred sfreq:', sfreq)

# --- 5. PLOTTING: TIME-SERIES DECODING ---
# Define figure layout based on plot type
nrows, ncols, figsize, sharey, sharex = layout_config.get(to_plot, layout_config['default'])
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey, sharex=sharex)
axes = np.array(axes).reshape(-1)  # flatten in case of multiple plots
# gs = GridSpec(2, 2 if to_plot.endswith('_full') else 1) # Adjust GridSpec for multi-plots
gs = GridSpec(nrows, ncols)

# --- Main plotting loop ---
for i, analysis in enumerate(analyses):
    axis = plt.subplot(gs[i])
    plot_scores(axis, scores_group[i], analysis=analysis, chance=0.5)

    # Perform permutation cluster test
    good_clusters, cluster_pvals = permutation_tests(scores_group[i], timegen=False, against_chance=True)
    plot_clusters(good_clusters, scores=scores_group[i], analysis=analysis, ax=axis, cluster_pvals=cluster_pvals)

    # --- SAVE P-VALUES TO DISK ---
    # if cluster_pvals: # Check if the list of p-values is not empty
    #     pval_filename = op.join(figures_dir, f'stats_p-values_{analysis}_{roi}.txt')
    #     np.savetxt(pval_filename, cluster_pvals, fmt='%.4f', header=f'Cluster p-values for {analysis} in {roi}')
    #     print(f"Saved cluster p-values to {pval_filename}")


    if cluster_pvals: # Check if the list of p-values is not empty
        cluster_stats = []
        for cluster, p_val in zip(good_clusters, cluster_pvals):
            if analysis.split('_')[-1] in ['subsective','privative','testSub','testPri']:
                times = np.linspace(0., 0.8, scores_group[i].shape[1])
            else:
                times = np.linspace(-0.2, 1.4, scores_group[i].shape[1])            
            cluster_start = cluster[0].start
            cluster_stop = cluster[0].stop
            cluster_stats.append([p_val, round(times[cluster_start],3), round(times[cluster_stop],3)])
            print(f"  Found significant cluster for '{analysis}': p={p_val:.4f}, extent={cluster_start:.3f}s - {cluster_stop:.3f}s")

        # Save all cluster stats for this analysis to a single file
        stats_filename = op.join(figures_dir, f'stats_clusters_{analysis}_{roi}')
        if micro_ave:
            stats_filename += '_micro-ave.txt'
        else:
            stats_filename += '.txt'
        header = f'Cluster statistics for {analysis} in {roi}\nColumns: p-value, start_time (s), end_time (s)'
        np.savetxt(stats_filename, cluster_stats, fmt='%.4f', header=header)
        print(f"Saved cluster stats to {stats_filename}.")


    # --- Add annotations and style axes within the same loop ---
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)

    if to_plot == 'denotation+concreteness':
        axis.set_xlim(-0.2, 1.4)
        if data_type == 'ROI':
            axis.set_yticks([0.5, 0.52, 0.54])
        else:
            axis.set_yticks([0.5, 0.52, 0.54, 0.56])
        if i == 0: # Top plot
            axis.set_xlabel('')
            axis.set_xticks([])
            axis.spines['bottom'].set_visible(False)
            # Add legend
            legend_elements = [Line2D([0], [0], color=plt.get_cmap(color_scheme[an])(0.7), lw=3) for an in analyses]
            axis.legend(legend_elements, ['denotation', 'concreteness'], frameon=False)
        if i > 0: # Bottom plot, add time indicators
            if data_type == 'ROI':
                y = 0.54
            else:
                y = 0.56
            add_time_window_annotation(axis, x=0.0, y=y, width=0.3, height=0.02, label='adjective', color='black', fontsize=15, facecolor='lightgrey', alpha=0.5)
            add_time_window_annotation(axis, x=0.6, y=y, width=0.3, height=0.02, label='noun', color='black', fontsize=15, facecolor='lightgrey', alpha=0.5)

    elif to_plot.startswith('concreteness_xcond'):
        axis.set_xlim(0, 0.8)
        # Add transparent vertical spans to highlight time windows
        axis.axvspan(0.3, 0.5, alpha=0.50, color='lightgrey', zorder=0)
        axis.axvspan(0.5, 0.8, alpha=0.75, color='lightgrey', zorder=0)
        if i == 0:
            axis.set_xlabel('')
            axis.set_xticks([])
            axis.spines['bottom'].set_visible(False)
            axis.set_yticks([0.5, 0.52, 0.54])
            # Add legend
            legend_elements = [Line2D([0], [0], color=plt.get_cmap(color_scheme[an])(0.7), lw=3) for an in analyses]
            axis.legend(legend_elements, ['test on subsective', 'test on privative'], frameon=False, loc='upper left')
        elif i == 1:
            axis.set_yticks([0.48, 0.5, 0.52])
            # Add time indicators only to the second plot
            add_time_window_annotation(axis, x=0.30, y=0.47, width=0.2, height=0.0075, label='early', color='black', fontsize=9, facecolor='lightgrey', alpha=0.75)
            add_time_window_annotation(axis, x=0.50, y=0.47, width=0.3, height=0.0075, label='late', color='black', fontsize=9, facecolor='lightgrey', alpha=1)

plt.tight_layout()
decoding_fig_fname = op.join(figures_dir, f'decode_diagonal_{roi}_{sfreq}Hz')
if micro_ave:
    decoding_fig_fname += '_micro-ave.png'
else:
    decoding_fig_fname += '.png'
plt.savefig(decoding_fig_fname)
plt.close()
print(f"Saved decoding scores to {decoding_fig_fname}.")


# --- 6. STATISTICAL ANALYSIS & BAR PLOT (for cross-condition designs) ---

if to_plot.startswith('concreteness_xcond'):
    print("\n--- Running cross-condition interaction analysis ---")
    # Prepare DataFrame for ANOVA
    df_list = []
    time_windows = ANOVA_TIME_WINDOWS.get(window, {}).get(sfreq)

    if time_windows:
        # Determine iteration parameters based on plot type
        if to_plot == 'concreteness_xcond_full':
            iter_params = list(zip(analyses, ['subsective','privative','subsective','privative'], ['subsective','subsective','privative','privative']))
        else:
            iter_params = list(zip(analyses, ['subsective','privative']))

        # Loop to populate the DataFrame
        for i, params in enumerate(iter_params):
            analysis_name, evaluation = params[0], params[1]
            for timewindow_name, tw_slice in time_windows.items():
                scores_timewindow = scores_group[i][:, tw_slice].mean(axis=1)
                for subj_idx, score in enumerate(scores_timewindow):
                    row = {'score': score, 'timewindow': timewindow_name, 'evaluation': evaluation, 'subject': subjects[subj_idx]}
                    if to_plot == 'concreteness_xcond_full':
                        row['train_on'] = params[2]
                    df_list.append(row)
        averaged_data = pd.DataFrame(df_list)

        # --- RUN AND SAVE ANOVA ---
        print("Fitting OLS model and running ANOVA...")
        model = ols('score ~ C(evaluation) * C(timewindow)', data=averaged_data).fit()
        anova_results = anova_lm(model, typ=2)
        print(anova_results)
        anova_filename = op.join(figures_dir, f'stats_anova_{roi}_{sfreq}Hz.csv')
        anova_results.to_csv(anova_filename)
        print(f"Saved ANOVA results to {anova_filename}")

        # Note: Other statistical models like MixedLM or follow-up tests can be run here.
        # For example:
        # model = smf.mixedlm('score ~ C(evaluation) * C(timewindow)', averaged_data, groups=averaged_data['subject']).fit()
        # print(model.summary())

        # --- Create and save bar plot ---
        color_palette = [plt.get_cmap(color_scheme['concreteness_trainWord_testSub'])(0.7),
                         plt.get_cmap(color_scheme['concreteness_trainWord_testPri'])(0.7)]

        fig, axis = plt.subplots(figsize=(3, 3.5))
        sns.barplot(
            data=averaged_data, x='timewindow', y='score', hue='evaluation',
            palette=color_palette, errorbar='se', ax=axis
        )
        axis.set(ylim=(0.45, 0.55), xlabel='Time Window', ylabel='Decoding Accuracy (AUC)')
        axis.legend(title='', frameon=False)
        plt.tight_layout()
        plt.savefig(op.join(figures_dir, f'fig_barplot_{roi}_{sfreq}Hz.png'))
        plt.close()
    else:
        print(f"Warning: No defined time windows for sfreq={sfreq} and window='{window}'. Skipping ANOVA.")