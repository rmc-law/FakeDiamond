#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads pre-calculated source time courses (STCs) for different
experimental conditions, extracts data from specified regions of interest (ROIs),
and plots the average time course. It calculates and displays within-subject
error bars.

For time windows with significant effects (from pre-computed stats), it also
generates corresponding bar plots of the mean activity.
"""

import sys
import os
import os.path as op
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import mne
from mne import read_source_estimate, read_labels_from_annot, read_source_spaces

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config
from config_plotting import color_scheme

# stylesheet
try:
    mpl.rc_file('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/univariate/fake_diamond.rc')
except FileNotFoundError:
    print("Stylesheet not found. Using default matplotlib styles.")

DATA_DIR = op.join(config.project_repo, 'data')
FIGS_DIR = op.join(config.project_repo, 'figures', 'univariate')
RESULTS_DIR = '/imaging/hauk/rl05/fake_diamond/results/neural/roi/anova/'
STC_PATH = op.join(DATA_DIR, 'stcs')
SUBJECTS_DIR = op.join(DATA_DIR, 'mri')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR

PARC = 'semantics'
FSAFE_SRC_FNAME = op.join(SUBJECTS_DIR, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')

def get_analysis_parameters(analysis: str) -> dict:
    """Returns a dictionary of parameters for a given analysis type."""
    params = {
        'composition': {
            'conditions': ['concrete-baseline', 'concrete-subsective', 'abstract-baseline', 'abstract-subsective'],
            'relevant_conditions': ['baseline', 'subsective'],
            'legend_labels': ['word', 'phrase'],
        },
        'denotation': {
            'conditions': ['concrete-subsective', 'concrete-privative', 'abstract-subsective', 'abstract-privative'],
            'relevant_conditions': ['subsective', 'privative'],
            'legend_labels': ['subsective', 'privative'],
        },
        'specificity': {
            'conditions': ['low', 'mid', 'high'],
            'relevant_conditions': ['low', 'mid', 'high'],
            'legend_labels': ['low', 'mid', 'high'],
        },
        'specificity_word': {
            'conditions': ['low', 'high'],
            'relevant_conditions': ['low', 'high'],
            'legend_labels': ['low', 'high'],
        }
    }
    if analysis not in params:
        raise ValueError(f"Analysis '{analysis}' not recognized. Available: {list(params.keys())}")
    
    p = params[analysis]
    p['colors_bar'] = [color_scheme.get(c, 'gray') for c in p['conditions']]
    
    # Define colors for time courses based on relevant conditions
    if analysis == 'composition':
        p['colors_tc'] = [color_scheme.get('baseline'), color_scheme.get('subsective')]
    elif analysis == 'denotation':
        p['colors_tc'] = [color_scheme.get('subsective'), color_scheme.get('privative')]
    elif analysis == 'specificity':
        p['colors_tc'] = plt.cm.Blues([0.4, 0.6, 0.8]).tolist()
    elif analysis == 'specificity_word':
        p['colors_tc'] = plt.cm.Blues([0.4, 0.8]).tolist()
        
    return p

def calculate_pooled_ws_sem(data: np.ndarray) -> np.ndarray:
    """
    Calculates the pooled within-subject SEM via Loftus & Masson (1994).

    This method removes between-subject variability and then computes a single,
    pooled error term from the Mean Square Error of the subject-by-condition
    interaction, which is appropriate for repeated-measures visualizations.

    Args:
        data: A numpy array of shape (n_subjects, n_conditions, ...).
              The trailing dimension is typically time or another variable.

    Returns:
        A single pooled SEM array of shape (...). This same error value
        should be applied to the means of all conditions.
    """
    n_subjects, n_conditions = data.shape[:2]
    if n_subjects <= 1 or n_conditions <= 1:
        return np.zeros(data.shape[2:]) # Cannot compute interaction with 1 subject/condition

    # Step 1: Remove between-subject variability by normalizing each subject's data.
    subject_means = data.mean(axis=1, keepdims=True)
    grand_mean = data.mean(axis=(0, 1), keepdims=True)
    normalized_data = data - subject_means + grand_mean

    # Step 2: Calculate pooled error from the subject-by-condition interaction.
    # This is derived from the Mean Square Error of the interaction term in a
    # repeated measures ANOVA.
    condition_means = normalized_data.mean(axis=0, keepdims=True)
    residuals = normalized_data - condition_means
    ss_interaction = np.sum(residuals**2, axis=(0, 1))
    df_interaction = (n_subjects - 1) * (n_conditions - 1)
    
    # Avoid division by zero if df is 0
    if df_interaction == 0:
        return np.zeros(data.shape[2:])

    mse_interaction = ss_interaction / df_interaction

    # The pooled SEM is sqrt(MSE_interaction / n_subjects)
    pooled_sem = np.sqrt(mse_interaction / n_subjects)

    return pooled_sem

def load_stc_data(subjects: list, conditions: list) -> dict:
    """Loads STC data into a dictionary, organized by condition."""
    all_data = {condition: [] for condition in conditions}
    for subject in subjects:
        print(f'Reading in STCs for {subject}.')
        for condition in conditions:
            stc_fname = op.join(STC_PATH, subject, f'{subject}_{condition}_MEEG-lh.stc')
            if not op.exists(stc_fname):
                print(f"Warning: File not found {stc_fname}")
                continue
            stc = read_source_estimate(stc_fname, subject='fsaverage_src')
            stc = stc.crop(tmin=0.6, tmax=1.4)
            all_data[condition].append(stc)
    return all_data

def plot_significant_bars(all_data: dict, subjects: list, roi_label: mne.Label, src: mne.SourceSpaces,
                          cluster_info: dict, params: dict, roi_name: str, analysis: str):
    """Generates and saves a bar plot for a significant time cluster."""
    stc_template = all_data[params['conditions'][0]][0]
    times = stc_template.times
    t_start, t_stop = cluster_info['tstart'], cluster_info['tstop']
    
    mean_activity = {}
    for cond in params['conditions']:
        stcs = all_data[cond]
        tcs = mne.extract_label_time_course(stcs, roi_label, src, mode='mean', return_generator=False)
        tcs_in_window = np.mean(tcs[:, (times >= t_start) & (times <= t_stop)], axis=1)
        mean_activity[cond] = tcs_in_window

    activity_array = np.stack([mean_activity[cond] for cond in params['conditions']], axis=1)
    group_means = activity_array.mean(axis=0)
    error_bars = _get_within_subject_sem(activity_array[:, :, np.newaxis]).flatten()

    fig_bar, ax = plt.subplots(figsize=(4, 3.5))
    if analysis in ['composition', 'denotation']:
        x_pos = [0.8, 1.6, 3.2, 4.0]
        ax.set_xticks([1.2, 3.6]); ax.set_xticklabels(['Concrete', 'Abstract'])
    else:
        x_pos = np.arange(len(params['conditions'])) * 1.5
        ax.set_xticks(x_pos); ax.set_xticklabels(params['legend_labels'])
        
    ax.bar(x_pos, group_means, yerr=error_bars, color=params['colors_bar'], width=0.7, capsize=4)
    ax.set_ylabel('Activity (MNE)'); ax.spines[['top', 'right']].set_visible(False)
    fig_bar.tight_layout()

    fname = f"{analysis}_{roi_name}_cluster{cluster_info['id']}_bars.png"
    fig_bar.savefig(op.join(FIGS_DIR, fname)); plt.close(fig_bar)


def plot_significant_bars(all_data: dict, roi_label: mne.Label, src: mne.SourceSpaces,
                          cluster_info: dict, params: dict, roi_name: str, analysis: str):
    """Generates and saves a bar plot for a significant time cluster."""
    stc_template = all_data[params['conditions'][0]][0]
    times = stc_template.times
    t_start, t_stop = cluster_info['tstart'], cluster_info['tstop']

    # Extract mean activity in the time window for each subject and condition
    mean_activity_list = []
    for cond in params['conditions']:
        stcs = all_data[cond]
        tcs = mne.extract_label_time_course(stcs, roi_label, src, mode='mean', return_generator=False)
        tcs_in_window = np.mean(tcs[:, (times >= t_start) & (times <= t_stop)], axis=1)
        mean_activity_list.append(tcs_in_window)

    # Calculate group means and the single, pooled error bar
    activity_array = np.stack(mean_activity_list, axis=1)
    group_means = activity_array.mean(axis=0)
    # Add a trailing dimension for the time-agnostic calculation
    pooled_error_bar = calculate_pooled_ws_sem(activity_array[..., np.newaxis]).item()

    # --- Plotting ---
    fig_bar, ax = plt.subplots(figsize=(4, 3.5))
    if analysis in ['composition', 'denotation']:
        x_pos = [0.8, 1.6, 3.2, 4.0]
        ax.bar(x_pos, group_means, yerr=pooled_error_bar, color=params['colors_bar'], width=0.7, capsize=4)
        ax.set_xticks([1.2, 3.6])
        ax.set_xticklabels(['Concrete', 'Abstract'])
    else: # Specificity analyses
        x_pos = np.arange(len(params['conditions'])) * 1.5
        ax.bar(x_pos, group_means, yerr=pooled_error_bar, color=params['colors_bar'], width=0.7, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(params['legend_labels'])

    ax.set_ylabel('Activity (MNE)')
    ax.spines[['top', 'right']].set_visible(False)
    fig_bar.tight_layout()

    # Save figure
    fname = f"{analysis}_{roi_name}_cluster{cluster_info['id']}_bars.png"
    fig_bar.savefig(op.join(FIGS_DIR, fname))
    plt.close(fig_bar)


def plot_roi_timecourses(all_data: dict, analysis: str, params: dict, rois: dict,
                         labels: list, src: mne.SourceSpaces):
    """Plots time courses for specified ROIs using MNE-native functions."""
    n_rois = len(rois)
    rows = (n_rois + 1) // 2
    fig, axes = plt.subplots(rows, 2, sharex=True, sharey=True, figsize=(10, 2.5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (roi_name_str, roi_name_plot) in enumerate(rois.items()):
        ax = axes[i]
        try:
            roi_label = next(lbl for lbl in labels if lbl.name == roi_name_str)
        except StopIteration:
            print(f"Warning: Label '{roi_name_str}' not found. Skipping.")
            ax.set_title(f"{roi_name_plot}\n(Label not found)")
            continue

        # Aggregate data to plot main effects (e.g., 'word' vs 'phrase')
        data_for_sem = []
        n_subjects = len(config.subject_ids)
        n_times = all_data['concrete-subsective'][0].times.shape[0]
        for cond_type in params['relevant_conditions']:
            conds_to_average = [c for c in params['conditions'] if cond_type in c.split('-')]
            # Average across other factors (e.g., concreteness) for each subject
            tcs_to_sum = []
            for cond in conds_to_average:
                stcs = all_data.get(cond)
                if not stcs:
                    continue
                extracted_tc = mne.extract_label_time_course(stcs, roi_label, src, mode='mean', return_generator=False)
                tcs_to_sum.append(extracted_tc)
            
            if not tcs_to_sum:
                # Handle case where no data was found for this cond_type
                # We can't determine n_subjects/n_times, so we skip this cond_type
                print(f"Warning: No data found for condition type '{cond_type}' in ROI '{roi_name_str}'.")
                continue

            # Sum the collected arrays and then average
            summed_tc = np.sum(tcs_to_sum, axis=0)
            avg_tc = summed_tc / len(tcs_to_sum)
            data_for_sem.append(avg_tc)

        # Calculate group means and the single, pooled SEM time course
        group_means = [d.mean(axis=0) for d in data_for_sem]
        sem_data_stack = np.stack(data_for_sem, axis=1) # Shape: (n_sub, n_cond_type, n_times)
        pooled_sem_tc = calculate_pooled_ws_sem(sem_data_stack)

        times = all_data[params['conditions'][0]][0].times
        for j, _ in enumerate(params['relevant_conditions']):
            ax.plot(times, group_means[j][0], color=params['colors_tc'][j], lw=3)
            # Apply the same pooled SEM to all condition means
            ax.fill_between(times, group_means[j][0] - pooled_sem_tc, group_means[j][0] + pooled_sem_tc,
                            alpha=0.2, color=params['colors_tc'][j])

        # --- Decoration ---
        ax.set_title(roi_name_plot)
        ax.set_xlim(times.min(), times.max())
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=-1)

        # --- Add Significance Clusters ---
        pickle_fname = op.join(RESULTS_DIR, roi_name_str, analysis, f'{roi_name_str}.pickle')
        if op.exists(pickle_fname):
            with open(pickle_fname, 'rb') as f:
                res = pickle.load(f)
            for cluster_idx, p_val in enumerate(res.clusters['p']):
                if p_val < 0.05:
                    cluster = res.clusters[cluster_idx]
                    t_start, t_stop = cluster['tstart'] - 0.6, cluster['tstop'] - 0.6
                    ymin, ymax = ax.get_ylim()
                    ax.fill_betweenx([ymin, ymax], t_start, t_stop, color='yellow', alpha=0.3, zorder=-2)
                    
                    cluster_info = {'id': cluster_idx + 1, 'tstart': t_start, 'tstop': t_stop}
                    plot_significant_bars(all_data, roi_label, src, cluster_info, params, roi_name_plot, analysis)

    # --- Final Figure Adjustments ---
    fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Activity (MNE)', ha='center', va='center', rotation='vertical')
    legend_elements = [Line2D([0], [0], c=c, lw=3) for c in params['colors_tc']]
    axes[1].legend(legend_elements, params['legend_labels'], loc='upper right', frameon=False)
    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])

    roi_label_fname = 'ATLs' if len(rois) == 2 else 'allROIs'
    fig_name = op.join(FIGS_DIR, f'{analysis}_timecourse_{roi_label_fname}.png')
    fig.savefig(fig_name)
    plt.close(fig)
    print(f"Figure saved to: {fig_name}")


def main():
    """Main function to run the ROI analysis and plotting pipeline."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    subjects = [f'sub-{sid}' for sid in config.subject_ids if sid not in ['16']]
    print(f'Subjects (n={len(subjects)}): {subjects}')
    
    analysis = input('Analysis type? (composition, denotation, specificity, specificity_word): ')
    roi_to_plot = input('ROI set? (ATL, all): ')

    try:
        params = get_analysis_parameters(analysis)
    except (ValueError, KeyError) as e:
        print(f"Error setting parameters: {e}")
        return

    if roi_to_plot == 'ATL':
        rois = {'anteriortemporal-lh': 'LATL', 'anteriortemporal-rh': 'RATL'}
    elif roi_to_plot == 'all':
        rois = {
            'anteriortemporal-lh': 'LATL', 'anteriortemporal-rh': 'RATL',
            'posteriortemporal-lh': 'LPTL', 'posteriortemporal-rh': 'RPTL',
            'inferiorfrontal-lh': 'LIFG', 'inferiorfrontal-rh': 'RIFG',
            'temporoparietal-lh': 'LTPJ', 'temporoparietal-rh': 'RTPJ'
        }
    else:
        print("Invalid ROI set. Please choose 'ATL' or 'all'.")
        return

    all_data = load_stc_data(subjects, params['conditions'])
    
    if not any(all_data.values()):
        print("No data was loaded. Exiting.")
        return

    # Load annotation labels and source spaces needed for ROI extraction
    labels_lh = read_labels_from_annot('fsaverage_src', parc=PARC, hemi='lh', subjects_dir=SUBJECTS_DIR)
    labels_rh = read_labels_from_annot('fsaverage_src', parc=PARC, hemi='rh', subjects_dir=SUBJECTS_DIR)
    src = read_source_spaces(FSAFE_SRC_FNAME, verbose=False)

    plot_roi_timecourses(all_data, analysis, params, rois, labels_lh + labels_rh, src)


if __name__ == '__main__':
    main()