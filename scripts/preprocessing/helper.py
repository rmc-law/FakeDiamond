#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:36:39 2023

@author: rl05
"""

import os
import os.path as op
import shutil
import numpy as np
import pandas as pd
from glob import glob

from scipy.stats import binom

import matplotlib.pyplot as plt
import seaborn as sns

import mne
# from mne.viz import tight_layout

from autoreject import Ransac  # noqa
# from autoreject.utils import interpolate_bads  # noqa

import config

project = config.project
cbu_repo_meg = '/megdata/cbu'
cbu_repo_mri = '/mridata/cbu'
project_repos = '/imaging/hauk/rl05'


# =============================================================================
# 
# =============================================================================

def check_preprocessed_files(preprocessed_data_path, subjects):
    for subject in subjects:
        subject = f'sub-{subject}'
        ica_path = op.join(preprocessed_data_path, subject, 'ica')
        epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
        ica_fname = glob(op.join(ica_path, '*ica.fif'))
        epoch_fname = glob(op.join(epoch_path, '*epo.fif'))
        print(subject)
        print('------')
        for path in [ica_path, epoch_path]:    
            if not op.isdir(path):
                os.makedirs(path, exist_ok=True)
        if ica_fname:
            print(f'ica: {subject} ica exists.')
            print(ica_fname)
        else:
            print(f'ica: {subject} ica not computed.')
        if epoch_fname:
            print(f'epoch: {subject} epoch fif exists.')
            print(epoch_fname)
        else:
            print(f'epoch: {subject} epoch fif not computed.')
        print()

def copy_log_files(project, subject):
    destination = op.join(project_repos, project, 'data', 'logs', subject)
    if not op.isdir(destination):  
        os.mkdir(destination)
    subject = subject.split('_')[1]
    for block_num in range(1,6):
        source = op.join('/home/rl05/ownCloud/projects/fake_diamond/scripts/stimulus_presentation/logs',
                              f'logfile_subject{subject}_block{block_num}.csv')
        if not op.isfile(op.join(destination, f'logfile_subject{subject}_block{block_num}.csv')):  
            shutil.copy2(source, destination)
            


# =============================================================================
# Continuous data preprocessing
# =============================================================================

def read_raws(preprocessed_data_path, subject, runs):
    print('Reading raws from all runs.')
    raws = []
    for run in runs:
        # print('run', run)
        raw_sss_fname = op.join(preprocessed_data_path, subject, 'maxfiltered', f'run-{run}', f'run{run}_sss_raw.fif')
        raw = mne.io.read_raw_fif(raw_sss_fname, verbose=False)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True, chpi=False)
        raw = raw.pick(picks)
        raws.append(raw)
    # check if all raws have same number of channels
    if not all([raw.info['nchan']==raws[0].info['nchan'] for raw in raws]):
        print('Raw fifs have different numbers of channels.')
        raws = [raw.drop_channels('STI010') if 'STI010' in raw.info['ch_names'] else raw for raw in raws]
    concatenated_raw = mne.concatenate_raws(raws)
    print('Done.')
    return concatenated_raw

def find_bad_chs(raw):
    event_id = config.event_id_semantic_word1
    events = mne.find_events(raw, min_duration=0.005)
    events = mne.pick_events(events, include=list(event_id.values()))
    bad_chs = []
    if raw.info['projs']:
        raw.del_proj()
    epochs = mne.Epochs(raw, events, picks='data', tmin=-0.5, tmax=1.4, reject=None,
                    baseline=(None, -0.3), detrend=0, preload=True, verbose=False)
    for ch_type in ['eeg','mag','grad']: # ransac can only do on ch_type at a time
        ransac = Ransac(verbose=True, picks=ch_type, n_jobs=1, random_state=42)
        ransac.fit(epochs)
        if ransac.bad_chs_:
            bad_chs.append(ransac.bad_chs_)
    return bad_chs

def plot_noisy_channel_detection(auto_scores, ch_type):
    # Your existing code here
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]
    scores = auto_scores["scores_noisy"][ch_subset]
    limits = auto_scores["limits_noisy"][ch_subset]
    bins = auto_scores["bins"]
    bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]
    data_to_plot = pd.DataFrame(
        data=scores,
        columns=pd.Index(bin_labels, name="Time (s)"),
        index=pd.Index(ch_names, name="Channel"),
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(
        f"Automated noisy channel detection: {ch_type}", fontsize=16, fontweight="bold"
    )
    sns.heatmap(data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0])
    [
        ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[0].set_title("All Scores", fontweight="bold")

    sns.heatmap(
        data=data_to_plot,
        vmin=np.nanmin(limits),
        cmap="Reds",
        cbar_kws=dict(label="Score"),
        ax=ax[1],
    )
    [
        ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[1].set_title("Scores > Limit", fontweight="bold")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig



# =============================================================================
# Epoched data preprocessing
# =============================================================================

def find_semantic_events(raw, subject='', semantic_triggers=''):
    print('Finding semantic events.')
    events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, 
                         initial_event=True, verbose=False)
    events_semantic = mne.pick_events(events, include=semantic_triggers)
    if subject == 'sub-35':
        # manually add an event because participant pressed on button box through these trials
        events_to_add = [[507731, 0, 71], # low
                         [511784, 0, 231]] # abstract baseline
        events_added = np.concatenate((events_semantic, np.array(events_to_add)))
        sort_indices = np.argsort(events_added[:,0])
        events_semantic = events_added[sort_indices]
    if len(events_semantic) < 900:
        try:
        #     print('Not finding 900 semantic events. Tring unit_cast method.')
        #     events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, 
        #                              uint_cast=True, verbose=False)
        #     events_semantic = mne.pick_events(events, include=semantic_triggers)
        #     assert(len(events_semantic) == 900)
        # except AssertionError:
        #     print('Not finding 900 semantic events. Tring consecutive=True.')
        #     events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, 
        #                              verbose=True, consecutive=False)
        #     events_semantic = mne.pick_events(events, include=semantic_triggers)
        #     assert(len(events_semantic) == 900)
        # except AssertionError:
            print('Not finding 900 semantic events. Trying a mask.')
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, 
                                     mask=8192, mask_type='not_and', verbose=False)
            events_semantic = mne.pick_events(events, include=semantic_triggers)
            assert(len(events_semantic) == 900)
        except AssertionError:
            print('Not finding 900 semantic events. Trying another mask.')
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, 
                                     mask=512, mask_type='not_and', verbose=False)
            events_semantic = mne.pick_events(events, include=semantic_triggers)
            assert(len(events_semantic) == 900)
        finally:
            print(f'Found {len(events_semantic)} semantic events.')
    return events_semantic

def correct_display_delay(events_semantic, events_photodiode):
    print('Checking photodiode-trigger delays.')
    match = []
    for time_semantic in events_semantic[:,0]:
        for time_photodiode in events_photodiode[:,0]:
            if abs(time_photodiode - time_semantic) <= 50:
                match.append(time_photodiode-time_semantic)
                
    delay = np.abs(np.asarray(match))
    print('===')
    print(f'Number of matches (total target triggers={len(events_semantic)}): {len(delay)}')
    print(f'Mean latency: {delay.mean():.3f} ms')
    print(f'SD latency: {delay.std():.3f} ms')
    print('===')

    if len(events_semantic) == len(delay):
        print('Matching number of events :) Correcting delay using actual delay!')
        events_semantic[:,0] += delay
    else:
        print('Mismatching number of events :( Correcting delay using mean delay.')
        events_semantic[:,0] += round(delay.mean())
        
    return events_semantic

def compare_reject_thresholds(autoreject, hardcoded):
    ch_types = ['grad','mag','eeg']
    composite = {ch_type: min(hardcoded[ch_type], autoreject[ch_type]) for ch_type in ch_types}
    print("Composite reject dictionary:", composite)
    return composite



# =============================================================================
# Decoding analysis helper functions
# =============================================================================

def extract_target_epochs(epochs, analysis=''):
    if analysis == 'lexicality': 
        epochs_word = epochs[(epochs.metadata.condition == 'mid') | 
                             (epochs.metadata.condition == 'concrete_subsective') | 
                             (epochs.metadata.condition == 'concrete_privative') | 
                             (epochs.metadata.condition == 'abstract_subsective') | 
                             (epochs.metadata.condition == 'abstract_privative')]
        # 'nord' is short for non-word 
        epochs_nord = epochs[(epochs.metadata.condition == 'low') | 
                             (epochs.metadata.condition == 'high') | 
                             (epochs.metadata.condition == 'concrete_baseline') | 
                             (epochs.metadata.condition == 'abstract_baseline')]
        epochs_word = mne.epochs.combine_event_ids(epochs_word, list(epochs_word.event_id.keys()), {'word': 99}, copy=True)
        epochs_nord = mne.epochs.combine_event_ids(epochs_nord, list(epochs_nord.event_id.keys()), {'nord': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_word, epochs_nord])
        epochs.crop(tmin=-0.2, tmax=0.6)
    return epochs



# =============================================================================
# Calculation helpers
# =============================================================================

def calculate_analytical_chance(n_trial, alpha=0.05, class_ratio=0.5):
    #n_trial = int(n_trial[0])
    chance = binom.ppf(q=1-alpha,  # 1 minus alpha 
                       n=n_trial,  # number of trials 
                       p=max(class_ratio, 1-class_ratio)
                       ) / n_trial # percent point function: inverse of cdf
    return chance

def calculate_avg_sem(array):
    avg = np.mean(array, axis=0)
    sem = np.std(array, axis=0) / np.sqrt(len(array))
    return avg, sem