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
# import pandas as pd
from glob import glob

# import matplotlib.pyplot as plt
# import seaborn as sns

from mne import find_events, pick_events, concatenate_raws, Epochs, pick_types
from mne.io import read_raw_fif
# from mne.viz import tight_layout
from autoreject import Ransac  # noqa
# from autoreject.utils import interpolate_bads  # noqa


import config

project = config.project
cbu_repo_meg = '/megdata/cbu'
cbu_repo_mri = '/mridata/cbu'
project_repos = '/imaging/hauk/rl05'


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


def read_raws(preprocessed_data_path, subject, runs):
    print('Reading raws from all runs.')
    raws = []
    for run in runs:
        # print('run', run)
        raw_sss_fname = op.join(preprocessed_data_path, subject, 'maxfiltered', f'run-{run}', f'run{run}_sss_raw.fif')
        raw = read_raw_fif(raw_sss_fname, verbose=False)
        picks = pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True, chpi=False)
        raw = raw.pick(picks)
        raws.append(raw)
    # check if all raws have same number of channels
    if not all([raw.info['nchan']==raws[0].info['nchan'] for raw in raws]):
        print('Raw fifs have different numbers of channels.')
        raws = [raw.drop_channels('STI010') if 'STI010' in raw.info['ch_names'] else raw for raw in raws]
    concatenated_raw = concatenate_raws(raws)
    print('Done.')
    return concatenated_raw

def find_bad_chs(raw):
    event_id = config.event_id_semantic_word1
    events = find_events(raw, min_duration=0.005)
    events = pick_events(events, include=list(event_id.values()))
    bad_chs = []
    if raw.info['projs']:
        raw.del_proj()
    epochs = Epochs(raw, events, picks='data', tmin=-0.5, tmax=1.4, reject=None,
                    baseline=(None, -0.3), detrend=0, preload=True, verbose=False)
    for ch_type in ['eeg','mag','grad']: # ransac can only do on ch_type at a time
        ransac = Ransac(verbose=True, picks=ch_type, n_jobs=1, random_state=42)
        ransac.fit(epochs)
        if ransac.bad_chs_:
            bad_chs.append(ransac.bad_chs_)
    return bad_chs


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
            

# def plot_noisy_channel_detection(auto_scores, ch_type):
#     # Your existing code here
#     ch_subset = auto_scores["ch_types"] == ch_type
#     ch_names = auto_scores["ch_names"][ch_subset]
#     scores = auto_scores["scores_noisy"][ch_subset]
#     limits = auto_scores["limits_noisy"][ch_subset]
#     bins = auto_scores["bins"]
#     bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]
#     data_to_plot = pd.DataFrame(
#         data=scores,
#         columns=pd.Index(bin_labels, name="Time (s)"),
#         index=pd.Index(ch_names, name="Channel"),
#     )

#     fig, ax = plt.subplots(1, 2, figsize=(12, 8))
#     fig.suptitle(
#         f"Automated noisy channel detection: {ch_type}", fontsize=16, fontweight="bold"
#     )
#     sns.heatmap(data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0])
#     [
#         ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
#         for x in range(1, len(bins))
#     ]
#     ax[0].set_title("All Scores", fontweight="bold")

#     sns.heatmap(
#         data=data_to_plot,
#         vmin=np.nanmin(limits),
#         cmap="Reds",
#         cbar_kws=dict(label="Score"),
#         ax=ax[1],
#     )
#     [
#         ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
#         for x in range(1, len(bins))
#     ]
#     ax[1].set_title("Scores > Limit", fontweight="bold")

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     return fig


# def filter_raw(raw):
#     # this section visualises the effects of filtering on sensor signals
#     # make a copy of the original channels
#     print ('Making a copy of raw data...')
#     x0 = raw.copy().pick(['MEG1113']).crop(tmin = 60., tmax = 120.) # unfiltered

#     # apply band-pass filter to raw
#     l_freq = 0 # high-pass filter in Hz
#     h_freq = 40 # low-pass filter in Hz
#     raw_filtered = raw.filter(l_freq, h_freq, method='fir', fir_window='hamming', fir_design='firwin')
#     # the method argument is default to fir in MNE 0.17 but iir in previous versions

#     # then make a copy of filtered channels
#     print ('Making a copy of filtered data...')
#     x1 = raw.copy().pick(['MEG1113']).crop(tmin = 60., tmax = 120.) # filtered
#     # plot these two copies using matplotlib
#     from scipy import fftpack
#     import matplotlib.pyplot as plt
#     print('Plotting comparison between filtered and unfiltered data...')
#     x0, times = x0[:,:]
#     x0 = x0[0]
#     x1, times = x1[:,:]
#     x1 = x1[0]

#     # you might need to use plt.ion() to turn on interactive mode
#     fig_filter_effect, axes = plt.subplots(1,3, figsize = (12,6))
#     axes[0].plot(times, x1)
#     axes[0].plot(times, x0 - 1e-12)
#     axes[0].set(xlabel = 'Time (sec)', xlim = [times[0], times[1000]])
#     X0 = fftpack.fft(x0)
#     X1 = fftpack.fft(x1)
#     freqs = fftpack.fftfreq(len(x0), 1./raw.info['sfreq'])
#     mask = freqs >= 0
#     X0 = X0[mask]
#     X1 = X1[mask]
#     freqs = freqs[mask]
#     axes[1].plot(freqs, 20 * np.log10(np.maximum(np.abs(X1), 1e-16)))
#     axes[1].plot(freqs, 20 * np.log10(np.maximum(np.abs(X0), 1e-16)))
#     axes[1].set(xlim = [0, 60], xlabel = 'Frequency (Hz)', ylabel = 'Magnitude (dB)')
#     angles = (np.angle(X1) - np.angle(X0)) / (2 * np.pi)
#     angles[np.absolute(angles) > 0.9] = angles[np.absolute(angles) > 0.9] + 1.
#     axes[2].plot(freqs, angles)
#     #axes[2].plot(freqs[1:], np.abs(np.remainder(np.angle(X1[1:]), 2. * np.pi) - np.remainder(np.angle(X0[1:]), 2 * np.pi)) / (2 * np.pi * freqs[1:]))
#     #axes[2].plot(freqs, np.angle(X0))
#     axes[2].set(xlim = [0, 60], xlabel = 'Frequency (Hz)', ylabel = 'Phase delay [cycle]')
#     tight_layout()
#     return raw_filtered, fig_filter_effect


def shift_event_photodiode(events, target_triggers=None, photodiode_trigger=512):
    
    print('Checking photodiode-trigger delays.')
    events_semantic = pick_events(events, include=target_triggers)
    events_photodiode = pick_events(events, include=photodiode_trigger)
    
    match = []
    for time_semantic in events_semantic[:,0]:
        for time_photodiode in events_photodiode[:,0]:
            if abs(time_photodiode - time_semantic) <= 50:
                match.append([time_semantic, time_photodiode, time_photodiode-time_semantic])
                
    delay = np.abs(np.asarray(match)[:,2])
    print('===')
    print(f'Number of matches (total target triggers={len(events_semantic)}): {len(delay)}')
    print(f'Mean latency: {delay.mean():.3f} ms')
    print(f'SD latency: {delay.std():.3f} ms')
    print('===')
    return delay

def calculate_avg_sem(array):
    avg = np.mean(array, axis=0)
    sem = np.std(array, axis=0) / np.sqrt(len(array))
    return avg, sem