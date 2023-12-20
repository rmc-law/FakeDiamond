#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:49:51 2023

@author: rl05

Suppress artifact - filter, ICA, identify bads, epoch, epoch rejection
"""

import os.path as op
import pickle
import json

from mne import pick_types, find_events, pick_events, Epochs
from mne.preprocessing import ICA, create_eog_epochs, read_ica
from autoreject import get_rejection_threshold

from eelbrain.gui import select_epochs
from eelbrain.load import unpickle

import config
from helper import check_preprocessed_files, read_raws, shift_event_photodiode, find_bad_chs

subjects = config.subject_ids
runs = [1,2,3,4,5]

l_freq = config.l_freq
h_freq = config.h_freq

preprocessed_data_path = '/imaging/hauk/rl05/fake_diamond/data/preprocessed'

check_preprocessed_files(preprocessed_data_path, subjects)

subject = input('sub to preprocess: ')
subject = f'sub-{subject}'
ica_path = op.join(preprocessed_data_path, subject, 'ica')
ica_fname = op.join(ica_path, f'{subject}_ica.fif')
epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
epoch_rej_fname = op.join(epoch_path, f'{subject}_epo_rej.pickled')

# concatenate raws from different runs/blocks
raw = read_raws(preprocessed_data_path, subject, runs=runs)
raw.load_data(verbose=False)



# =============================================================================
# Suppress artifacts
# =============================================================================

# filter 
print(f'Applying bandpass filter: {l_freq}-{h_freq} Hz')
raw.filter(l_freq=l_freq, 
           h_freq=h_freq, 
           picks=['meg','eeg'], 
           verbose=False, 
           n_jobs=-1)

raw.info['bads'] += config.bad_chs[subject[-2:]]['eeg']
raw.info['bads'] += config.bad_chs[subject[-2:]]['meg']

# use autoreject to detect bad channels
autoreject_bad_chs_fname = op.join(epoch_path, 'autoreject_bad_chs.txt')
if op.exists(autoreject_bad_chs_fname):
    with open(autoreject_bad_chs_fname, 'r') as file:
        content = file.readlines()
    autoreject_bad_chs = [line.strip() for line in content]
    print('autoreject detected bad channels: ', autoreject_bad_chs)
else:
    autoreject_bad_chs = find_bad_chs(raw)
    if autoreject_bad_chs:
        with open(autoreject_bad_chs_fname, 'w') as file:
            for bad_ch in autoreject_bad_chs[0]:
                file.write(f'{bad_ch}\n')
        autoreject_bad_chs = autoreject_bad_chs[0]
raw.info['bads'] += autoreject_bad_chs
print('Bad channels: ', raw.info['bads'])

# set eeg reference
raw = raw.set_eeg_reference(ref_channels='average', projection=True)

# ica to attenuate eye-related components
if op.exists(ica_fname):
    ica = read_ica(ica_fname)
    print(ica)
else:
    reject = dict(grad=4e-10, mag=1e-11, eeg=1e-3)
    ica = ICA(n_components=0.95, random_state=42, method='fastica', verbose=True)
    picks = pick_types(raw.info, meg=True, eeg=True, eog=False, exclude='bads')
    ica.fit(raw, decim=2, reject=reject, picks=picks)
    del picks 
    
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    if eog_indices:
        # barplot of ICA component "EOG match" scores
        fig_eog_scores = ica.plot_scores(eog_scores)
        fig_eog_scores.savefig(op.join(ica_path, 'ica_eog_scores.png'))
        
        fig_eog_properties = ica.plot_properties(raw, picks=eog_indices)
        fig_eog_properties[0].savefig(op.join(ica_path, 'ica_eog_properties.png'))
        
        eog_evoked = create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        figs_eog_evoked = eog_evoked.plot_joint()
        for fig, ch_type in zip(figs_eog_evoked, ['eeg','mag','grad']):
            fig.savefig(op.join(ica_path, f'eog_evoked_{ch_type}.png'))
            
        fig_eog_sources = ica.plot_sources(eog_evoked)
        fig_eog_sources.savefig(op.join(ica_path, 'ica_eog_sources.png'))
    
    ica.plot_components()
    ica.plot_sources(raw)
    
    # visual inspection after ICA component rejection
    for ch_type in ['eeg','meg']:
        ica_overlay = ica.plot_overlay(raw, picks=ch_type, start=4., stop=7.)
        ica_overlay_fname = op.join(ica_path, f'ica_overlay_{ch_type}.png')
        ica_overlay.savefig(ica_overlay_fname)
    
    print('ica components marked for exclusion: ', ica.exclude)
    ica.save(ica_fname, overwrite=True)

ica.apply(raw)



# =============================================================================
# Epoch data and reject bad epochs
# =============================================================================

print('Bad channels: ', raw.info['bads'])

# help find bad channels using butterfly mode
raw.plot(butterfly=True)

# if raw.info['bads']:
#     print('Interpolating bads.')
#     raw.interpolate_bads(reset_bads=True, mode='accurate')

# epoch parameters
tmin = -0.5 # include pre-fixation period for noise covariance
tmax_epo = 1.4 # include both words
baseline = (-0.5,-0.3)
picks = pick_types(raw.info, 
                   meg=True, 
                   eeg=True, 
                   exclude=() # not excluding bads to ensure bads info stay in epochs objects
                   ) 
events_all = find_events(raw, stim_channel='STI101', min_duration=0.005)

# shift events by photodiode delay
delay = shift_event_photodiode(events_all, 
                               target_triggers=list(config.event_id_semantic.values()),
                               photodiode_trigger=512
                               )
event_id = config.event_id_semantic
events_semantic = pick_events(events_all, include=list(event_id.values()))

# Check to see if photodiode and target events match:
if len(events_semantic) == len(delay):
    print('Matching number of events! Adding the shift!')
    events_semantic[:,0] = events_semantic[:,0] + delay
else:
    print('Mismatching number of events :( using mean shift')
    events_semantic[:,0] = events_semantic[:,0] + round(delay.mean())


event_id = config.event_id_semantic_word1
events = pick_events(events_semantic, include=list(event_id.values()))

# segment continuous data into epoch
print('Segmenting into epochs.')
epochs = Epochs(raw, events, event_id=event_id, 
                tmin=tmin, 
                tmax=tmax_epo, 
                baseline=baseline, 
                preload=True, 
                picks=picks, 
                reject=None, # empirically determine rejection threshold below
                verbose=False)
print('Number of epochs found: ', len(epochs))


print('Resampling epochs to 250Hz for speed.')
epochs = epochs.resample(sfreq=epochs.info['sfreq']/4)



# automatic rejection of bad epochs
autoreject_threshold_fname = op.join(epoch_path, 'autoreject_threshold.json')
if op.exists(autoreject_threshold_fname):
    with open(autoreject_threshold_fname, 'r') as file:
        reject = json.load(autoreject_threshold_fname)    
    print('empirically determined thresholds: ', reject)
else:
    reject = get_rejection_threshold(epochs, random_state=42)
    print('The rejection dictionary is %s' % reject)
    with open(autoreject_threshold_fname, 'w') as file:
        json.dump(reject, file)
        
epochs.drop_bad(reject=reject)
evoked = epochs.average()
figs_evoked = evoked.plot_joint(show=False, times=[0.17,0.77,1.0])
# figs_evoked.savefig(op.join(epoch_path, f'{subject}_autoreject_grand-evoked.png'))

drop_log_fname = op.join(epoch_path, 'epochs_drop_log.pkl')
with open(drop_log_fname, 'wb') as file:
    pickle.dump(epochs.drop_log, file)

figs_plot_drop_log = epochs.plot_drop_log(show=False)
figs_plot_drop_log.savefig(op.join(epoch_path, 'epochs_drop_log_stats.png'))


# # epoch rejection using eelbrain
# select_epochs(epochs, mark=['MEG0112'])

# rejfile = unpickle(epoch_rej_fname)
# rejs = rejfile['accept'].x
# epochs = epochs[rejs]

# plot joint to see if it's quite clean
figs_evoked = epochs.average().plot_joint()

epochs.save(epoch_fname, overwrite=True)
