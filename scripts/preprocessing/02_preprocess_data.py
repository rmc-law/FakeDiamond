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

from mne import pick_types, find_events, Epochs
from mne.preprocessing import ICA, create_eog_epochs, read_ica
from autoreject import get_rejection_threshold

import config
import helper

subjects = config.subject_ids
runs = [1,2,3,4,5]

l_freq = config.l_freq
h_freq = config.h_freq

preprocessed_data_path = '/imaging/hauk/rl05/fake_diamond/data/preprocessed'

helper.check_preprocessed_files(preprocessed_data_path, subjects)

subject = input('sub to preprocess: ')
subject = f'sub-{subject}'
ica_path = op.join(preprocessed_data_path, subject, 'ica')
ica_fname = op.join(ica_path, f'{subject}_ica.fif')
epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')

# concatenate raws from different runs/blocks
raw = helper.read_raws(preprocessed_data_path, subject, runs=runs)
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
        autoreject_bad_chs = [line.strip() for line in file.readlines()]
else:
    autoreject_bad_chs = helper.find_bad_chs(raw)
    with open(autoreject_bad_chs_fname, 'w') as file:
        file.writelines(f'{bad_ch}\n' for bad_ch in autoreject_bad_chs)
raw.info['bads'] += autoreject_bad_chs
raw.info['bads'] = list(set(raw.info['bads'])) # if there are overlaps
print('autoreject detected bad channels:', autoreject_bad_chs)
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
        ica_overlay = ica.plot_overlay(raw, picks=ch_type, start=8., stop=10.)
        ica_overlay_fname = op.join(ica_path, f'ica_overlay_{ch_type}.png')
        ica_overlay.savefig(ica_overlay_fname)
    
    print('ica components marked for exclusion: ', ica.exclude)
    ica.save(ica_fname, overwrite=True)

ica.apply(raw)



# =============================================================================
# Epoch data and reject bad epochs
# =============================================================================

# help find bad channels using butterfly mode
# raw.plot(butterfly=True)

if raw.info['bads']:
    print('Interpolating bads.')
    raw.interpolate_bads(reset_bads=False, mode='accurate')
print('Bad channels: ', raw.info['bads'])

# epoch parameters
tmin = -0.5 # include pre-fixation period for noise covariance
tmax = 1.4  # include both words
baseline = (-0.5,-0.3)
picks = pick_types(raw.info, meg=True, eeg=True, exclude=())  # not excluding bads to ensure bads info stay in epochs objects

# find semantic and photodiode events
events_semantic = helper.find_semantic_events(raw, subject=subject, semantic_triggers=list(config.event_id_semantic_word1.values()))
assert(len(events_semantic) == 900) # sanity
events_photodiode = find_events(raw, stim_channel='STI010', min_duration=0.002, verbose=False)

# shift events by photodiode delay
events_semantic_corrected = helper.correct_display_delay(events_semantic, events_photodiode)

# segment continuous data into epoch
print('Segmenting into epochs.')
epochs = Epochs(raw, events_semantic_corrected, event_id=config.event_id_semantic_word1, 
                tmin=tmin, 
                tmax=tmax, 
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
    with open(autoreject_threshold_fname, 'rb') as file:
        autoreject_thresholds = json.load(file)    
    print('empirically determined thresholds: ', autoreject_thresholds)
else:
    autoreject_thresholds = get_rejection_threshold(epochs, random_state=42)
    print('The rejection dictionary is %s' % autoreject_thresholds)
    with open(autoreject_threshold_fname, 'w') as file:
        json.dump(autoreject_thresholds, file)
        
epochs.drop_bad(reject=autoreject_thresholds)

# if automatic rejection did not drop any epochs, use hardcoded threshold
if len(epochs) == 900: 
    composite_rejection_thresholds = helper.compare_reject_thresholds(autoreject_thresholds, config.hardcoded_thresholds)
    epochs.drop_bad(reject=composite_rejection_thresholds)
    

drop_log_fname = op.join(epoch_path, 'epochs_drop_log.pkl')
with open(drop_log_fname, 'wb') as file:
    pickle.dump(epochs.drop_log, file)

figs_plot_drop_log = epochs.plot_drop_log(show=False)
figs_plot_drop_log.savefig(op.join(epoch_path, 'epochs_drop_log_stats.png'))

# plot joint to see if it's quite clean
figs_evoked = epochs.average().plot_joint(times=[0, 0.17, 0.6, 0.77, 1.0],
                                          exclude='bads',
                                          show=False)
for fig_evoked, ch_type in zip(figs_evoked, ['eeg', 'mag', 'grad']):
    fig_evoked.savefig(op.join(epoch_path, f'fig_sensor_{ch_type}_grand_ave.png'))


# visually inspect epochs
epochs.plot(butterfly=True)


epochs.save(epoch_fname, overwrite=True)
