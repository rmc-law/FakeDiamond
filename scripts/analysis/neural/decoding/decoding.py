#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Sep 13 17:14:31 2023

@author: rl05
'''

import argparse
import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 



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
    elif analysis == 'composition':
        epochs_phrase = epochs[(epochs.metadata.condition == 'concrete_subsective') | 
                               (epochs.metadata.condition == 'concrete_privative') | 
                               (epochs.metadata.condition == 'abstract_subsective') | 
                               (epochs.metadata.condition == 'abstract_privative')]
        epochs_word = epochs[(epochs.metadata.condition == 'concrete_baseline') | 
                             (epochs.metadata.condition == 'abstract_baseline')]
        epochs_phrase = mne.epochs.combine_event_ids(epochs_phrase, list(epochs_phrase.event_id.keys()), {'phrase': 99}, copy=True)
        epochs_word = mne.epochs.combine_event_ids(epochs_word, list(epochs_word.event_id.keys()), {'word': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_phrase, epochs_word])
    elif analysis == 'concreteness':
        epochs_concrete = epochs[(epochs.metadata.condition == 'concrete_subsective') | 
                                 (epochs.metadata.condition == 'concrete_privative') | 
                                 (epochs.metadata.condition == 'concrete_baseline')]
        epochs_abstract = epochs[(epochs.metadata.condition == 'abstract_subsective') | 
                                 (epochs.metadata.condition == 'abstract_privative') |
                                 (epochs.metadata.condition == 'abstract_baseline')]
        epochs_concrete = mne.epochs.combine_event_ids(epochs_concrete, list(epochs_concrete.event_id.keys()), {'concrete': 99}, copy=True)
        epochs_abstract = mne.epochs.combine_event_ids(epochs_abstract, list(epochs_abstract.event_id.keys()), {'abstract': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_concrete, epochs_abstract])
    elif analysis == 'denotation':
        epochs_privative = epochs[(epochs.metadata.condition == 'abstract_privative') | 
                                  (epochs.metadata.condition == 'concrete_privative')]
        epochs_subsective = epochs[(epochs.metadata.condition == 'abstract_subsective') | 
                                   (epochs.metadata.condition == 'concrete_subsective')]
        epochs_privative = mne.epochs.combine_event_ids(epochs_privative, list(epochs_privative.event_id.keys()), {'privative': 99}, copy=True)
        epochs_subsective = mne.epochs.combine_event_ids(epochs_subsective, list(epochs_subsective.event_id.keys()), {'subsective': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_privative, epochs_subsective])
    return epochs



# =============================================================================
# Define key decoding functions
# =============================================================================

def decode(X, y, classifier=''):
    if classifier == 'logistic':
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))
    elif classifier == 'svc':
        clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel='linear')))
    time_decod = SlidingEstimator(clf, scoring='roc_auc', verbose=False)
    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=42) # 5-fold cross validation, with random_state set for reproducibility
    scores = cross_val_multiscore(time_decod, X, y, cv=cv) 
    time_decod.fit(X, y) # retrieve spatial patterns and spatial filters for interpretability
    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    return scores, coef

def generalise(X, y, classifier=''):
    if classifier == 'logistic':
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))
    elif classifier == 'svc':
        clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel='linear')))
    elif classifier == 'naive_bayes':
        clf = make_pipeline(StandardScaler(), GaussianNB())
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', verbose=False)
    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=42) # 5-fold cross validation, with random_state set for reproducibility
    scores = cross_val_multiscore(time_gen, X, y, cv=cv)
    return scores





def main():
    
    project_repo = config.project_repo
    preprocessed_data_path = op.join(project_repo, 'data/preprocessed')

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True, help='Run subject-specific analysis')
    parser.add_argument('--analysis', type=str, required=True, help='Specify analysis (see README for more)')
    parser.add_argument('--classifier', type=str, required=True, help='Specify classifier: logistic or svc')
    # parser.add_argument('--generalise', action='store_true', help='Perform temporal generalisation')
    # parser.add_argument('--spatial', action='store_true', help='Whether the analysis is in source space')
    args = parser.parse_args()

    subject = f'sub-{args.subject}'
    analysis = args.analysis
    classifier = args.classifier
    analysis_output_dir = op.join(project_repo, f'scripts/analysis/neural/decoding/output/{analysis}/{subject}')
    if not op.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir, exist_ok=True)

    print(subject)

    print('Getting epochs.')
    epoch_fname = op.join(preprocessed_data_path, subject, 'epoch', f'{subject}_epo.fif')
    epochs = mne.read_epochs(epoch_fname, preload=False, verbose=False)

    # get subject trial info (logfile)
    log_fname = op.join(config.logs_dir, f'{subject}_logfile.csv')
    trial_info = pd.read_csv(log_fname)

    # get epochs drop log as a mask, then apply it to trial info as epochs.metadata
    epochs_drop_mask = [not bool(epoch) for epoch in epochs.drop_log]
    assert(epochs_drop_mask.count(True) == len(epochs.events)) # sanity
    epochs.metadata = trial_info[epochs_drop_mask]

    # set up decoder input
    epochs_target = extract_target_epochs(epochs, analysis=analysis)
    epochs_target.info['bads'] = [] # remove bads info for averaging spatial patterns later
    X = epochs_target.get_data()
    y = epochs_target.events[:,2]

    # calculate class ratio
    class_ratio = len(epochs_target[list(epochs_target.event_id.keys())[0]]) / (len(epochs_target))

    print('Fitting decoders.')
    scores, coef = decode(X, y, classifier=classifier)
    scores = np.mean(scores, axis=0) # average scores across cross-validation splits

    print('Saving decoding scores and spatial coefficients.')
    np.save(os.path.join(analysis_output_dir, f'scores_{analysis}_{classifier}_sensor_coef.npy'), coef)
    np.save(os.path.join(analysis_output_dir, f'scores_{analysis}_{classifier}_sensor_scores_temporal.npy'), scores)

    print('Plotting decoding scores.')
    fig, ax = plt.subplots()
    ax.plot(epochs_target.times, scores, label='score')
    ax.axhline(0.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(0.0, color='k', linestyle='-')
    ax.set_title(f'Decoding {analysis} in {subject} in sensor space (class ratio={class_ratio})')
    fig.savefig(op.join(analysis_output_dir, f'fig_decoding_{analysis}_scores.png'))
    plt.close(fig)

    print('Plotting spatial patterns.')
    evoked_time_decod = mne.EvokedArray(coef, epochs_target.info, tmin=epochs_target.times[0])
    joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
    if analysis == 'lexicality':
        times = np.arange(0.0, 0.5, 0.1)
    else:
        times = np.arange(0.6, 1.4, 0.1)
    fig_evokeds = evoked_time_decod.plot_joint(
        times=times, title='patterns', show=False, **joint_kwargs
    )
    for fig_evoked, ch_type in zip(fig_evokeds, ['eeg', 'mag', 'grad']):
        fig_evoked.savefig(op.join(analysis_output_dir, f'fig_decoding_{analysis}_spatial_patterns_{ch_type}.png'))
        plt.close(fig_evoked)

    print('Done.')

if __name__ == '__main__':
    main()