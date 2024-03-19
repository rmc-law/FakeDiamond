#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Sep 13 17:14:31 2023

@author: rl05
'''

import argparse
import os
import os.path as op
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from mne import read_source_morph
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

n_splits = 10

def extract_target_epochs(epochs, analysis=''):
    if analysis == 'composition':
        epochs = epochs['experiment.str.startswith("compose")']
        epochs_phrase = epochs['composition.str.startswith("phrase")']
        epochs_word = epochs['composition.str.startswith("word")']
        epochs_phrase = mne.epochs.combine_event_ids(epochs_phrase, list(epochs_phrase.event_id.keys()), {'phrase': 99}, copy=True)
        epochs_word = mne.epochs.combine_event_ids(epochs_word, list(epochs_word.event_id.keys()), {'word': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_phrase, epochs_word])
    elif analysis == 'concreteness':
        epochs_concrete = epochs['concrete']
        epochs_abstract = epochs['abstract']
        epochs_concrete = mne.epochs.combine_event_ids(epochs_concrete, list(epochs_concrete.event_id.keys()), {'concrete': 99}, copy=True)
        epochs_abstract = mne.epochs.combine_event_ids(epochs_abstract, list(epochs_abstract.event_id.keys()), {'abstract': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_concrete, epochs_abstract])
    elif analysis == 'concreteness_word':
        epochs = epochs['baseline']
    elif analysis == 'denotation':
        epochs_subsective = epochs['subsective']
        epochs_privative = epochs['privative']
        epochs_subsective = mne.epochs.combine_event_ids(epochs_subsective, list(epochs_subsective.event_id.keys()), {'subsective': 99}, copy=True)
        epochs_privative = mne.epochs.combine_event_ids(epochs_privative, list(epochs_privative.event_id.keys()), {'privative': 100}, copy=True)
        epochs = mne.concatenate_epochs([epochs_subsective, epochs_privative])
    elif analysis == 'specificity': 
        epochs = epochs['experiment.str.startswith("specificity")']
    elif analysis == 'specificity_word': 
        epochs = epochs['composition.str.startswith("word") and experiment.str.startswith("specificity")']
    elif analysis == 'denotation_cross_condition':
        train_test_pairs = [(('baseline','privative'),('subsective')),
                            (('baseline','subsective'),('privative'))]
        epochs_train_test_pairs = []
        for train_pair, test_target in train_test_pairs:
            epochs_train_concrete = epochs[(epochs.metadata.condition == f'concrete_{train_pair[0]}') | 
                                           (epochs.metadata.condition == f'concrete_{train_pair[1]}')]
            epochs_train_abstract = epochs[(epochs.metadata.condition == f'abstract_{train_pair[0]}') |
                                           (epochs.metadata.condition == f'abstract_{train_pair[1]}')]
            epochs_train_concrete = mne.epochs.combine_event_ids(epochs_train_concrete, list(epochs_train_concrete.event_id.keys()), {'concrete': 99}, copy=True)
            epochs_train_abstract = mne.epochs.combine_event_ids(epochs_train_abstract, list(epochs_train_abstract.event_id.keys()), {'abstract': 100}, copy=True)
            epochs_train = mne.concatenate_epochs([epochs_train_concrete, epochs_train_abstract])
            epochs_train.load_data().crop(tmin=0.6, tmax=1.4).resample(100)
            epochs_test = epochs[(epochs.metadata.condition == f'concrete_{test_target}') | 
                                 (epochs.metadata.condition == f'abstract_{test_target}')]
            epochs_test.load_data().crop(tmin=0.6, tmax=1.4).resample(100)
            epochs_train_test_pair = [epochs_train, epochs_test]
            epochs_train_test_pairs.append(epochs_train_test_pair)
        epochs = epochs_train_test_pairs
    if analysis == 'denotation_cross_condition':
        pass
    else: 
        epochs.load_data().resample(100)
    return epochs



# =============================================================================
# Decoding functions
# =============================================================================

def choose_pipelines(classifier='', analysis=''):
    if classifier == 'logistic':
        # clf = LogisticRegressionCV(Cs=10, solver='liblinear', class_weight='balanced', multi_class='auto', cv=5, max_iter=10000)
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=10000)
        clf = make_pipeline(RobustScaler(), LinearModel(clf))
    elif classifier == 'logisticCV':
        clf = LogisticRegressionCV(Cs=10, solver='liblinear', class_weight='balanced', multi_class='auto', cv=5, max_iter=10000)
        clf = make_pipeline(RobustScaler(), LinearModel(clf))
    elif classifier == 'svc':
        clf = make_pipeline(RobustScaler(), LinearModel(SVC(kernel='linear')))
    elif classifier == 'naive_bayes':
        clf = make_pipeline(RobustScaler(), GaussianNB())
    return clf

def choose_scorer(analysis=''):
    if analysis == 'specificity':
        scorer = 'roc_auc_ovo_weighted' # three class problem
    else:
        scorer = 'roc_auc'
    return scorer

def decode_diagonal(X, y, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier, analysis=analysis)
    scorer = choose_scorer(analysis=analysis)
    time_decod = SlidingEstimator(clf, scoring=scorer, verbose=True)
    cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42) 
    scores = cross_val_multiscore(time_decod, X, y, cv=cv) 
    time_decod.fit(X, y) # retrieve spatial patterns and spatial filters for interpretability
    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    return scores, coef

def decode_generalise(X, y, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    time_gen = GeneralizingEstimator(clf, scoring=scorer, verbose=True)
    cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42) 
    scores = cross_val_multiscore(time_gen, X, y, cv=cv)
    return scores

# cross condition decoding: train on condition A, test on condition B
def decode_diagonal_train(X_train, y_train, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    estimator = SlidingEstimator(clf, scoring=scorer, verbose=True)
    estimator.fit(X_train, y_train)
    return estimator

def decode_diagonal_test(X_test, y_test, estimator=None, classifier=''):
    scores = estimator.score(X_test, y_test)
    if classifier == 'naive_bayes':
        return scores
    else:
        coef = get_coef(estimator, 'patterns_', inverse_transform=True)
        return scores, coef

def decode_generalise_train(X_train, y_train, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    estimator = GeneralizingEstimator(clf, scoring=scorer, verbose=True)
    estimator.fit(X_train, y_train)
    return estimator

def decode_generalise_test(X_test, y_test, estimator=None):
    scores = estimator.score(X_test, y_test)
    return scores

# =============================================================================
# Plotting functions
# =============================================================================

def plot_scores(times=None, scores=None, analysis='', subject='', class_ratio=''):
    print('Plotting decoding scores.')
    fig, ax = plt.subplots()
    ax.plot(times, scores, label='score')
    ax.axhline(0.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC') 
    ax.legend()
    # ax.axvline(0.0, color='k', linestyle='-')
    ax.set_title(f'Decoding {analysis} in {subject} (class ratio={class_ratio:.2f})')
    plt.tight_layout()
    return fig, ax

def plot_patterns(coef=None, info=None, tmin=0., times=[]):
    print('Plotting spatial patterns.')
    evoked = mne.EvokedArray(coef, info, tmin=tmin)
    joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
    fig_evokeds = evoked.plot_joint(times=times, title='patterns', show=False, **joint_kwargs)
    return fig_evokeds



def main():
    
    data_dir = '/imaging/hauk/rl05/fake_diamond/data'
    preprocessed_data_path = op.join(data_dir, 'preprocessed')
    decoding_output_dir = '/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding/output'

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, required=True, help='Run subject-specific analysis')
    parser.add_argument('--analysis', type=str, required=True, default=None, help='Specify analysis (see README for more)')
    parser.add_argument('--classifier', type=str, required=True, default='logistic', help='Specify classifier: logistic or svc')
    parser.add_argument('--data_type', type=str, required=True, default='MEEG', help='MEEG or MEG or source: ')
    parser.add_argument('--generalise', action='store_true', default=False, help='Perform temporal generalisation?')
    args = parser.parse_args()

    subject = f'sub-{args.subject}'
    analysis = args.analysis
    print(analysis)
    classifier = args.classifier
    print(classifier)
    data_type = args.data_type
    print(data_type)
    generalise = args.generalise
    print('generalise: ', generalise)


    print()
    print(subject)
    start_time = time.time()

    print('Getting epochs and metadata.')
    epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
    epochs = mne.read_epochs(op.join(epoch_path, f'{subject}_epo.fif'), preload=False, verbose=False)
    epochs.info['bads'] = [] # remove bads info for averaging spatial patterns later
    trial_info = pd.read_csv(op.join(data_dir, 'logs', f'{subject}_logfile.csv'))

    # get epochs drop log as a mask, then apply it to trial info as epochs.metadata
    epochs_drop_mask = [not bool(epoch) for epoch in epochs.drop_log]
    assert(epochs_drop_mask.count(True) == len(epochs.events)) # sanity
    epochs.metadata = trial_info[epochs_drop_mask]

    # set up decoder input
    epochs_target = extract_target_epochs(epochs, analysis=analysis)
    if data_type == 'source':
        meg_sensor_type = input('MEEG or MEG: ')
        stc_path = op.join(data_dir, 'stcs', subject)
        if op.exists(stc_path):
            print(f'{subject} stc path exists.')
        else:
            print(f'Making {subject} stc path.')
            os.makedirs(stc_path, exist_ok=True)
        snr = 2.0 # SNR assumption for evoked; for epoch use 2
        lambda2 = 1.0 / snr ** 2
        method = 'MNE'
        inv_fname = op.join(epoch_path, f'{subject}_{meg_sensor_type}_inv.fif')
        inv = read_inverse_operator(inv_fname, verbose=False)
        stcs = apply_inverse_epochs(epochs, inv, lambda2, method=method)
        morph = read_source_morph(op.join(data_dir, f'mri/{subject}/{subject}-morph.h5'))
        stcs = [morph.apply(stc) for stc in stcs]
        X = np.stack([stc._data for stc in stcs])
    else:
        if data_type == 'MEG':
            epochs_target = epochs_target.pick(picks='meg')
        if analysis == 'denotation_cross_condition':
            X_train = [epochs_target[0][0].get_data(), epochs_target[1][0].get_data()]
            y_train = [epochs_target[0][0].events[:,2], epochs_target[1][0].events[:,2]]
            X_test = [epochs_target[0][1].get_data(), epochs_target[1][1].get_data()]
            y_test = [epochs_target[0][1].events[:,2], epochs_target[1][1].events[:,2]]
            class_ratios = [len(epochs_target[0][1]['concrete/subsective']) / (len(epochs_target[0][1]['subsective'])),
                            len(epochs_target[1][1]['concrete/privative']) / (len(epochs_target[1][1]['privative'])),]
        else:
            X = epochs_target.get_data()
            y = epochs_target.events[:,2]
            class_ratio = len(epochs_target[list(epochs_target.event_id.keys())[0]]) / (len(epochs_target))

        
    # some plotting-related parameters
    if analysis == 'denotation_cross_condition':
        times = np.arange(0.6, 1.4, 0.2)
    else:
        times = np.arange(-0.2, 1.4, 0.2)

    if data_type == 'MEEG':
        ch_types = ['eeg', 'mag', 'grad']
    elif data_type == 'MEG':
        ch_types = ['mag', 'grad']


    print(f'\nStarting decoding. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min')
    if generalise:
        analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/timegen/{classifier}/{data_type}/{subject}')
        if not op.exists(analysis_output_dir):
            os.makedirs(analysis_output_dir, exist_ok=True)
        if analysis == 'denotation_cross_condition':
            for i, condition in enumerate(['subsective','privative']):
                time_gen = decode_generalise_train(X_train[i], y_train[i], classifier=classifier)
                scores = decode_generalise_test(X_test[i], y_test[i], estimator=time_gen)
                np.save(os.path.join(analysis_output_dir, f'scores_time_gen_{data_type}_test-on-{condition}.npy'), scores)
        else:
            scores = decode_generalise(X, y, analysis=analysis, classifier=classifier)
            scores = np.mean(scores, axis=0) # average scores across cross-validation splits
            np.save(os.path.join(analysis_output_dir, f'scores_time_gen_{data_type}.npy'), scores)
    else:
        analysis_output_dir = op.join(decoding_output_dir, f'{analysis}/diagonal/{classifier}/{data_type}/{subject}')
        if not op.exists(analysis_output_dir):
            os.makedirs(analysis_output_dir, exist_ok=True)

        if analysis == 'denotation_cross_condition':
            for i, condition in enumerate(['subsective','privative']):
                time_decod = decode_diagonal_train(X_train[i], y_train[i], classifier=classifier)
                scores, coef = decode_diagonal_test(X_test[i], y_test[i], estimator=time_decod, classifier=classifier)
                np.save(os.path.join(analysis_output_dir, f'scores_time_decod_{data_type}_test-on-{condition}.npy'), scores)
                np.save(os.path.join(analysis_output_dir, f'scores_coef_{data_type}_test-on-{condition}.npy'), coef)
                class_ratio = class_ratios[i]

                fig, _ = plot_scores(times=epochs_target[0][1].times, scores=scores, 
                                      analysis=analysis, subject=subject, class_ratio=class_ratio)
                fig.savefig(op.join(analysis_output_dir, f'fig_time_decod_test-on-{condition}.png'))
                plt.close(fig)
        else:
            scores, coef = decode_diagonal(X, y, analysis=analysis, classifier=classifier)
            scores = np.mean(scores, axis=0) # average scores across cross-validation splits
            np.save(os.path.join(analysis_output_dir, f'scores_time_decod_{data_type}.npy'), scores)
            np.save(os.path.join(analysis_output_dir, f'scores_coef_{data_type}.npy'), coef)

            fig, _ = plot_scores(times=epochs_target.times, scores=scores, 
                                    analysis=analysis, subject=subject, class_ratio=class_ratio)
            fig.savefig(op.join(analysis_output_dir, 'fig_time_decod.png'))
            plt.close(fig)

        fig_evokeds = plot_patterns(coef, info=epochs_target.info, tmin=epochs_target.times[0], times=times)
        for fig_evoked, ch_type in zip(fig_evokeds, ch_types):
            fig_evoked.savefig(op.join(analysis_output_dir, f'fig_patterns_{data_type}_{ch_type}.png'))
            plt.close(fig_evoked)
    
    print(f'Finished decoding. Elapsed time since the script began: {(time.time()-start_time)/60:.2f}min\n')

if __name__ == '__main__':
    main()