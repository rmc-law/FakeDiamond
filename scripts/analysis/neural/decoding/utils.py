#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:01:09 2024

@author: rl05
"""

import os.path as op
import numpy as np
import yaml
from itertools import compress
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)


decoding_dir = '/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding'
# read in analysis config
with open(op.join(decoding_dir, 'analysis_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)


trigger_scheme = dict(
    low=71,
    mid=81,
    high=91,
    concrete_subsective=111,
    concrete_privative=121,
    concrete_baseline=131,
    abstract_subsective=211,
    abstract_privative=221,
    abstract_baseline=231
)

def get_contrasts(config, analysis_name):
    for analysis in config['analyses']:
        if analysis['name'] == analysis_name:
            target_contrasts = [contrast['name'] for contrast in analysis['contrasts']]
            conditions_to_merge = [contrast['conditions'] for contrast in analysis['contrasts']]
            return target_contrasts, conditions_to_merge

def map_event_ids(trigger_scheme, conditions):
    event_ids = [trigger_scheme[condition] for condition in conditions if condition in trigger_scheme]
    return event_ids

def merge_contrast_events(events, target_contrasts, conditions_to_merge):
    for i in range(len(target_contrasts)):    
        events = mne.merge_events(events, map_event_ids(trigger_scheme, conditions_to_merge[i]), i)
    return events

def get_analysis_X_y(trials, events, metadata, analysis_name='', spatial=False):
    # input data and events both contain all trials
    
    if spatial:
        X = np.stack([stc._data for stc in trials]) # stc data for all trials
        if analysis_name.startswith('denotation_cross_condition') or analysis_name.startswith('concreteness_'):
            if X[0].shape[1] == 160:
                X = X[:,:,80:] # only analyse second word for this analysis
    else:
        if analysis_name.startswith('denotation_cross_condition') or analysis_name.startswith('concreteness_'):
            if not trials.preload:
                trials.load_data()
            trials.crop(tmin=0.6, tmax=1.4)
            if trials.info['sfreq'] != 100: 
                trials.resample(100)
        X = trials.get_data()
        
    target_contrasts, conditions_to_merge = get_contrasts(config, analysis_name)
    events = merge_contrast_events(events, target_contrasts, conditions_to_merge)        
    y = events[:,2]
    
    if analysis_name == 'specificity':
        experiment = 'specificity'
    else:
        experiment = 'compose'
    mask_experiment = (metadata.experiment==experiment).values 
    metadata = metadata[metadata.experiment==experiment]
    X = np.array(list(compress(X, mask_experiment)))
    y = np.array(list(compress(y, mask_experiment)))
    assert len(X) == len(y)

    if analysis_name == 'denotation':
        mask_remove_baseline = [not t for t in (metadata.denotation=='baseline').values] # remove baseline (single word) trials
        X = X[mask_remove_baseline]
        y = [y[i] for i, include in enumerate(mask_remove_baseline) if include]
    elif analysis_name == 'denotation_cross_condition_test_on_subsective':
        mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
        mask_remove_subsective = [not t for t in mask_get_subsective] # then negate that, leaving subsective trials for testing
        X_train = X[mask_remove_subsective]
        y_train = [y[i] for i, include in enumerate(mask_remove_subsective) if include]
        X_test = X[mask_get_subsective]
        y_test = [y[i] for i, include in enumerate(mask_get_subsective) if include]
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name == 'denotation_cross_condition_test_on_privative':
        mask_get_privative = (metadata.denotation=='privative').values # get private trials
        mask_remove_privative = [not t for t in mask_get_privative] # then negate that, leaving privative trials for testing
        X_train = X[mask_remove_privative]
        y_train = [y[i] for i, include in enumerate(mask_remove_privative) if include]
        X_test = X[mask_get_privative]
        y_test = [y[i] for i, include in enumerate(mask_get_privative) if include]
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name.startswith('concreteness_trainWord'):
        mask_get_word = (metadata.denotation=='baseline').values # get single words only for training
        X_train = X[mask_get_word]
        y_train = [y[i] for i, include in enumerate(mask_get_word) if include]
        if analysis_name.endswith('testSub'):
            mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
            X_test = X[mask_get_subsective]
            y_test = [y[i] for i, include in enumerate(mask_get_subsective) if include]
        elif analysis_name.endswith('testPri'):
            mask_get_privative = (metadata.denotation=='privative').values # get private trials
            X_test = X[mask_get_privative]
            y_test = [y[i] for i, include in enumerate(mask_get_privative) if include]
        X = (X_train, X_test)
        y = (y_train, y_test)
    elif analysis_name.startswith('concreteness_general'):
        mask_get_word = (metadata.denotation=='baseline').values # get single words
        X_word = X[mask_get_word]
        y_word = y[mask_get_word]
        n_splits = 10
        cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
        # get train-test splits for subsective trials
        mask_get_subsective = (metadata.denotation=='subsective').values # get subsective trials
        X_sub = X[mask_get_subsective]
        y_sub = y[mask_get_subsective]
        X_sub_train, X_sub_test = [], []
        y_sub_train, y_sub_test = [], []
        for train_index, test_index in cv.split(X_sub, y_sub):
            X_sub_train_split, X_sub_test_split = X_sub[train_index], X_sub[test_index]
            X_sub_train.append(X_sub_train_split)
            X_sub_test.append(X_sub_test_split)
            y_sub_train_split, y_sub_test_split = y_sub[train_index], y_sub[test_index]
            y_sub_train.append(y_sub_train_split)
            y_sub_test.append(y_sub_test_split)
        # get train-test splits for privative trials
        mask_get_privative = (metadata.denotation=='privative').values # get private trials
        X_pri = X[mask_get_privative]
        y_pri = y[mask_get_privative]
        X_pri_train, X_pri_test = [], []
        y_pri_train, y_pri_test = [], []
        for train_index, test_index in cv.split(X_pri, y_pri):
            X_pri_train_split, X_pri_test_split = X_pri[train_index], X_pri[test_index]
            y_pri_train_split, y_pri_test_split = y_pri[train_index], y_pri[test_index]
            X_pri_train.append(X_pri_train_split)
            X_pri_test.append(X_pri_test_split)
            y_pri_train.append(y_pri_train_split)
            y_pri_test.append(y_pri_test_split)
        X_train, X_test = [], []
        y_train, y_test = [], []
        for i in range(n_splits):
            print('split', i+1)
            X_train.append(np.concatenate((X_word, X_sub_train[i], X_pri_train[i]), axis=0))
            print('len y_word', len(y_word))
            print(y_word)
            print('len y_sub_train', len(y_sub_train[i]))
            print(y_sub_train[i])
            print('len y_pri_train', len(y_pri_train[i]))
            print(y_pri_train[i])
            y_train.append(np.concatenate((y_word, y_sub_train[i], y_pri_train[i]), axis=0))

            if analysis_name.endswith('testSub'):
                X_test.append(X_sub_test[i])
                y_test.append(y_sub_test[i])
            if analysis_name.endswith('testPri'):
                X_test.append(X_pri_test[i])
                y_test.append(y_pri_test[i])
        print(y_test)
        X = (X_train, X_test)
        y = (y_train, y_test)
    return X, y


# =============================================================================
# Decoding functions
# =============================================================================

def choose_pipelines(classifier='', analysis=''):
    if classifier == 'logistic':
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=10000)
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
    cv = StratifiedKFold(shuffle=True, n_splits=10, random_state=42) 
    scores = cross_val_multiscore(time_decod, X, y, cv=cv) 
    time_decod.fit(X, y) # retrieve spatial patterns and spatial filters for interpretability
    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    return scores, coef

def decode_generalise(X, y, analysis='', classifier=''):
    clf = choose_pipelines(classifier=classifier)
    scorer = choose_scorer(analysis=analysis)
    time_gen = GeneralizingEstimator(clf, scoring=scorer, verbose=True)
    cv = StratifiedKFold(shuffle=True, n_splits=10, random_state=42) 
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

def plot_scores(times=None, scores=None, analysis='', subject=''):
    print('Plotting decoding scores.')
    fig, ax = plt.subplots()
    ax.plot(times, scores, label='score')
    ax.axhline(0.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC') 
    ax.legend()
    # ax.axvline(0.0, color='k', linestyle='-')
    ax.set_title(f'Decoding {analysis} in {subject}')
    plt.tight_layout()
    return fig, ax

def plot_patterns(coef=None, info=None, tmin=0., times=[]):
    print('Plotting spatial patterns.')
    evoked = mne.EvokedArray(coef, info, tmin=tmin)
    joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
    fig_evokeds = evoked.plot_joint(times=times, title='patterns', show=False, **joint_kwargs)
    return fig_evokeds