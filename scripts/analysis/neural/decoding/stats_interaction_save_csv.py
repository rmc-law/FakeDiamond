#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:21:26 2024

@author: rl05

Save cross condition decoding scores into a csv.
For later mixedlm analysis in R for interaction between time window and denotation.
"""


import sys
import os
import os.path as op
import numpy as np
import pandas as pd

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
from plot_decoding import *

# plots = ['concreteness_xcond','concreteness_xcond_general','concreteness_xcond_full']
plots = ['concreteness_xcond']
# windows = ['single','sliding']
windows = ['single']
subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')

for plot in plots:

    if plot == 'denotation+concreteness':
        analyses = ['denotation','concreteness']
    elif plot == 'composition':
        analyses = ['composition']
    elif plot == 'specificity':
        analyses = ['specificity']
    elif plot == 'concreteness_xcond':
        analyses = ['concreteness_trainWord_testSub','concreteness_trainWord_testPri']
    elif plot == 'concreteness_xcond_general':
        analyses = ['concreteness_general_testSub','concreteness_general_testPri']
    elif plot == 'concreteness_xcond_full':
        analyses = ['concreteness_trainSub_testSub','concreteness_trainSub_testPri',
                    'concreteness_trainPri_testSub','concreteness_trainPri_testPri']

    for window in windows:

        # make directory if doesn't exist
        figures_dir = op.join(config.project_repo, f'figures/decoding/{plot}/MEEG/{window}')
        if not op.exists(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)

        # read in decoding scores
        scores_group = []
        for analysis in analyses:
            scores = read_decoding_scores(subjects, analysis, 'logistic', 'MEEG', window=window, roi=None, timegen=False)
            scores_group.append(scores)

        # track this for slicing timewindows; also for fname
        sfreq = int(scores_group[0].shape[1] / 0.8)
        print('sfreq:',sfreq)

        # get time windowed data and put into pandas df
        if plot in ['concreteness_xcond','concreteness_xcond_general']:
            columns = ['timewindow','test_on','score','subject']
        elif plot == 'concreteness_xcond_full':
            columns = ['timewindow','test_on','score','subject','train_on']
        averaged_data = pd.DataFrame(columns=columns)
        list_timewindow, list_test_on, list_score, list_train_on = [], [], [], []
        if window == 'single':
            if sfreq == 100:
                timewindow_early = dict(start=30,stop=50)
                timewindow_late = dict(start=50,stop=80)
        elif window == 'sliding':
            if sfreq == 95:
                timewindow_early = dict(start=29,stop=48)
                timewindow_late = dict(start=48,stop=76)
            elif sfreq == 97:
                timewindow_early = dict(start=29,stop=49)
                timewindow_late = dict(start=49,stop=78)

        if plot in ['concreteness_xcond','concreteness_xcond_general']:
            for i, test_on in enumerate(['subsective','privative']):
                for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
                    scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                    list_score.extend(scores_timewindow)
                    list_timewindow.extend(np.repeat(timewindow_name, len(scores_timewindow)))
                    list_test_on.extend(np.repeat(test_on, len(scores_timewindow)))
            averaged_data['timewindow'] = list_timewindow
            averaged_data['test_on'] = list_test_on
            averaged_data['score'] = list_score
            averaged_data['subject'] = subjects * 4
        elif plot == 'concreteness_xcond_full':
            for i, (test_on, train_on) in enumerate(zip(['subsective','privative','subsective','privative'],['subsective','subsective','privative','privative'])):
                for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
                    scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                    list_score.extend(scores_timewindow)
                    list_timewindow.extend(np.repeat(timewindow_name, len(scores_timewindow)))
                    list_test_on.extend(np.repeat(test_on, len(scores_timewindow)))
                    list_train_on.extend(np.repeat(train_on, len(scores_timewindow)))
            averaged_data['timewindow'] = list_timewindow
            averaged_data['test_on'] = list_test_on
            averaged_data['score'] = list_score
            averaged_data['subject'] = subjects * 8
            averaged_data['train_on'] = list_train_on


        csv_fname = op.join(figures_dir, f'scores_timewindow-averaged_{plot}_{window}_{sfreq}Hz.csv')
        print(csv_fname)
        averaged_data.to_csv(csv_fname, sep=',')