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

plots = ['concreteness_xcond']#,'concreteness_xcond_general','concreteness_xcond_full']
windows = ['single']#,'sliding']
data_type = input('MEEG or ROI: ')
micro_ave = True
subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
classifier = 'logistic'

if data_type == 'ROI':
    rois = ['anteriortemporal-lh','posteriortemporal-lh','inferiorfrontal-lh','temporoparietal-lh']
else:
    rois = [None]  # placeholder for single MEEG pass

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
    
    for roi in rois:
        for window in windows:

            # make directory if doesn't exist
            figures_dir = op.join(config.project_repo, f'figures/decoding/{plot}/diagonal/logistic/{data_type}/{window}')
            if micro_ave:
                figures_dir = figures_dir + '/micro_ave'
            if not op.exists(figures_dir):
                os.makedirs(figures_dir, exist_ok=True)

            # read in decoding scores
            scores_group = [
                read_decoding_scores(subjects, analysis, classifier, data_type, window=window, roi=roi, micro_ave=micro_ave)
                for analysis in analyses
            ]
            # track this for slicing timewindows; also for fname
            sfreq = int(scores_group[0].shape[1] / 0.8)
            print('sfreq:',sfreq)

            # set up time windows
            if window == 'single':
                timewindow_early = dict(start=30, stop=50) if sfreq == 100 else None
                timewindow_late = dict(start=50, stop=80) if sfreq == 100 else None
            elif window == 'sliding':
                timewindow_early = dict(start=29, stop=48) if sfreq == 95 else None
                timewindow_late = dict(start=48, stop=76) if sfreq == 95 else None

            # prepare dataframe
            averaged_data = pd.DataFrame()
            list_timewindow, list_test_on, list_score, list_train_on = [], [], [], []

            if plot in ['concreteness_xcond', 'concreteness_xcond_general']:
                for i, test_on in enumerate(['subsective', 'privative']):
                    for timewindow_name, timewindow in zip(['early', 'late'], [timewindow_early, timewindow_late]):
                        scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                        list_score.extend(scores_timewindow)
                        list_timewindow.extend([timewindow_name] * len(scores_timewindow))
                        list_test_on.extend([test_on] * len(scores_timewindow))
                averaged_data = pd.DataFrame({
                    'timewindow': list_timewindow,
                    'test_on': list_test_on,
                    'score': list_score,
                    'subject': subjects * 4
                })

            elif plot == 'concreteness_xcond_full':
                for i, (test_on, train_on) in enumerate(zip(
                    ['subsective', 'privative', 'subsective', 'privative'],
                    ['subsective', 'subsective', 'privative', 'privative']
                )):
                    for timewindow_name, timewindow in zip(['early', 'late'], [timewindow_early, timewindow_late]):
                        scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                        list_score.extend(scores_timewindow)
                        list_timewindow.extend([timewindow_name] * len(scores_timewindow))
                        list_test_on.extend([test_on] * len(scores_timewindow))
                        list_train_on.extend([train_on] * len(scores_timewindow))
                averaged_data = pd.DataFrame({
                    'timewindow': list_timewindow,
                    'test_on': list_test_on,
                    'score': list_score,
                    'subject': subjects * 8,
                    'train_on': list_train_on
                })

            # Save CSV
            roi_label = roi if roi is not None else 'MEEG'
            csv_fname = op.join(figures_dir, f'scores_timewindow-averaged_{roi_label}_{sfreq}Hz.csv')
            print(f"Saving to {csv_fname}")
            averaged_data.to_csv(csv_fname, sep=',', index=False)
            
            # # get time windowed data and put into pandas df
            # if plot in ['concreteness_xcond','concreteness_xcond_general']:
            #     columns = ['timewindow','test_on','score','subject']
            # elif plot == 'concreteness_xcond_full':
            #     columns = ['timewindow','test_on','score','subject','train_on']
            # averaged_data = pd.DataFrame(columns=columns)
            # list_timewindow, list_test_on, list_score, list_train_on = [], [], [], []
            # if window == 'single':
            #     if sfreq == 100:
            #         timewindow_early = dict(start=30,stop=50)
            #         timewindow_late = dict(start=50,stop=80)
            # elif window == 'sliding':
            #     if sfreq == 95:
            #         timewindow_early = dict(start=29,stop=48)
            #         timewindow_late = dict(start=48,stop=76)

            # if plot in ['concreteness_xcond','concreteness_xcond_general']:
            #     for i, test_on in enumerate(['subsective','privative']):
            #         for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
            #             scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
            #             list_score.extend(scores_timewindow)
            #             list_timewindow.extend(np.repeat(timewindow_name, len(scores_timewindow)))
            #             list_test_on.extend(np.repeat(test_on, len(scores_timewindow)))
            #     averaged_data['timewindow'] = list_timewindow
            #     averaged_data['test_on'] = list_test_on
            #     averaged_data['score'] = list_score
            #     averaged_data['subject'] = subjects * 4
            # elif plot == 'concreteness_xcond_full':
            #     for i, (test_on, train_on) in enumerate(zip(['subsective','privative','subsective','privative'],['subsective','subsective','privative','privative'])):
            #         for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
            #             scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
            #             list_score.extend(scores_timewindow)
            #             list_timewindow.extend(np.repeat(timewindow_name, len(scores_timewindow)))
            #             list_test_on.extend(np.repeat(test_on, len(scores_timewindow)))
            #             list_train_on.extend(np.repeat(train_on, len(scores_timewindow)))
            #     averaged_data['timewindow'] = list_timewindow
            #     averaged_data['test_on'] = list_test_on
            #     averaged_data['score'] = list_score
            #     averaged_data['subject'] = subjects * 8
            #     averaged_data['train_on'] = list_train_on


            # csv_fname = op.join(figures_dir, f'scores_timewindow-averaged_{plot}_{window}_{}_{sfreq}Hz.csv')
            # print(csv_fname)
            # averaged_data.to_csv(csv_fname, sep=',')