#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:59:13 2024

@author: rl05

Targeted analysis of ATL ROI activity: composition effect (phrase>word) 
200-250ms 
"""

import os
import os.path as op
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

from mne import read_source_spaces, read_source_estimate, read_labels_from_annot
from eelbrain._stats.stats import variability

import matplotlib.pyplot as plt
import seaborn as sns

import config 


# get group level data
subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

fsaverage_src_fname = op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_fsaverage = read_source_spaces(fsaverage_src_fname, verbose=False)

roi = 'anteriortemporal-lh'
annot = read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
label.subject = 'fsaverage_src'

poster_style = {
    'font.size': 20,       # Increase font size for titles and labels
    'axes.titlesize': 20,  # Increase title font size
    'axes.labelsize': 18,  # Increase label font size
    'figure.figsize': (10, 8),  # Set figure size (width, height)
    'axes.edgecolor': 'grey',  # Set axis color
    'axes.linewidth': 3,      # Increase axis linewidth
    'axes.grid': False,        # Show grid lines
    'grid.color': 'lightgray', # Set grid color
    'lines.linewidth': 3,     # Increase line thickness
    'xtick.labelsize': 18,    # Increase x-axis tick label size
    'ytick.labelsize': 18,    # Increase y-axis tick label size
    'xtick.color': 'grey',
    'ytick.color': 'grey',
    'xtick.major.size': 9,
    'ytick.major.size': 9,
    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'legend.fontsize': 14,    # Increase legend font size
    'legend.frameon': True,   # Display legend frame
    'legend.edgecolor': 'black'  # Legend frame color
}

#%% stats

data = pd.DataFrame(columns=['concreteness','composition','mne'])
# condition_list = []
concreteness_list = []
composition_list = []
roi_activity = []
subject_list = []
conditions = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
concreteness = ['concrete','abstract']
composition = ['baseline','subsective']

for c in concreteness:
    for d in composition:
        for subject in subjects:
            subject = f'sub-{subject}'
            print(f'Reading in source estimates from {subject}.')
            stc_path = op.join(data_dir, 'stcs', subject)
            stc = read_source_estimate(op.join(stc_path, f'{subject}_{c}-{d}_MEEG-lh.stc'), subject='fsaverage_src')
            stc = stc.crop(0.8, 0.85) # ATL composition effect
            stc = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0].mean() # average in space and time
            roi_activity.append(stc)
            concreteness_list.append(c)
            composition_list.append(d)
            subject_list.append(subject)
data['concreteness'] = concreteness_list
data['composition'] = composition_list
data['mne'] = roi_activity
data['subject'] = subject
# model = ols('mne ~ C(concreteness) + C(composition) + C(concreteness):C(composition)', data=data).fit()
model = smf.mixedlm("mne ~ C(concreteness) + C(composition) + C(concreteness):C(composition)", data, groups=data['subject'])
# result = sm.stats.anova_lm(model, typ=3)
result = model.fit()
# print(result)
print(result.summary())


results_fname = '/imaging/hauk/rl05/fake_diamond/results/neural/roi/anova/anteriortemporal-lh/replicateB&P/anova.txt'
print(result.summary(), file=open(results_fname, 'x'))

#%% plot 

# plt.figure(figsize=(5, 3))
# sns.boxplot(x='concreteness', y='mne', data=data, hue='composition')#, palette=palette)
# # plt.title('Boxplot of Score by Concreteness and Composition')
# plt.xlabel('Concreteness')
# plt.ylabel('Activity (MNE)')
# # plt.legend(title='Composition')
# plt.show()

g = sns.catplot(
    data=data, kind='bar',
    x='concreteness', y='mne', hue='composition',
    errorbar="se", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels('', 'Activity (MNE)')
# g.legend.set_title("")

# plot in two bar plots, dark vs. light for phrase-word, color for concreteness