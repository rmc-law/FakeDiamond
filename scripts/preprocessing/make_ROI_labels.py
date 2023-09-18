#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:01:21 2023

@author: rl05
"""

import os
import os.path as op

from mne import read_labels_from_annot, read_label, write_labels_to_annot
from mne.viz import Brain

import config 

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')

subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir


# initialise target label list
labels_fake_diamond = []


# get visual areas for quality assurance
annot = 'PALS_B12_Brodmann'
label_visual = []
parc = read_labels_from_annot('fsaverage_src', 
                              annot, 
                              hemi='lh'
                              )
label_visual += [label for label in parc if 'Brodmann.17' in label.name]
label_visual += [label for label in parc if 'Brodmann.18' in label.name]
label_visual = [label_visual[0] + label_visual[1]]
label_visual[0].name = 'V1+V2-lh'
labels_fake_diamond += label_visual


# get lateral occipital ROI from aparc parcellation
annot = 'wrdlst2'
regions = ['TempPol',
           'TempMidLaSTS',
           'TempMidLpSTS',
           'LexTPJ',
           'FrontInfTri'
           ]

for region in regions:
    parcellation = read_labels_from_annot('fsaverage_src', 
                                          annot, 
                                          hemi='lh'
                                          )
    for label in parcellation:
        if region in label.name:
            labels_fake_diamond.append(label)

    
# writes labels to annot in fsavrage_src label folder
write_labels_to_annot(labels_fake_diamond, 
                      subject='fsaverage_src', 
                      parc='fake_diamond',
                      overwrite=True,
                      hemi='both'
                      )


# plot parcellation and labels
brain = Brain('fsaverage_src',
              hemi='lh',
              surf='partially_inflated',
              cortex='low_contrast',
              size=600,
              views='lateral',
              # views='medial',
              background='white'
              )
brain.add_annotation('fake_diamond')


# write rahimi's left and right ATLs labels to annot
labels = []
names = ['lh.ventral_ATL', 'rh.ventral_ATL']
for name in names:
    label = read_label(f'/imaging/hauk/rl05/fake_diamond/data/mri/fsaverage_src/label/{name}.label')
    labels.append(label)
write_labels_to_annot(labels, 
                      subject='fsaverage_src', 
                      parc='ventral_ATL',
                      overwrite=True,
                      hemi='both'
                      )

