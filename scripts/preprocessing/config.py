#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:34:39 2023

@author: rl05
"""

import os
import os.path as op

project = 'fake_diamond'

cbu_repo_meg = op.join('/megdata/cbu', project)
project_repo = op.join('/imaging/hauk/rl05/', project)
bids_root = op.join(project_repo, 'data', 'raw')
# data_dir = op.join(project_repo, 'data', 'meg')
# subjects_dir = op.join(project_repo, 'data', 'mri')
logs_dir = op.join(project_repo, 'data', 'logs')

if not op.isdir(project_repo):  
    os.mkdir(project_repo)

# maxwell filtering files
calibration_fname = '/neuro_triux/databases/sss/sss_cal.dat'
crosstalk_fname = '/neuro_triux/databases/ctc/ct_sparse.fif'

# =============================================================================
# Mapping between subject and filename
# =============================================================================

subject_ids = ['01','02','03','04','05','06','07','08','09','10',
               '11','12','13','14','15','16','17','18','19','20',
               '21','22','23','24','25','26'
               ]

map_subjects_meg = {
        '01': ('meg23_285', '230807'),
        '02': ('meg23_293', '230810'),
        '03': ('meg23_297', '230814'),
        '04': ('meg23_306', '230816'),
        '05': ('meg23_308', '230817'),
        '06': ('meg23_312', '230821'),
        '07': ('meg23_318', '230823'),
        '08': ('meg23_325', '230830'),
        '09': ('meg23_331', '230904'),
        '10': ('meg23_332', '230905'),
        '11': ('meg23_336', '230906'),
        '12': ('meg23_340', '230907'),
        '13': ('meg23_342', '230908'),
        '14': ('meg23_345', '230911'),
        '15': ('meg23_351', '230912'),
        '16': ('meg23_353', '230913'),
        '17': ('meg23_357', '230914'),
        '18': ('meg23_458', '231107'),
        '19': ('meg23_462', '231109'),
        '20': ('meg23_491', '231123'),
        '21': ('meg23_505', '231130'),
        '22': ('meg23_511', '231205'),
        '23': ('meg23_518', '231208'),
        '24': ('meg23_526', '231212'),
        '25': ('meg23_534', '231215'),
        '26': ('meg23_539', '231218'),
        '27': ('meg23_', '23'),
        '28': ('meg23_', '23'),
        '29': ('meg23_', '23'),
    }

# subject names of MRI data
map_subjects_mri = {
        '01': ('CBU230541'),
        '02': ('CBU210688'),
        '03': ('CBU220705'),
        '04': ('CBU230500'),
        '05': ('CBU230539'),
        '06': ('CBU220202'),
        '07': ('CBU230656'),
        '08': ('CBU230164'),
        '09': ('CBU230623'),
        '10': ('CBU230627'),
        '11': ('CBU230647'),
        '12': ('CBU230625'),
        '13': ('CBU230641'),
        '14': ('CBU220323'),
        '15': ('CBU230614'),
        '16': ('CBU230645'),
        '17': ('CBU230560'),
        '18': ('CBU230578'),
        '19': ('CBU230897'),
        '20': ('CBU230870'),
        '21': ('CBU230871'),
        '22': ('CBU210576'),
        '23': ('CBU230886'),
        '24': ('CBU230888'),
        '25': ('CBU230587'),
        '26': ('CBU230910')
    }

runs = 5


# =============================================================================
# Bad and flat channels
# =============================================================================

bad_chs = {
        '01': {'eeg': ['EEG001','EEG002','EEG005','EEG061','EEG064'],
               'meg': []},
        '02': {'eeg': [],
               'meg': []},
        '03': {'eeg': ['EEG004','EEG008','EEG017','EEG061','EEG062','EEG063'],
               'meg': []},
        '04': {'eeg': ['EEG008','EEG016','EEG017'],
               'meg': []},
        '05': {'eeg': ['EEG039'],
               'meg': []},
        '06': {'eeg': ['EEG008','EEG009','EEG028','EEG029','EEG051'],
               'meg': []},
        '07': {'eeg': ['EEG002'],
               'meg': []},
        '08': {'eeg': ['EEG061'],
               'meg': []},
        '09': {'eeg': ['EEG004','EEG010','EEG025','EEG047','EEG051'],
               'meg': []},
        '10': {'eeg': [],
               'meg': []},
        '11': {'eeg': ['EEG038','EEG055'],
               'meg': []},
        '12': {'eeg': ['EEG018','EEG029','EEG034','EEG045'],
               'meg': []},
        '13': {'eeg': ['EEG002','EEG004','EEG008','EEG009','EEG029','EEG039'],
               'meg': []},
        '14': {'eeg': ['EEG002','EEG004','EEG008','EEG058'],
               'meg': []},
        '15': {'eeg': ['EEG001','EEG002','EEG003','EEG004','EEG005','EEG008','EEG010','EEG016','EEG040'],
               'meg': []},
        '16': {'eeg': ['EEG044','EEG055'],
               'meg': []},
        '17': {'eeg': ['EEG008'],
               'meg': []},
        '18': {'eeg': ['EEG034', 'EEG055', 'EEG052'],
               'meg': []},
        '19': {'eeg': ['EEG002','EEG004','EEG008','EEG029','EEG032','EEG034','EEG035','EEG036','EEG063'],
               'meg': []},
        '20': {'eeg': ['EEG002','EEG036'],
               'meg': []},
        '21': {'eeg': ['EEG006','EEG007'],
               'meg': []},
        '22': {'eeg': [],
               'meg': []},
        '23': {'eeg': ['EEG028'],
               'meg': []},
        '24': {'eeg': ['EEG035'],
               'meg': []},
        '25': {'eeg': ['EEG002','EEG008'],
               'meg': []},
        '26': {'eeg': [],
               'meg': []},
        '27': {'eeg': [],
               'meg': []},
        '28': {'eeg': [],
               'meg': []}
    }


manual_rejection_threshold = {
        '14': {'grad': 6e-11,
               'mag': 2.7e-12,
               'eeg': 90e-6},
        # '15': {'grad': ,
        #        'mag': ,
        #        'eeg': },
    }

# digital filtering parameters
l_freq = 0.1
h_freq = 40

# event ids
event_id_semantic = {
    'concrete/subsective/word1':111, 
    'concrete/subsective/word2':112,
    'concrete/privative/word1':121, 
    'concrete/privative/word2':122,
    'concrete/baseline/word1':131, 
    'concrete/baseline/word2':132,
    'abstract/subsective/word1':211, 
    'abstract/subsective/word2':212,
    'abstract/privative/word1':221, 
    'abstract/privative/word2':222,
    'abstract/baseline/word1':231, 
    'abstract/baseline/word2':232,
    'low/word1':71, 
    'low/word2':72,
    'mid/word1':81, 
    'mid/word2':82,
    'high/word1':91, 
    'high/word2':92,
    'BAD_ACQ_SKIP':999
}

event_id_semantic_word1 = {
    'concrete/subsective/word1':111, 
    'concrete/privative/word1':121, 
    'concrete/baseline/word1':131, 
    'abstract/subsective/word1':211, 
    'abstract/privative/word1':221, 
    'abstract/baseline/word1':231, 
    'low/word1':71, 
    'mid/word1':81, 
    'high/word1':91, 
}

event_id_misc = {
    'button/left':4096,
    'button/right':8192,
    'photodiode':512
}



# =============================================================================
# Inverse solution
# =============================================================================

# source space spacing
src_spacing = 'oct6'