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
               '11','12','13','14','15','17','18','19','20', # not analysing sub-16 because of bad eeg digitisation
               '21','22','23','24','25','26','27','28','29','30',
               '31','32','33','34','35','36']

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
        '27': ('meg23_539', '231218'),
        '28': ('meg24_012', '240117'),
        '29': ('meg24_013', '240118'),
        '30': ('meg24_022', '240123'),
        '31': ('meg24_025', '240124'),
        '32': ('meg24_029', '240125'),
        '33': ('meg24_032', '240126'),
        '34': ('meg24_039', '240130'),
        '35': ('meg24_042', '240201'),
        '36': ('meg24_064', '240221'),
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
        '26': ('CBU230910'),
        '27': ('CBU240056'),
        '28': ('CBU240032'),
        '29': ('CBU240013'),
        '30': ('CBU230898'),
        '31': ('CBU240031'),
        '32': ('CBU240057'),
        '33': ('CBU240037'),
        '34': ('CBU240024'),
        '35': ('CBU240059'),
        '36': ('CBU240082'),
    }

runs = 5


# =============================================================================
# Bad and flat channels
# =============================================================================

bad_chs = {
        '01': {'eeg': ['EEG001','EEG002','EEG005','EEG061','EEG064'],
               'meg': []},
        '02': {'eeg': ['EEG038'],
               'meg': []},
        '03': {'eeg': ['EEG002','EEG004','EEG008','EEG017','EEG061','EEG062','EEG063'],
               'meg': []},
        '04': {'eeg': ['EEG001','EEG002','EEG004','EEG007','EEG008','EEG016','EEG017','EEG023','EEG028','EEG050','EEG059','EEG061'],
               'meg': ['MEG0211']},
        '05': {'eeg': ['EEG039'],
               'meg': []},
        '06': {'eeg': ['EEG008','EEG009','EEG028','EEG029','EEG051'],
               'meg': []},
        '07': {'eeg': ['EEG002'],
               'meg': []},
        '08': {'eeg': ['EEG042','EEG045','EEG046','EEG061'],
               'meg': []},
        '09': {'eeg': ['EEG034','EEG051','EEG019','EEG002','EEG004','EEG005','EEG007','EEG008','EEG009','EEG010','EEG016','EEG017','EEG023','EEG025','EEG027','EEG047','EEG045'],
               'meg': ['MEG1411','MEG1413','MEG0921','MEG0922','MEG0911']},
        '10': {'eeg': ['EEG008','EEG034','EEG060'],
               'meg': []},
        '11': {'eeg': ['EEG002','EEG004','EEG005','EEG006','EEG007','EEG008','EEG034','EEG038','EEG055'],
               'meg': []},
        '12': {'eeg': ['EEG018','EEG029','EEG034','EEG045'],
               'meg': []},
        '13': {'eeg': ['EEG002','EEG004','EEG008','EEG009','EEG029','EEG039'],
               'meg': []},
        '14': {'eeg': ['EEG002','EEG004','EEG008','EEG058'],
               'meg': []},
        '15': {'eeg': ['EEG001','EEG002','EEG003','EEG004','EEG005','EEG007','EEG008','EEG010','EEG016','EEG040','EEG052','EEG062','EEG063','EEG064'],
               'meg': ['MEG2343','MEG1922','MEG2331','MEG2511']},
        '16': {'eeg': ['EEG008','EEG034','EEG044','EEG045','EEG050','EEG054','EEG055'],
               'meg': ['MEG0121','MEG0131','MEG0211','MEG0341']},
        '17': {'eeg': ['EEG008','EEG016'],
               'meg': ['MEG2011']},
        '18': {'eeg': ['EEG034', 'EEG055', 'EEG052'],
               'meg': []},
        '19': {'eeg': ['EEG002','EEG004','EEG008','EEG029','EEG032','EEG034','EEG035','EEG036','EEG063'],
               'meg': []},
        '20': {'eeg': ['EEG002','EEG008','EEG036'],
               'meg': ['MEG1411']},
        '21': {'eeg': ['EEG006','EEG007'],
               'meg': ['MEG1321','MEG1311','MEG1221']},
        '22': {'eeg': [],
               'meg': []},
        '23': {'eeg': ['EEG028'],
               'meg': []},
        '24': {'eeg': ['EEG017','EEG035','EEG047','EEG057'],
               'meg': []},
        '25': {'eeg': ['EEG002','EEG008'],
               'meg': []},
        '26': {'eeg': [],
               'meg': []},
        '27': {'eeg': ['EEG028','EEG056'],
               'meg': ['MEG1633','MEG1911']},
        '28': {'eeg': ['EEG006','EEG007','EEG047'],
               'meg': []},
        '29': {'eeg': ['EEG002','EEG004','EEG008','EEG015','EEG017','EEG026'],
               'meg': []},
        '30': {'eeg': ['EEG002','EEG004','EEG006','EEG007','EEG021','EEG035','EEG050','EEG056','EEG061','EEG062'],
               'meg': ['MEG0213','MEG0343','MEG0133','MEG0123','MEG0121','MEG0131','MEG0111','MEG0113','MEG0221','MEG2632','MEG1441']},
        '31': {'eeg': ['EEG002','EEG004','EEG005'],
               'meg': ['MEG1531','MEG1541','MEG1923','MEG1943']},
        '32': {'eeg': ['EEG015','EEG017','EEG026'],
               'meg': []},
        '33': {'eeg': ['EEG002','EEG008','EEG017','EEG029'],
               'meg': []},
        '34': {'eeg': ['EEG002','EEG004','EEG006','EEG008','EEG015','EEG017','EEG018','EEG026','EEG051'],
               'meg': ['MEG2122','MEG1932']},
        '35': {'eeg': ['EEG002','EEG005','EEG006','EEG015','EEG023','EEG026','EEG028','EEG035','EEG055','EEG056','EEG058','EEG062'],
               'meg': ['MEG2112','MEG2322']},
        '36': {'eeg': ['EEG030','EEG055','EEG057','EEG063','EEG064'],
               'meg': ['MEG2312','MEG2432']},
    }

# if autoreject did not reject any epochs, use this as a hard threshold
hardcoded_thresholds = dict(grad=3000e-13,
                            mag=3500e-15,
                            eeg=200e-6)

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



# =============================================================================
# Decoding config
# =============================================================================

## map event trigger to decoding classes
# triggers (concrete: 100; abstract: 200; 
#           subsective: 10; privative: 20; baseline: 30; 
#           low: 70; mid: 80; high: 90; 
#           word1: 1; word2: 2)
# they add up to make the trigger for each condition 
# (e.g., concrete_subsective_word1 is 100+10+1=111; abstract_baseline_word2 is 200+30+2=232)


# decoding word vs. non-word in word 1 position
mapping_lexical = {81: 1, 
                   111: 1,
                   121: 1,
                   211: 1,
                   221: 1,
                   71: 2,
                   91: 2,
                   131: 2,
                   231: 2
                   }