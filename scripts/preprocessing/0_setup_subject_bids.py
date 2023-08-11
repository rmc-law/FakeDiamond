#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:26:18 2023

@author: rl05

Set up MNE-BIDS format for each subject
"""

import os.path as op

from mne import set_log_level, find_events, pick_events
from mne.io import read_raw_fif
from mne_bids import (BIDSPath, write_raw_bids, print_dir_tree)
from mne_bids.stats import count_events

import config

set_log_level(verbose='WARNING')

bids_root = config.bids_root 
print('BIDS root:', bids_root)
print() 

subjects = config.subject_ids
print('Subjects:', subjects)
print()

print('Setting up MNE-BIDS format for each subject...')
for subject in subjects:
    if not op.isdir(op.join(bids_root, f'sub-{subject}')):
        print(f'sub-{subject}: MNE-BIDS directory does not exist. Creating one...')
        bids_path = BIDSPath(subject=subject, root=bids_root)
        meg_id = config.map_subjects_meg[subject][0]
        meg_date = config.map_subjects_meg[subject][1]
        cbu_data_path = op.join(config.cbu_repo_meg, meg_id, meg_date)
        for run in range(1,config.runs+1):
            bids_path = BIDSPath(
                subject=subject,
                session='01',
                task='semantic',
                run=str(run),
                root=bids_root
            )
            raw_fname = op.join(cbu_data_path, f'block{run}_raw.fif')
            raw = read_raw_fif(raw_fname, preload=False)
            events = find_events(raw, stim_channel='STI101', min_duration=0.005)
            events = pick_events(events, exclude=[4096, 8192, 512])
            write_raw_bids(
                raw=raw,
                bids_path=bids_path,
                events=events,
                event_id=config.event_id,
            )
        print('Done.')
        print(print_dir_tree(bids_path.directory))

        counts = count_events(bids_path)
        print(counts)
    else:
        print(f'sub-{subject}: MNE-BIDS files already exists.')
print()

print('MNE-BIDS setup for all subjects.')