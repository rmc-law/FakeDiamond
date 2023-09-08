#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:15:20 2023

@author: rl05
"""

import sys
import os
import os.path as op

from mne import set_log_level
from mne.io import read_info
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
from mne.preprocessing import maxwell_filter, find_bad_channels_maxwell
from mne.viz import plot_head_positions
from mne_bids import read_raw_bids, find_matching_paths

import config
from helper import plot_noisy_channel_detection

set_log_level(verbose='WARNING')

subject = sys.argv[1]
run = sys.argv[2]

preprocessed_data_path = op.join(config.project_repo, 'data', 'preprocessed', f'sub-{subject}')
os.makedirs(preprocessed_data_path, exist_ok=True)
bads_dir = op.join(preprocessed_data_path, 'maxfiltered', f'run-{run}', 'bads')
os.makedirs(bads_dir, exist_ok=True)

# select raw fifs from bids
extensions = ['.fif']
bids_root = config.bids_root
bids_paths = find_matching_paths(bids_root, extensions=extensions, subjects=subject)
assert len(bids_paths) == 5 # make sure all runs are in bids format
bids_path = bids_paths[int(run)-1] 

if not op.exists(bids_path):
    raise FileNotFoundError(f"BIDS file not found: {bids_path}")

raw = read_raw_bids(bids_path)

# using run-3 as reference run for all runs to compensate for movement via maxwell-filter 
ref_run_info_fname = op.join(preprocessed_data_path, 'info', 'run3-info.fif') 
if not op.exists(ref_run_info_fname):
    raise FileNotFoundError(f"Run 3 info file not found: {ref_run_info_fname}")
else: 
    print('Reference run raw info exists. Reading it now.')
    raw_ref_info = read_info(ref_run_info_fname)
dev_head_t_ref = raw_ref_info['dev_head_t']

raw.load_data()

print('Detecting bad channels automatically.')
auto_bads, auto_flats, auto_scores = find_bad_channels_maxwell(
    raw,
    cross_talk=config.crosstalk_fname,
    calibration=config.calibration_fname,
    return_scores=True,
    verbose=True,
)
print(f'Bad channels: {auto_bads}')
print(f'Flat channels: {auto_flats}')

print('Saving automatically detected bad channels.')
for ch_type in ['mag','grad']:
    fig_auto_bads = plot_noisy_channel_detection(auto_scores, ch_type)
    fig_auto_bads.savefig(op.join(bads_dir, f'fig_run{run}_auto_bads_{ch_type}.png'))

# save lists of automatically detected bads and flats to file for each raw fif
with open(op.join(bads_dir, f'run-{1}_auto_bad_chs.txt'), 'w') as f:
    f.write('\n'.join(auto_bads))
with open(op.join(bads_dir, f'run-{1}_auto_flat_chs.txt'), 'w') as f:
    f.write('\n'.join(auto_flats))

raw.info["bads"] += auto_bads + auto_flats + config.bad_chs[subject]['meg']

print('Computing head position.')
chpi_amplitudes = compute_chpi_amplitudes(raw)
chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes)
head_pos = compute_head_pos(raw.info, chpi_locs, verbose=True)

print('Saving head position figures.')
for head_pos_mode in ['traces','field']:
    fig_head_pos = plot_head_positions(head_pos, mode=head_pos_mode, show=False)
    fig_head_pos.savefig(op.join(bads_dir, f'fig_run{run}_head_pos_{head_pos_mode}.png'))

# maxwell filter
raw_sss = maxwell_filter(raw,
                         origin='auto',
                         calibration=config.calibration_fname, 
                         cross_talk=config.crosstalk_fname,
                         st_duration=10,
                         coord_frame='head',
                         destination=dev_head_t_ref,
                         head_pos=head_pos, 
                         verbose=True
                         )

raw_sss_fname = op.join(preprocessed_data_path, 'maxfiltered', f'run-{run}', f'run{run}_sss_raw.fif')
raw_sss.save(raw_sss_fname, overwrite=True)

del raw, raw_sss
print(f'Maxfilter applied to sub-{subject} run-{run} raw fif.')
