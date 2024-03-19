#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:02:32 2024

@author: rl05
"""

# import os
import os.path as op
import sys
# import numpy as np
import pandas as pd

# import mne

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 

subject = 'sub-01'

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
stc_epoch_path = op.join(data_dir, 'stcs_epochs', subject)
epochs_log_fname = op.join(stc_epoch_path, 'epochs_matched_logfile.csv')

trial_info = pd.read_csv(epochs_log_fname)
