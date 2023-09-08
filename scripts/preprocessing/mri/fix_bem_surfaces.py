#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:03:07 2023

@author: rl05
"""

import os
import os.path as op
import shutil
import mne

import config 

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subject = input('subject bem to be fixed: ')
subject = f'sub-{subject}'
problematic_surface = input('which surface to fix: ')
problematic_surface = f'{problematic_surface}.surf'
bem_dir = op.join(subjects_dir, subject, 'bem')

# Put the converted surfaces in a separate 'conv' folder
conv_dir = op.join(subjects_dir, subject, 'conv')
os.makedirs(conv_dir, exist_ok=True)

# read problematic surface
coords, faces = mne.read_surface(op.join(bem_dir, problematic_surface))

# write to obj to open in blender
mne.write_surface(op.join(conv_dir, 'inner_skull.obj'), coords, faces, overwrite=True)

# Also convert the outer skull surface.
coords, faces = mne.read_surface(op.join(bem_dir, 'outer_skull.surf'))
mne.write_surface(op.join(conv_dir,'outer_skull.obj'), coords, faces, overwrite=True)


###############################################################################

# once fixed:
    
coords, faces = mne.read_surface(op.join(conv_dir, 'inner_skull_fix.obj'))

# Backup the original surface
shutil.copy(op.join(bem_dir, 'inner_skull.surf'), op.join(bem_dir, 'inner_skull_orig.surf'))


# Overwrite the original surface with the fixed version
# In real study you should provide the correct metadata using ``volume_info=``
# This could be accomplished for example with:
_, _, vol_info = mne.read_surface(op.join(bem_dir, 'inner_skull_orig.surf'),
                                  read_metadata=True)
mne.write_surface(op.join(bem_dir, 'inner_skull.surf'), coords, faces,
                  volume_info=vol_info, overwrite=True)