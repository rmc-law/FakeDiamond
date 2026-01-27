import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from eelbrain import Dataset, load, Factor, concatenate, combine
from eelbrain._stats.stats import variability
from mne import read_source_estimate
import pickle

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
from config_plotting import *
import fig_constants
import fig_helpers as fh
import config


# 1. CONFIGURATION
analysis      = 'composition'
conditions   = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
relevant_conditions = ['baseline','subsective']
hemis        = ['lh','rh']
subjects     = [f"sub-{s}" for s in config.subject_ids]

subjects_dir = op.join(config.project_repo, 'data/mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
stc_path = op.join(config.project_repo, 'data/stcs')
results_dir = '/imaging/hauk/rl05/fake_diamond/results/neural/roi/anova/' # contains pickled permutation results
figures_dir  = op.join(config.project_repo, f'figures/paper/')
os.makedirs(figures_dir, exist_ok=True)
colors_tc = plt.cm.Greys([0.4, 0.7])
colors_bar = [color_scheme[condition] for condition in conditions]
times = np.linspace(0., 0.8, 200)

# 2. LOAD IN DATASET USING EELBRAIN
subjects_list, conditions_list, hemis_list, stcs = [], [], [], []
for subject in subjects:
    print(f'Reading in stc {subject}.')
    for condition in conditions:
        for hemi in hemis:
            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_MEEG-lh.stc')
            stc = read_source_estimate(stc_fname, subject='fsaverage_src')
            stc = stc.crop(tmin=0.6, tmax=1.4)
            stc.tmin = 0.
            stc = load.mne.stc_ndvar(stc, subject='fsaverage_src', src='oct-6', parc='semantics')
            stcs.append(stc)
            subjects_list.append(subject)
            conditions_list.append(condition)
            hemis_list.append(hemi)
            del stc


ds = Dataset()
concreteness = [condition.split('-')[0] for condition in conditions_list]
composition = [condition.split('-')[1] for condition in conditions_list]
# ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc='semantics') 
ds['stcs'] = concatenate(stcs, dim='case')
ds['subject'] = Factor(subjects_list, random=True)
ds['condition'] = Factor(conditions_list)
ds['condition'].sort_cells(conditions)
ds['composition'] = Factor(composition)
ds['composition'].sort_cells(['baseline','subsective'])
ds['concreteness'] = Factor(concreteness)
ds['concreteness'].sort_cells(['concrete','abstract'])
ds['hemi'] = Factor(hemis_list)
stc_reset = ds['stcs']

# now let's get results averaged across concreteness levels for the bilateral effect
# get hemi-specific data
ds_lh = ds.sub("hemi == 'lh'")
ds_rh = ds.sub("hemi == 'rh'")

# average within each ATL
stcs_lh = ds_lh['stcs'].sub(source='anteriortemporal-lh').mean('source')
stcs_rh = ds_rh['stcs'].sub(source='anteriortemporal-rh').mean('source')
time_courses_avg_clean = (stcs_lh + stcs_rh) / 2.0

ds_avg = Dataset()
ds_avg['subject'] = ds_lh['subject']
ds_avg['composition'] = ds_lh['composition']
ds_avg['concreteness'] = ds_lh['concreteness']
ds_avg['stcs'] = time_courses_avg_clean # Assign the correctly averaged data

print(f"Length of final averaged dataset: {len(ds_avg['stcs'])}")
# Expected Output: Length of final averaged dataset: 105

all_errors = variability(
    y=ds_avg['stcs'],  # The complete averaged NDVar
    x=ds_avg['composition'], # the complete composition factor
    match=ds_avg['subject'], # The complete 'subject' factor
    pool=False,
    spec='sem'
)

# 3. SET UP FIGURE 
mosaic = [
    ['A','A','A','A','B','B','B','B'],
    ['C','C','D','D','.','E','E','.']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH, 3.25),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    # sharey='row',
    gridspec_kw={
        'height_ratios': [1,1], # Set the relative heights of the rows
        'width_ratios': [1,1,1,1,1,1,1,1], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.005}
)

# 4. PANEL A: Time series for left ATL
axis = ax_dict['A']
for i, (relevant_condition, color, condition_name) in enumerate(zip(relevant_conditions,colors_tc,['word','phrase'])):
    data = ds_avg['stcs'][ds_avg[analysis].isin([relevant_condition])]
    data_group_avg = data.mean('case') # average over subjects
    # error = variability(y=data.x[:,:], x=ds_avg['composition'], match=ds_avg['subject'], pool=True, spec='sem')
    error_for_condition = all_errors[i]
    axis.plot(times, data_group_avg.x, color=color, lw=2.5, label=condition_name)
    axis.fill_between(times, data_group_avg.x-error_for_condition, data_group_avg.x+error_for_condition, alpha=0.2, color=color)
    axis.title.set_text('composition effects in\nbilateral anterior temporal lobes')
axis.set_xlim(0., 0.8)
axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
axis.set_xlabel('Time (s)')
axis.set_ylabel('Dipole moment (Am)')
imgA = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-lh_silver.png')
imagebox = OffsetImage(imgA, zoom=0.04)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(0.85, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)
imgA = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-rh_silver.png')
imagebox = OffsetImage(imgA, zoom=0.04)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(1.0, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)

leg = axis.legend(
    loc='upper left') 
leg.get_frame().set_facecolor('none')
leg.get_frame().set_edgecolor('none')
# for text, color in zip(leg.get_texts(), colors_tc):
#     text.set_color(color)



# read in permutation test pickle file
pickle_fname = op.join(results_dir, f'bilateralATL/composition*concreteness*hemisphere.pickle')
with open(pickle_fname, 'rb') as f:
    res = pickle.load(f)
pmin = 0.05
mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
sign_clusters = res.clusters[mask_sign_clusters] 

# since we know there are two clusters, I'll plot them intentionally
for i, axis in zip(range(sign_clusters.n_cases), [ax_dict['C'],ax_dict['D']]):
    cluster_tstart = sign_clusters[i]['tstart'] - 0.6
    cluster_tstop = sign_clusters[i]['tstop'] - 0.6
    cluster_effect = sign_clusters[i]['effect']
    cluster_pval = sign_clusters[i]['p']
    if cluster_effect != analysis:
        continue
    if cluster_pval < 0.05:
        alpha = 0.3
        color_cluster = 'yellow'
    else:
        alpha = 0.2
        color_cluster = 'grey'

    # add rectangles to demarcate clusters
    x_limits = (0.0, 0.8)
    span_coords = [(cluster_tstart, cluster_tstop)]
    alphas = [0.25]
    fh.add_background_spans(ax_dict['A'], span_coords, x_limits, alphas, color=color_cluster)

    # roi_activity = time_courses.mean(time=(cluster_tstart,cluster_tstop))

    # PANELS C & D: Bar plots for left ATL clusters
    x_positions = [1,1.5,2.25,2.75]
    j = 0
    sfreq = 250 
    start_index = int(round(cluster_tstart * sfreq))
    stop_index = int(round(cluster_tstop * sfreq))
    for k, c in enumerate(['concrete','abstract']):
        error_condition = all_errors[k]
        for d_label, d in zip(['word','phrase'],['baseline','subsective']):
            cond = ds_avg['stcs'][ds_avg['concreteness'].isin([c]) & ds_avg['composition'].isin([d])]
            error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
            cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
            cond_bar_err = error_condition[start_index:stop_index].mean()
            # cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
            bar = axis.bar(x_positions[j], cond_bar_mean, width=0.3, yerr=cond_bar_err, color=colors_bar[j], error_kw={'linewidth':1})
            fh.label_bars(bar, axis, d_label)
            j += 1 
        axis.set_xticklabels(['concrete', 'abstract'])
        axis.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
        axis.set_title(f'{int(round(cluster_tstart*1000,-1))}-{int(round(cluster_tstop*1000,-1))} ms')
        if i == 0: 
            axis.set_ylabel('Dipole moment\n(Am)')


# plot hemisphere effects
# get specificity-specific data
ds_lh = ds.sub("hemi == 'lh'")
ds_rh = ds.sub("hemi == 'rh'")

# get subject-specific condition data
stcs_lh_avg_list = []
stcs_rh_avg_list = []
subjects_unique = ds_lh['subject'].cells # Get a unique list of subjects

for subject in subjects_unique:
    # Average for the left hemisphere
    stcs_lh_subject = ds_lh.sub(f"subject == '{subject}'")['stcs']
    stcs_lh_avg_list.append(stcs_lh_subject.mean('case'))
    
    # Average for the right hemisphere
    stcs_rh_subject = ds_rh.sub(f"subject == '{subject}'")['stcs']
    stcs_rh_avg_list.append(stcs_rh_subject.mean('case'))

# Concatenate the lists of individual subject averages into two final NDVars
stcs_lh_avg = concatenate(stcs_lh_avg_list, dim='case').sub(source='inferiorfrontal-lh').mean('source')
stcs_rh_avg = concatenate(stcs_rh_avg_list, dim='case').sub(source='inferiorfrontal-rh').mean('source')

ds_avg = Dataset()
ds_avg['stcs'] = combine([stcs_lh_avg, stcs_rh_avg], name='stcs')
ds_avg['subject'] = Factor(list(subjects_unique), tile=2, random=True)
ds_avg['hemi'] = Factor(['lh','rh'], repeat=35)

all_errors = variability(
    y=ds_avg['stcs'],  # The complete averaged NDVar
    x=ds_avg['hemi'],             # The complete 'hemi' factor
    match=ds_avg['subject'],        # The complete 'subject' factor
    pool=False,
    spec='sem'
)

# 5. PANEL B: Time series for right ATL
axis = ax_dict['B']
colors_hemi = [plt.cm.Purples(0.3),plt.cm.Purples(0.5)]
for i, (relevant_condition, color) in enumerate(zip(['lh','rh'],colors_hemi)):
    data = ds_avg['stcs'][ds_avg['hemi'].isin([relevant_condition])]
    data_group_avg = data.mean('case') # average over subjects
    # error = variability(y=data.x[:,:], x=ds_avg['hemi'], match=ds_avg['subject'], pool=True, spec='sem')
    error_for_condition = all_errors[i]
    axis.plot(times, data_group_avg.x, color=color, lw=2.5)
    axis.fill_between(times, data_group_avg.x-error_for_condition, data_group_avg.x+error_for_condition, alpha=0.2, color=color)
    axis.title.set_text('hemisphere effect in\nbilateral anterior temporal lobes')
axis.set_xlim(0., 0.8)
axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
axis.set_xlabel('Time (s)')
axis.set_ylabel('')
imgA = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-lh_silver.png')
imagebox = OffsetImage(imgA, zoom=0.04)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(0.85, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)
imgA = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-rh_silver.png')
imagebox = OffsetImage(imgA, zoom=0.04)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(1.0, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)


# permutation test pickle file same as before
axis = ax_dict['B']
for i in range(sign_clusters.n_cases):
    cluster_tstart = sign_clusters[i]['tstart'] - 0.6
    cluster_tstop = sign_clusters[i]['tstop'] - 0.6
    cluster_effect = sign_clusters[i]['effect']
    cluster_pval = sign_clusters[i]['p']
    if cluster_effect != 'hemisphere':
        continue
    if cluster_pval < 0.05:
        alpha = 0.3
        color_cluster = 'yellow'
    else:
        alpha = 0.2
        color_cluster = 'grey'

    # add rectangles to demarcate clusters
    x_limits = (0.0, 0.8)
    span_coords = [(cluster_tstart, cluster_tstop)]
    alphas = [0.25]
    fh.add_background_spans(ax_dict['B'], span_coords, x_limits, alphas, color=color_cluster)

    # panel D -> bar plots for hemispehre effect
    x_positions = [1, 1.25] # Define positions for the bars

    # Loop through each condition to calculate its mean and error for the bar
    axis_bar = ax_dict['E']
    for j, condition in enumerate(hemis):
        # Select the bilaterally-averaged data for this condition
        cond_data = ds_avg['stcs'][ds_avg['hemi'].isin([condition])]
        
        # Calculate the mean activity within the cluster time window for the bar height
        cond_bar_mean = cond_data.mean(time=(cluster_tstart, cluster_tstop)).x.mean()
        
        # Calculate the average SEM within the cluster time window for the error bar
        # Convert time in seconds to array indices (assuming 250 Hz sampling rate)
        sfreq = 250 
        start_index = int(round(cluster_tstart * sfreq))
        stop_index = int(round(cluster_tstop * sfreq))
        
        # Slice the pre-calculated error array for the current condition and time window
        error_slice = all_errors[j, start_index:stop_index]
        cond_bar_err = error_slice.mean()

        # Plot the bar with its error bar
        axis_bar.bar(x_positions[j], cond_bar_mean, width=0.2, yerr=cond_bar_err, color=colors_hemi[j], error_kw={'linewidth': 1})

    # Set the x-axis tick labels to match the conditions
    axis_bar.set_xticklabels(['left','right'])
    axis_bar.set_xticks(x_positions)
    axis_bar.set_title(f'{int(round(cluster_tstart*1000,-1))}-{int(round(cluster_tstop*1000,-1))} ms')

    # PANELS C & D: Bar plots for left ATL clusters
    # x_positions = [1,1.5,2.25,2.75]
    
    # j = 0
    # for c in ['concrete','abstract']:
    #     for i, hemi in enumerate(hemis):
    #         cond = ds_avg['stcs'][ds_avg['hemi'].isin([d])]
    #         error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
    #         cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
    #         cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
    #         bar = axis.bar(x_positions[j], cond_bar_mean, width=0.3, yerr=cond_bar_err, color=colors_bar[j], error_kw={'elinewidth':1})
    #         # bar = axis.boxplot(x_positions[j], cond_bar_mean)
    #         fh.label_bars(bar, axis, d_label)
    #         j += 1
    #     axis.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
    #     axis.set_xticklabels(['concrete', 'abstract'])
    #     axis.set_title(f'{int(round(cluster_tstart*1000,-1))}-{int(round(cluster_tstop*1000,-1))} ms')


fh.label_panels_mosaic(fig, ax_dict, size = 14)

# plt.suptitle('Main effect of composition in anterior temporal lobe', fontweight='bold')

# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'fig3_composition_bilateral.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
