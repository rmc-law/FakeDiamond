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
from eelbrain import Dataset, load, Factor
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
classifier   = 'logistic'
window       = 'single'
micro_ave    = True
data_type    = 'ROI'
roi          = 'anteriortemporal-lh'
analysis      = 'composition'
conditions   = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
relevant_conditions = ['baseline','subsective']
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
subjects_list, conditions_list, stcs = [], [], []
for subject in subjects:
    print(f'Reading in stc {subject}.')
    for condition in conditions:
        stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_MEEG-lh.stc')
        stc = read_source_estimate(stc_fname, subject='fsaverage_src')
        stc = stc.crop(tmin=0.6, tmax=1.4)
        stc.tmin = 0.
        stcs.append(stc)
        subjects_list.append(subject)
        conditions_list.append(condition)
        del stc

ds = Dataset()
concreteness = [condition.split('-')[0] for condition in conditions_list]
composition = [condition.split('-')[1] for condition in conditions_list]
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc='semantics') 
ds['subject'] = Factor(subjects_list, random=True)
ds['condition'] = Factor(conditions_list)
ds['condition'].sort_cells(conditions)
ds['composition'] = Factor(composition)
ds['composition'].sort_cells(['baseline','subsective'])
ds['concreteness'] = Factor(concreteness)
ds['concreteness'].sort_cells(['concrete','abstract'])
stc_reset = ds['stcs']

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
roi = 'anteriortemporal-lh'
ds['stcs'] = stc_reset
stcs = ds['stcs']
stcs_region = stcs.sub(source = roi)
time_courses = stcs_region.mean('source')
ds['stcs'] = time_courses
axis = ax_dict['A']
for relevant_condition, color, condition_name in zip(relevant_conditions,colors_tc,['word','phrase']):
    data = ds['stcs'][ds[analysis].isin([relevant_condition])]
    data_group_avg = data.mean('case') # average over subjects
    error = variability(y=data.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
    axis.plot(times, data_group_avg.x, color=color, lw=2.5, label=condition_name)
    axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.2, color=color)
    axis.title.set_text('left anterior temporal lobe')
axis.set_xlim(0., 0.8)
axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
axis.set_xlabel('Time (s)')
axis.set_ylabel('Dipole moment (Am)')
imgA = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-lh_silver.png')
imagebox = OffsetImage(imgA, zoom=0.05)  # adjust zoom as needed
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
pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
with open(pickle_fname, 'rb') as f:
    res = pickle.load(f)
pmin = 0.1
mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
sign_clusters = res.clusters[mask_sign_clusters] 

# since we know there are two clusters, I'll plot them intentionally
for i, axis in zip(range(sign_clusters.n_cases), [ax_dict['C'],ax_dict['D']]):
    cluster_tstart = sign_clusters[i]['tstart'] - 0.6
    cluster_tstop = sign_clusters[i]['tstop'] - 0.6
    cluster_effect = sign_clusters[i]['effect']
    cluster_pval = sign_clusters[i]['p']
    if analysis == 'specificity_word':
        cluster_effect = 'specificity_word'
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
    for c in ['concrete','abstract']:
        for d_label, d in zip(['word','phrase'],['baseline','subsective']):
            cond = ds['stcs'][ds['concreteness'].isin([c]) & ds['composition'].isin([d])]
            error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
            cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
            cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
            bar = axis.bar(x_positions[j], cond_bar_mean, width=0.3, yerr=cond_bar_err, color=colors_bar[j], error_kw={'elinewidth':1})
            fh.label_bars(bar, axis, d_label)
            # subject_means = cond.mean(time=(cluster_tstart, cluster_tstop)).x
            # x_jittered = x_positions[j] + np.random.normal(0, 0.05, size=len(subject_means))
            # axis.scatter(
            #     # [x_positions[j]] * len(subject_means),
            #     x_jittered,
            #     subject_means,
            #     color=colors_bar[j],
            #     edgecolor='k',
            #     alpha=0.6,
            #     zorder=3,
            #     s=5
            # )
            j += 1
        axis.set_xticklabels(['concrete', 'abstract'])
        axis.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
        axis.set_title(f'{int(round(cluster_tstart*1000,-1))}-{int(round(cluster_tstop*1000,-1))} ms')
        if i == 0: 
            axis.set_ylabel('Dipole moment\n(Am)')

# 5. PANEL B: Time series for right ATL
roi = 'anteriortemporal-rh'
ds['stcs'] = stc_reset
stcs = ds['stcs']
stcs_region = stcs.sub(source = roi)
time_courses = stcs_region.mean('source')
ds['stcs'] = time_courses
axis = ax_dict['B']
for relevant_condition, color in zip(relevant_conditions,colors_tc):
    data = ds['stcs'][ds[analysis].isin([relevant_condition])]
    data_group_avg = data.mean('case') # average over subjects
    error = variability(y=data.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
    axis.plot(times, data_group_avg.x, color=color, lw=2.5)
    axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.2, color=color)
    axis.title.set_text('right anterior temporal lobe')
axis.set_xlim(0., 0.8)
axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
axis.set_xlabel('Time (s)')
axis.set_ylabel('')
imgB = mpimg.imread('/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_anteriortemporal-rh_silver.png')
imagebox = OffsetImage(imgB, zoom=0.05)  # adjust zoom as needed
ab = AnnotationBbox(
    imagebox,
    xy=(1.0, 1.0),             # upper-right in axis coordinates
    xycoords='axes fraction', # interpret xy as relative to axes
    box_alignment=(1, 1),     # align image top-right
    frameon=False             # no border around image
)
axis.add_artist(ab)


# read in permutation test pickle file
pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
with open(pickle_fname, 'rb') as f:
    res = pickle.load(f)
pmin = 0.1
mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
sign_clusters = res.clusters[mask_sign_clusters]

for i, axis in zip(range(sign_clusters.n_cases), [ax_dict['E']]):
    cluster_tstart = sign_clusters[i]['tstart'] - 0.6
    cluster_tstop = sign_clusters[i]['tstop'] - 0.6
    cluster_effect = sign_clusters[i]['effect']
    cluster_pval = sign_clusters[i]['p']
    if analysis == 'specificity_word':
        cluster_effect = 'specificity_word'
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
    fh.add_background_spans(ax_dict['B'], span_coords, x_limits, alphas, color=color_cluster)

        # PANELS C & D: Bar plots for left ATL clusters
    x_positions = [1,1.5,2.25,2.75]
    
    j = 0
    for c in ['concrete','abstract']:
        for d_label, d in zip(['word','phrase'],['baseline','subsective']):
            cond = ds['stcs'][ds['concreteness'].isin([c]) & ds['composition'].isin([d])]
            error = variability(y=cond.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
            cond_bar_mean = cond.mean(time=(cluster_tstart,cluster_tstop)).x.mean()
            cond_bar_err = error[int(cluster_tstart*250):int(cluster_tstop*250)].mean()
            bar = axis.bar(x_positions[j], cond_bar_mean, width=0.3, yerr=cond_bar_err, color=colors_bar[j], error_kw={'elinewidth':1})
            # bar = axis.boxplot(x_positions[j], cond_bar_mean)
            fh.label_bars(bar, axis, d_label)
            # subject_means = cond.mean(time=(cluster_tstart, cluster_tstop)).x
            # axis.boxplot(
            #     subject_means,
            #     positions=[x_positions[j]],
            #     widths=0.3,
            #     patch_artist=True,
            #     boxprops=dict(facecolor=colors_bar[j], color='black'),
            #     medianprops=dict(color='black'),
            #     whiskerprops=dict(color='black'),
            #     capprops=dict(color='black'),
            #     flierprops=dict(markerfacecolor='gray', markersize=4, linestyle='none')
            # )
            # x_jittered = x_positions[j] + np.random.normal(0, 0.05, size=len(subject_means))
            # axis.scatter(
            #     # [x_positions[j]] * len(subject_means),
            #     x_jittered,
            #     subject_means,
            #     color=colors_bar[j],
            #     edgecolor='k',
            #     alpha=0.6,
            #     zorder=3,
            #     s=5,

            # )
            j += 1
        axis.set_xticks([(x_positions[0]+x_positions[1])/2, (x_positions[2]+x_positions[3])/2])
        axis.set_xticklabels(['concrete', 'abstract'])
        axis.set_title(f'{int(round(cluster_tstart*1000,-1))}-{int(round(cluster_tstop*1000,-1))} ms')


fh.label_panels_mosaic(fig, ax_dict, size = 14)

# plt.suptitle('Main effect of composition in anterior temporal lobe', fontweight='bold')

# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'fig3_composition.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
