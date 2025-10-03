import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
from plot_decoding import *
import fig_constants
import fig_helpers as fh
import config
# mpl.rc_file('fake_diamond.rc')


# 1. CONFIGURATION
classifier   = 'logistic'
window       = 'single'
micro_ave    = True
data_type    = 'ROI'
roi          = 'anteriortemporal-lh'
to_plot      = 'denotation+concreteness'
analyses     = ['denotation','concreteness']
subjects     = [f"sub-{s}" for s in config.subject_ids]
figures_dir = op.join(config.project_repo, f'figures/paper/')
os.makedirs(figures_dir, exist_ok=True)

# 2. LOAD DECODING SCORES
scores_diag = [read_decoding_scores(subjects, a, classifier, data_type, window, roi, timegen=False, micro_ave=micro_ave)
               for a in analyses]
scores_tgm  = [read_decoding_scores(subjects, a, classifier, data_type, window, roi, timegen=True,  micro_ave=micro_ave)
               for a in analyses]

sfreq = int(scores_diag[0].shape[1] / 0.8)

# 3. SET UP FIGURE 
mosaic = [
    ['A','B',],
    ['C','D',]]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH, 4.),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    gridspec_kw={
        'height_ratios': [1, 1], # Set the relative heights of the rows
        'width_ratios': [2, 1], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.001}
)

# 4. PANEL A: Time series for test‑on‑subsective
axis = ax_dict['A']
plot_scores(axis, scores_diag[0], analysis=analyses[0], chance=0.5)
good_clusters, cluster_pvals = permutation_tests(scores_diag[0], timegen=False, against_chance=True)
plot_clusters(good_clusters, scores=scores_diag[0], analysis=analyses[0], ax=ax_dict['A'], cluster_pvals=cluster_pvals)
axis.set_title('decoding adjective denotation')
axis.set_xlim([-0.2, 1.4])
axis.set_yticks([0.48, 0.50, 0.52, 0.54])
fh.add_time_window_annotation(axis, x=0.0, y_offset_pct=0.05, width=0.3, height_pct=0.1, label='adjective', color='black', fontsize=6, facecolor='lightgrey', alpha=0.75)
fh.add_time_window_annotation(axis, x=0.6, y_offset_pct=0.05, width=0.3, height_pct=0.1,label='noun', color='black', fontsize=6, facecolor='lightgrey', alpha=0.75)


# # 5. PANEL C: Time series for concreteness
axis = ax_dict['C']
plot_scores(axis, scores_diag[1], analysis=analyses[1], chance=0.5)
good_clusters, cluster_pvals = permutation_tests(scores_diag[1], timegen=False, against_chance=True)
plot_clusters(good_clusters, scores=scores_diag[1], analysis=analyses[1], ax=ax_dict['C'], cluster_pvals=cluster_pvals)
axis.set_xlim([-0.2, 1.4])
axis.set_yticks([0.48, 0.50, 0.52, 0.54])
axis.set_title('decoding noun concreteness')
fh.add_time_window_annotation(axis, x=0.0, y_offset_pct=0.05, width=0.3, height_pct=0.1,label='adjective', color='black', fontsize=6, facecolor='lightgrey', alpha=0.75)
fh.add_time_window_annotation(axis, x=0.6, y_offset_pct=0.05, width=0.3, height_pct=0.1,label='noun', color='black', fontsize=6, facecolor='lightgrey', alpha=0.75)

# # 6. PANEL B: TGM for denotation
group_avg = np.mean(scores_tgm[0], axis=0)
vmax = round(np.max(group_avg), 2)
imB = ax_dict['B'].imshow(group_avg, origin='lower',
                 extent=[0,1.0,0,1.0], vmin=0.5, vmax=vmax,
                 cmap=color_scheme[analyses[0]])
good_clusters, cluster_pv = permutation_tests(scores_tgm[0], timegen=True, against_chance=True, t_threshold=None)
extent = np.array([0., 1.0])
times = np.linspace(extent[0], extent[1], scores_tgm[0][0].shape[0])
X, Y = np.meshgrid(times, times)
if cluster_pv:
    cluster_stats = []
    for i, (cluster, pval) in enumerate(zip(good_clusters, cluster_pv)):
        # Plot contours
        if pval < 0.05:
            ax_dict['B'].contour(X, Y, cluster, colors=['black'], linewidths=1)
        else:
            ax_dict['B'].contour(X, Y, cluster, colors=['grey'], linewidths=1)
fh.locatable_axes(ax_dict['B'], [0.5,vmax], imB)
ax_dict['B'].plot([0,1],[0,1],'--',color='lightgrey',lw=1.5,zorder=0)
ax_dict['B'].set_xlabel('Test time (s)')
ax_dict['B'].set_ylabel('Train time (s)')

# # 7. PANEL D: TGM for concreteness
group_avg = np.mean(scores_tgm[1], axis=0)
imD = ax_dict['D'].imshow(group_avg, origin='lower',
                 extent=[0.6,1.4,0.6,1.4], vmin=0.5, vmax=vmax,
                 cmap=color_scheme[analyses[1]])
good_clusters, cluster_pv = permutation_tests(scores_tgm[1], timegen=True, against_chance=True, t_threshold=None)
extent = np.array([0.6, 1.4])
times = np.linspace(extent[0], extent[1], scores_tgm[1][0].shape[0])
X, Y = np.meshgrid(times, times)
if cluster_pv:
    cluster_stats = []
    for i, (cluster, pval) in enumerate(zip(good_clusters, cluster_pv)):
        # Plot contours
        if pval < 0.05:
            ax_dict['D'].contour(X, Y, cluster, colors=['black'], linewidths=1)
        else:
            ax_dict['D'].contour(X, Y, cluster, colors=['grey'], linewidths=1)
fh.locatable_axes(ax_dict['D'], [0.5,vmax], imD)
ax_dict['D'].plot([0.6,1.4],[0.6,1.4],'--',color='lightgrey',lw=1.5,zorder=0)
ax_dict['D'].set_xlabel('Test time (s)')
ax_dict['D'].set_ylabel('Train time (s)')



fh.label_panels_mosaic(fig, ax_dict, size = 14)



# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'fig5_denotation+concreteness.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
