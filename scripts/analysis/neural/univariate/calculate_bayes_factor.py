'''
Time‐Resolved Bayes Factor Analysis on Source‐Localized MEG Data
----------------------------------------------------------------

This script loops through participants and experimental conditions in a 2×2 design (A × B),
loads source‐localized MEG time series (as MNE SourceEstimate files), crops to a time window
(t = 0.6–1.4 s), extracts the label‐averaged time series within a given anatomical label,
and then computes Bayes Factors **at each time point** for the main effects of A and B.
Results (BF₁₀ and BF₀₁ for A and B across time) are saved to a single CSV.

'''

import sys
import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc_file('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/univariate/fake_diamond.rc')

# Add path to your preprocessing/config if needed
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config  # assumes this module defines `subject_ids`

# =========================================
# 1. USER-DEFINED VARIABLES / PATHS
# =========================================

# Base directory containing subject STC folders
data_dir = '/imaging/hauk/rl05/fake_diamond/data/stcs'

# Tell MNE where to find the MRI source space (for label reading/morphing)
os.environ['SUBJECTS_DIR'] = '/imaging/hauk/rl05/fake_diamond/data/mri'

analysis = input('Which analysis (composition or denotation): ')

# Output directory: where CSVs and plots will be written
output_dir = f'/imaging/hauk/rl05/fake_diamond/results/neural/bayes/{analysis}'
os.makedirs(output_dir, exist_ok=True)

# Participants: exclude '16' per existing workflow
subjects = [f'sub-{subj_id}'
            for subj_id in config.subject_ids
            if subj_id != '16']
print(f'Subjects (n={len(subjects)}): {subjects}')

if analysis == 'composition':
    # Conditions (2×2 design):
    #   Factor A: 'concrete' (A1) vs 'abstract' (A2)
    #   Factor B: 'baseline' (B1) vs 'subsective' (B2)
    conditions = [
        'concrete-baseline',
        'concrete-subsective',
        'abstract-baseline',
        'abstract-subsective'
    ]

    # Map condition → A‐level and condition → B‐level
    cond2A = {
        'concrete-baseline':   'A1',
        'concrete-subsective': 'A1',
        'abstract-baseline':    'A2',
        'abstract-subsective':  'A2',
    }
    cond2B = {
        'concrete-baseline':   'B1',
        'concrete-subsective': 'B2',
        'abstract-baseline':    'B1',
        'abstract-subsective':  'B2',
    }
elif analysis == 'denotation':
    # Conditions (2×2 design):
    #   Factor A: 'concrete' (A1) vs 'abstract' (A2)
    #   Factor B: 'baseline' (B1) vs 'subsective' (B2)
    conditions = [
        'concrete-subsective',
        'concrete-privative',
        'abstract-subsective',
        'abstract-privative'
    ]

    # Map condition → A‐level and condition → B‐level
    cond2A = {
        'concrete-subsective':   'A1',
        'concrete-privative': 'A1',
        'abstract-subsective':    'A2',
        'abstract-privative':  'A2',
    }
    cond2B = {
        'concrete-subsective':   'B1',
        'concrete-privative': 'B2',
        'abstract-subsective':    'B1',
        'abstract-privative':  'B2',
    }
# Time window of interest (in seconds)
tmin, tmax = 0.6, 1.4

# =========================================
# 2. FUNCTIONS
# =========================================

def compute_and_save_bf_for_hemi(hemi, cluster):
    '''
    For a given hemisphere ('lh' or 'rh'), load the ATL label, extract
    label‐averaged time series, compute BF₁₀/BF₀₁ for main effects A and B at each timepoint,
    save results to CSV, and return times & BF arrays.
    '''
    print(f'\n=== Processing hemisphere: {hemi} ===')

    avg_tmin, avg_tmax = cluster

    # --- 2a. Load the label for this hemisphere ---
    labels = mne.read_labels_from_annot(
        'fsaverage_src',
        parc='semantics',
        hemi=hemi
    )
    # Choose the first label (adjust index if you have multiple ATL labels)
    label = labels[0]
    print(f'Using label: name="{label.name}", hemi="{label.hemi}"')

    # --- 2b. Initialize and preallocate arrays ---

    n_subj = len(subjects)
    n_cond = len(conditions)

    # Read a template STC to determine n_times and time vector
    template_subj = subjects[0]
    template_cond = conditions[0]
    template_fname = f'{template_subj}_{template_cond}_MEEG-{hemi}.stc'
    template_path = os.path.join(data_dir, template_subj, template_fname)
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f'Template STC not found: {template_path}')

    stc_template = mne.read_source_estimate(template_path)
    stc_template.crop(tmin, tmax)
    n_times = stc_template.data.shape[1]
    times = stc_template.times.copy()

    print(f'Template STC loaded. n_times_in_window = {n_times}')

    # Preallocate array: [subjects × conditions × timepoints]
    data_ts = np.zeros((n_subj, n_cond, n_times))

    # Lookup dictionaries for indexing
    subj_index = {subj: idx for idx, subj in enumerate(subjects)}
    cond_index = {cond: idx for idx, cond in enumerate(conditions)}

    # --- 2c. Loop over subjects & conditions to fill data_ts ---
    for subj in subjects:
        s_idx = subj_index[subj]
        subj_dir = os.path.join(data_dir, subj)

        for cond in conditions:
            c_idx = cond_index[cond]
            fname = f'{subj}_{cond}_MEEG-{hemi}.stc'
            stc_path = os.path.join(subj_dir, fname)
            if not os.path.isfile(stc_path):
                raise FileNotFoundError(f'Missing STC file: {stc_path}')

            stc = mne.read_source_estimate(stc_path)
            stc.crop(tmin, tmax)
            stc_label = stc.in_label(label)
            if stc_label.data.size == 0:
                raise RuntimeError(
                    f'No data in label for {subj} / {cond} on hemi={hemi}.'
                )
            label_ts = stc_label.data.mean(axis=0)  # shape = (n_times,)
            data_ts[s_idx, c_idx, :] = label_ts

    print(f'Loaded all label‐averaged time series (shape = {data_ts.shape})')

    # --- 2d. Compute BF(t) for A and B ---

    BF10_A = np.zeros(n_times)
    BF01_A = np.zeros(n_times)
    BF10_B = np.zeros(n_times)
    BF01_B = np.zeros(n_times)

    A1_idx = [cond_index[c] for c in conditions if cond2A[c] == 'A1']
    A2_idx = [cond_index[c] for c in conditions if cond2A[c] == 'A2']
    B1_idx = [cond_index[c] for c in conditions if cond2B[c] == 'B1']
    B2_idx = [cond_index[c] for c in conditions if cond2B[c] == 'B2']

    if not (len(A1_idx) == len(A2_idx) == len(B1_idx) == len(B2_idx) == 2):
        raise RuntimeError('Expect exactly two conditions per level of A and B in a 2×2 design.')

    for ti in range(n_times):
        # Main effect of A
        arr_A1 = data_ts[:, A1_idx, ti].mean(axis=1)
        arr_A2 = data_ts[:, A2_idx, ti].mean(axis=1)
        t_val_A, _ = ttest_rel(arr_A1, arr_A2)
        bf10_a = pg.bayesfactor_ttest(t_val_A, n_subj, paired=True, r=0.5)
        BF10_A[ti] = bf10_a
        BF01_A[ti] = 1.0 / bf10_a

        # Main effect of B
        arr_B1 = data_ts[:, B1_idx, ti].mean(axis=1)
        arr_B2 = data_ts[:, B2_idx, ti].mean(axis=1)
        t_val_B, _ = ttest_rel(arr_B1, arr_B2)
        bf10_b = pg.bayesfactor_ttest(t_val_B, n_subj, paired=True, r=0.5)
        BF10_B[ti] = bf10_b
        BF01_B[ti] = 1.0 / bf10_b

    print('Completed Bayes Factor computation for hemi =', hemi)

    # --- 2e. Save BF time series to CSV ---
    df_bf = pd.DataFrame({
        'Time': times,
        'BF10_A': BF10_A,
        'BF01_A': BF01_A,
        'BF10_B': BF10_B,
        'BF01_B': BF01_B,
    })
    csv_fname = f'bayes_factor_time_series_{hemi}.csv'
    csv_path = os.path.join(output_dir, csv_fname)
    df_bf.to_csv(csv_path, index=False)
    print(f'Saved BF time series CSV to: {csv_path}')


    # --- 2f. Compute average BF values between avg_tmin and avg_tmax ---
    # Find the indices corresponding to that time window
    idx_window = np.where((times >= avg_tmin) & (times <= avg_tmax))[0]
    avg_BF10_A = BF10_A[idx_window].mean()
    avg_BF01_A = BF01_A[idx_window].mean()
    avg_BF10_B = BF10_B[idx_window].mean()
    avg_BF01_B = BF01_B[idx_window].mean()

    # Print the averages and common thresholds
    summary_lines = [
        f"Hemisphere: {hemi}",
        f"Time window for averaging: {avg_tmin:.3f} s to {avg_tmax:.3f} s",
        "",
        "Common Bayes Factor thresholds:",
        "  BF10 = 1/3 → Moderate evidence for H₀",
        "  BF10 = 1   → Inconclusive",
        "  BF10 = 3   → Moderate evidence for H₁",
        "  BF10 = 10  → Strong evidence for H₁",
        "",
        f"Average BF10_A over window: {avg_BF10_A:.3f}",
        f"Average BF01_A over window: {avg_BF01_A:.3f}",
        f"Average BF10_B over window: {avg_BF10_B:.3f}",
        f"Average BF01_B over window: {avg_BF01_B:.3f}",
    ]

    # Print to terminal
    for line in summary_lines:
        print(line)

    # Save summary to a text file
    txt_fname = f'bayes_factor_summary_{hemi}.txt'
    txt_path = os.path.join(output_dir, txt_fname)
    with open(txt_path, 'w') as f:
        for line in summary_lines:
            f.write(line + '\n')
    print(f"Saved BF summary to: {txt_path}")

    return times, BF10_A, BF01_A, BF10_B, BF01_B


def plot_bayes_factor_timeseries(
    times, BF10_A, BF01_A, BF10_B, BF01_B,
    clusters=None,
    title='Bayes Factor Time Series',
    save_path=None
):
    '''
    Plot BF10 and BF01 time series for main effects of A (purple) and B (black)

    Parameters:
    - times: numpy array, shape (n_times,)
    - BF10_A, BF01_A: numpy arrays, shape (n_times,)
    - BF10_B, BF01_B: numpy arrays, shape (n_times,)
    - clusters: list of (start_time, end_time) to shade significant intervals (optional)
    - title: figure title
    - save_path: if provided, save figure to this path
    '''

    plt.figure(figsize=(10, 5))

    # Base plot: thinner lines for all points
    plt.plot(times, BF10_A, color='purple', linewidth=1.0, label='BF10: Effect A')
    plt.plot(times, BF01_A, color='purple', linestyle='--', linewidth=1.0, label='BF01: Effect A')
    plt.plot(times, BF10_B, color='black', linewidth=1.0, label='BF10: Effect B')
    plt.plot(times, BF01_B, color='black', linestyle='--', linewidth=1.0, label='BF01: Effect B')

    # # Highlight moderate evidence (BF ≥ 3 or BF ≤ 1/3)
    # # For effect A:
    # mask_A = (BF10_A >= 3) | (BF01_A >= 3)
    # if mask_A.any():
    #     plt.plot(times[mask_A], BF10_A[mask_A], color='purple', linewidth=2.5)
    #     plt.plot(times[mask_A], BF01_A[mask_A], color='purple', linestyle='--', linewidth=2.5)

    # # For effect B:
    # mask_B = (BF10_B >= 3) | (BF01_B >= 3)
    # if mask_B.any():
    #     plt.plot(times[mask_B], BF10_B[mask_B], color='black', linewidth=2.5)
    #     plt.plot(times[mask_B], BF01_B[mask_B], color='black', linestyle='--', linewidth=2.5)

    # Plot interpretability thresholds
    plt.axhline(1, color='gray', linestyle=':', linewidth=1.0)
    plt.axhline(3, color='red', linestyle='--', linewidth=0.8)
    plt.axhline(1/3, color='red', linestyle='--', linewidth=0.8)

    # Shade permutation‐test‐identified clusters, if provided
    # if clusters is not None:
    #     for (t_start, t_end) in clusters:
    #         plt.axvspan(t_start, t_end, color='gray', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Bayes Factor (log scale)')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'Saved plot to {save_path}')
    else:
        plt.show()


# =========================================
# 3. MAIN SCRIPT
# =========================================

# Optionally, define your permutation‐test clusters (for both hemis).
# Example format: clusters_lh = [(0.72, 0.85), (1.02, 1.10)], clusters_rh = [...]
# If you don't have clusters yet, set clusters_lh = clusters_rh = None.

if analysis == 'composition':
    clusters_lh = (0.990,1.260)
    clusters_rh = (1.020,1.050)
elif analysis == 'denotation':
    clusters_lh = (0.980,1.040)

for hemi in ['lh', 'rh']:
    # Select the correct cluster list for this hemi
    if analysis == 'composition':
        clusters = clusters_lh if hemi == 'lh' else clusters_rh
    elif analysis == 'denotation':
        if hemi == 'lh':
            clusters = clusters_lh
        if hemi == 'rh':
            continue

    # Compute BF time series and save CSV
    times, BF10_A, BF01_A, BF10_B, BF01_B = compute_and_save_bf_for_hemi(hemi, clusters)

    # Plot and save figure
    plot_title = f'Bayes Factors Time Course ({hemi.upper()} ATL)'
    png_fname = f'bayes_factors_timecourse_{hemi}.png'
    png_path = os.path.join(output_dir, png_fname)

    plot_bayes_factor_timeseries(
        times=times,
        BF10_A=BF10_A,
        BF01_A=BF01_A,
        BF10_B=BF10_B,
        BF01_B=BF01_B,
        clusters=clusters,
        title=plot_title,
        save_path=png_path
    )

print('\n=== All done! ===')
