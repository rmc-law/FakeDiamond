import sys
import os.path as op
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config
from plot_decoding import read_decoding_scores

# get privative trials
df_beh = pd.read_csv('/imaging/hauk/rl05/fake_diamond/scripts/analysis/behavioural/group_data.csv')
df_beh = df_beh[['participant','trial_nr','condition','denotation','concreteness','RT','hit']]
df_beh.dropna(inplace=True)
df_beh = df_beh[(df_beh['hit'] == 1) & (df_beh['RT'] != 0)]
df_beh['logRT'] = np.log(df_beh['RT'])
df_priv = df_beh[df_beh['denotation'] == 'privative']

df_privative_stats = df_priv.groupby('participant').agg({
    'logRT': 'mean',
    'hit': 'mean',  # proportion correct (accuracy)
    'trial_nr': 'count'  # number of trials
})
print(df_privative_stats)

# --- Configuration ---
classifier = 'logistic'
window = 'single'
micro_ave = True
data_type = 'ROI'
roi = 'anteriortemporal-lh'

# --- Load decoding scores for all subjects and analyses ---
analyses = ['concreteness_trainWord_testSub', 'concreteness_trainWord_testPri']
subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]

scores_group = [
    read_decoding_scores(subjects, analysis, classifier, data_type, window=window, roi=roi, micro_ave=micro_ave)
    for analysis in analyses
]

# --- Infer sampling frequency from time axis ---
sfreq = int(scores_group[0].shape[1] / 0.8)
print('sfreq: ', sfreq)

# --- Extract windowed scores into DataFrame ---
df_list = []
time_windows = {'early': slice(30, 50), 'late': slice(50, 80)}
if time_windows is None:
    raise ValueError(f"No time windows defined for sfreq={sfreq} and window='{window}'.")

iter_params = list(zip(analyses, ['subsective', 'privative']))

for i, (analysis_name, evaluation) in enumerate(iter_params):
    for timewindow_name, tw_slice in time_windows.items():
        scores_timewindow = scores_group[i][:, tw_slice].mean(axis=1)
        for subj_idx, score in enumerate(scores_timewindow):
            df_list.append({
                'score': score,
                'participant': subjects[subj_idx],
                'analysis': analysis_name,
                'evaluation': evaluation,
                'timewindow': timewindow_name
            })

df_decoding = pd.DataFrame(df_list)

# get only privative decoding scores
df_decoding['participant'] = df_decoding['participant'].str.extract(r'(\d+)').astype(int)
print(df_decoding)

# ========== Privative Analysis ==========
df_privative = df_decoding[df_decoding['analysis'] == 'concreteness_trainWord_testPri']

# Split early and late
early_priv = df_privative[df_privative['timewindow'] == 'early']
late_priv = df_privative[df_privative['timewindow'] == 'late']

# Merge with RT stats
early_priv_merged = early_priv.merge(df_privative_stats, left_on='participant', right_index=True)
late_priv_merged = late_priv.merge(df_privative_stats, left_on='participant', right_index=True)

# Correlate decoding score with logRT
r_early_priv, p_early_priv = pearsonr(early_priv_merged['score'], early_priv_merged['logRT'])
r_late_priv, p_late_priv = pearsonr(late_priv_merged['score'], late_priv_merged['logRT'])

print(f"[Privative] Early: r = {r_early_priv:.3f}, p = {p_early_priv:.3g}")
print(f"[Privative] Late:  r = {r_late_priv:.3f}, p = {p_late_priv:.3g}")

# ========== Subsective Analysis ==========

df_subsective = df_decoding[df_decoding['analysis'] == 'concreteness_trainWord_testSub']

# Split early and late
early_sub = df_subsective[df_subsective['timewindow'] == 'early']
late_sub = df_subsective[df_subsective['timewindow'] == 'late']

# Merge with RT stats
early_sub_merged = early_sub.merge(df_privative_stats, left_on='participant', right_index=True)
late_sub_merged = late_sub.merge(df_privative_stats, left_on='participant', right_index=True)

# Correlate decoding score with logRT
r_early_sub, p_early_sub = pearsonr(early_sub_merged['score'], early_sub_merged['logRT'])
r_late_sub, p_late_sub = pearsonr(late_sub_merged['score'], late_sub_merged['logRT'])

print(f"[Subsective] Early: r = {r_early_sub:.3f}, p = {p_early_sub:.3g}")
print(f"[Subsective] Late:  r = {r_late_sub:.3f}, p = {p_late_sub:.3g}")