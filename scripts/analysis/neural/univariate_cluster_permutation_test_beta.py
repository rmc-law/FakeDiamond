import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (
    f_mway_rm,
    f_threshold_mway_rm,
    spatio_temporal_cluster_test,
    summarize_clusters_stc,
)

print(__doc__)

def permutation_tests(analyses=[], frequency_bands=[], classifier='logistic', baseline=None, 
                      subjects=[], times=None, generalisation=False):
    if times is None:
        times = np.linspace(-0.2, 1.5, 171)
    clusters_bands = [[], [], [], []]
    for i_band, band in enumerate(frequency_bands):
        if generalisation is False:
            scores_conditions = np.zeros((len(subjects),len(times))) 
        else:
            scores_conditions = np.zeros((len(subjects),len(times),len(times))) 
        for i_subject, subject in enumerate(subjects):
            if generalisation is False:
                scores_analyses = np.zeros((len(analyses),len(times))) 
            else:
                scores_analyses = np.zeros((len(analyses),len(times),len(times))) 
            for i_analysis, analysis in enumerate(analyses): 
                if generalisation is False:
                    scores = np.load(os.path.join(OUTPUT_DIR, subject, f'decode_{analysis}_{band}_baseline{baseline}_{classifier}_sensor_scores_temporal.npy'))  
                else: 
                    scores = np.load(os.path.join(OUTPUT_DIR, subject, f'decode_{analysis}_{band}_baseline{baseline}_{classifier}_sensor_scores_generalisation.npy'))
                scores = scores.mean(0)
                scores_analyses[i_analysis,:] = scores
            if len(analyses) == 2:
                scores_diff = np.subtract(scores_analyses[0],scores_analyses[1])
            elif len(analyses) == 1:
                scores_diff = np.subtract(scores_analyses[0],np.full(scores_analyses[0].shape,0.5))
            scores_conditions[i_subject,:] = scores_diff
        if generalisation is False:
            t_obs, clusters, cluster_pv, H0 = \
                mne.stats.permutation_cluster_1samp_test(scores_conditions, threshold=None, n_permutations=10000, seed=42, out_type='mask', verbose=True)
        else:
            t_obs, clusters, cluster_pv, H0 = \
                mne.stats.spatio_temporal_cluster_1samp_test(scores_conditions, threshold=None, n_permutations=10000, seed=42, out_type='mask', verbose=True)        
        clusters_bands[i_band].append(np.array([t_obs, clusters, cluster_pv, H0], dtype='object'))
    return clusters_bands

'''...................... Launch IPython with eelbrain ......................'''

import numpy as np
import pandas as pd
import mne, eelbrain, os, pickle
# from surfer import Brain


'''..................... directory, date, output folder .....................'''

# set up current directory
root = '/Users/ryanlaw/Analyses/wrdlst_filter'
subjects_dir = os.path.join(root,'MRI')
os.chdir(root)

# date of analysis
date = '3June2019'

# output folder path
output = os.path.join(root, 'GRP/%s/temporal' %date)


'''.................... lists of subjects and conditions ....................'''

subjects = ['A0182','A0205','A0208','A0231','A0238','A0290','A0297','A0305','A0320','A0333','A0335','A0338','A0339','A0345','A0350','A0356','A0360','A0371','A0373']
conditions = ['high_sent','high_list','low_sent','low_list']
words = ['word5','word6','word7']


'''.................. get fsaverage vertices for morphing ...................'''

print('Reading in source space...')
subject_to = 'fsaverage'
fs = mne.read_source_spaces('MRI/fsaverage/bem/fsaverage-ico-4-src.fif')
vertices_to = [fs[0]['vertno'], fs[1]['vertno']]


'''......................... read in morphed stcs ...........................'''

stcs,subjectlist,conditionlist,wordlist = [],[],[],[]

for subject in subjects:
    print('Reading in source estimates from Subject %s...' %subject)
    for condition in conditions:
        for word in words:
            tmp = mne.read_source_estimate('STC/%s/%s_%s_%s_dSPM-lh.stc' %(subject,subject,condition,word),subject='fsaverage')
            stcs.append(tmp)
            subjectlist.append(subject)
            conditionlist.append(condition)
            wordlist.append(word)


'''................ create dataset with stcs, subjs, conds ..................'''

print ('Creating dataset...')
ds = Dataset()
asso = [str.split(i,'_')[0] for i in conditionlist]
comp = [str.split(i,'_')[1] for i in conditionlist]
word = wordlist
print ('Reading in stcs...')
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage', src='ico-4', subjects_dir=subjects_dir, parc='wrdlst_7May2019') # parcellating source space
ds['Subject'] = Factor(subjectlist,random=True)
ds['Condition'] = Factor(conditionlist)
ds['Association'] = Factor(asso)
ds['Composition'] = Factor(comp)
ds['Position'] = Factor(word)
src_reset = ds['stcs']


'''.............................. run ROI test ..............................'''

# language network regions of interest
regions = ['IFG-lh', 'vmPFC-lh', 'ATL+MTL-lh', 'PTL-lh', 'TPJ-lh']
# 'V1-lh'

permutation_start = 0.15
permutation_stop = 0.55

for region in regions:
    print('Resetting source space data...')
    ds['stcs'] = src_reset
    src_region = src_reset.sub(source=region) # subset language network region data
    ds['stcs'] = src_region # assign this back to the ds

    # perform temporal permutation test in a particular region
    print('Performing temporal permutation test in %s...' %region)
    res = testnd.anova(ds['stcs'].mean('source'), 'Association*Composition*Position*Subject', ds=ds, samples=10000, pmin=0.3, tstart=permutation_start, tstop=permutation_stop, mintime=0.02, match='Subject')

    pickle.dump(res, open(os.path.join(output, '%s.pickle' %region),'wb'))

    f = open(os.path.join(output, '%s_results_table.txt' %region), 'w')
    f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=20ms\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()

    pmin = 0.05
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_nb = i+1
            cluster = sign_clusters[i]['cluster']
            tstart = sign_clusters[i]['tstart']
            tstop = sign_clusters[i]['tstop']
            effect = sign_clusters[i]['effect']

            effect = effect.replace(' x ', '%') # Changing format of string for interaction effects.

            print('Plotting time series for %s' %region)
            timecourse = src_region.mean('source')
            activation = eelbrain.plot.UTSStat(timecourse, effect, ds=ds, error='sem', match='Subject', legend='lower left', xlabel='Time (ms)', ylabel='Activation (dSPM)', xlim=(0,0.6), title='Cluster %s: Effect of %s at %s' %(i+1, effect, region))
            activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
            # activation.add_vline(0, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3.6, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(4.2, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            activation._axes[0].set_xticks([0,tstart,tstop])

            activation.save(os.path.join(output, 'clus%s_%s_%s_(%s-%s).png' %(i+1, tstart, tstop, effect, region)), dpi=250)
            activation.close()

            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
            bar = plot.Barplot(ds['average_source_activation'], effect, ds=ds, title='Average activation at %s' %region, match='Subject', ylabel='Average source activation (dSPM)')
            bar.save(os.path.join(output, 'cluster%s_BarGraph_(%s-%s)_effect=%s.png'%(i+1, tstart, tstop, effect)))
            bar.close()
