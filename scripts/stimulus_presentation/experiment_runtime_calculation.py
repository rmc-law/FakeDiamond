fixation_on  = 0.3

word1_on  = 0.3
word1_off = 0.3
word2_on  = 0.3
word2_off = 0.5 # capture late effects

ITI = 1.5 

probe   = 1.4 
p_probe = 0.1
p_noProbe = 1 - p_probe

trial_noProbe = fixation_on + word1_on + word1_off + word2_on + ITI
print(f'A trial without probe takes around {trial_noProbe} s.')

trial_probe   = fixation_on + word1_on + word1_off + word2_on + word2_off + probe + ITI
print(f'A trial with probe takes around {trial_probe} s.')

p_probe = 0.1
trial_total = 900
trial_total_probe = 900 * p_probe
trial_total_noProbe = 900 * (1 - p_probe)

runtime_probe = trial_probe * trial_total_probe
runtime_noProbe = trial_noProbe * trial_total_noProbe

runtime_total = runtime_probe + runtime_noProbe
print(f'For a total of {trial_total} trials, the total effective recording time is {runtime_total/60} minutes.')