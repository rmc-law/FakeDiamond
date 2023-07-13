# for f in [0.3]:

fixationOn  = 0.2
fixationOff = 0.0

wordOn  = 0.2
wordOff = 0.3

probe   = 1.4
probeProb = 0.1

# ITI_total = 1.5
ITI_1     = 0.5
ITI_2     = 1

trial_total = 900

# runtime = (((fixationOn + fixationOff + (wordOn + wordOff)*2 + ITI) * trial_total * (1-probeProb)) + 
# ((fixationOn + fixationOff + (wordOn + wordOff)*2 + probe + ITI) * trial_total * (probeProb)))
runtime = (((fixationOn + fixationOff + (wordOn + wordOff)*2 + ITI_1 + ITI_2) * trial_total * (1-probeProb)) + 
((fixationOn + fixationOff + (wordOn + wordOff)*2 + ITI_1 + probe + ITI_2) * trial_total * (probeProb)))

print(f'For a total of {trial_total} trials, the total effective recording time is {runtime/60} minutes.')
print(f'On average, ITI between trials without a probe is around {fixationOn+fixationOff+wordOff+ITI_1+ITI_2} s.')
print(f'On average, ITI between trials with a probe is around {fixationOn+fixationOff+wordOff+probe+ITI_1+ITI_2} s.')
print()