# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:12:38 2023

@author: rl05
"""

import numpy as np
import csv
import random
from psychopy import visual, core, data #event

from scansync.meg import MEGTriggerBox

MEG = MEGTriggerBox()
MEG.set_trigger_state(0) # reset trigger state

expInfo = {}
expInfo['Subject'] = input('subject: ')
expInfo['Run'] = input('prac/expt: ')

# =============================================================================
# Psychopy display settings
# =============================================================================

window = visual.Window(monitor="testMonitor", units="pix", fullscr=True)
word = visual.TextStim(window, text='', font='Courier New', wrapWidth=500, 
                       alignHoriz='center', height=30)
probe = visual.TextStim(window, text='', color='blue', font='Courier New', 
                        wrapWidth=500, alignHoriz='center', height=30)
word_instruction = visual.TextStim(window, text='', font='Courier New', 
                                   wrapWidth=600, alignHoriz='center', height=25)
fixation = visual.TextStim(window, text='+')
photodiode = visual.Rect(window, width=40, height=40, pos=[377,277], fillColor=[255,255,255], fillColorSpace='rgb255')
# pos (photodiode position) units: pixels; pos=[0,0] is screen centre

RT_clock = core.Clock()
window.mouseVisible = False


# =============================================================================
# Functions
# =============================================================================

def present_fix():
    for frame in range (36): # presents fix-cross: on 300ms, off 300ms
        if frame <= 18:
            fixation.draw()
        window.flip()

def present_word(trigger=None, photoDiode=True):
    if trigger is not None:
        # window.callOnFlip(triggerBox.activate_line, bitmask=int(trigger))
        window.callOnFlip(MEG.set_trigger_state(trigger, 20))
    for frame in range(36): # presents each word: on 300ms, off 300ms
        if frame < 18:
            word.draw()
            if photoDiode:
                photodiode.draw()
        window.flip()

def present_probe(text=''):
    RT_clock.reset()
    probe.setText(text) # presents memory probe
    probe.draw()
    window.flip()
    button_pressed, t = MEG.wait_for_button_press(allowed=['Ry','Rb'], timeout=None) # Ry = right yellow
    # return event.waitKeys(keyList=['1','2','q'], timeStamped=RT_clock)
    return button_pressed, t

def present_feedback(text=''):
    for frame in range (78): # presents feedback: on 1s, off 500ms
        if frame <= 60:
            word.draw()
        window.flip()

def present_instruction(text=''):
    word_instruction.setText(text)
    word_instruction.draw() # presents instructions
    window.flip()
    response, RT = MEG.wait_for_button_press(allowed=['Ry'], timeout=None) # Ry = right yellow
    return response, RT

def present_trial(stimulus, trial, run='', logfile=None):
    
    """Present an experimental trial
    
    Parameters
    ----------    
    stimuli : list 
        A csv file with rows of trial items
    trial : int 
        Trial index for selecting the trial from stimuli
    run : str
        Session run type, either 'prac' (practice) or 'expt' (experiment)
    logfile : object
        Log behavioural response (default is None)
    
    Returns
    -------
    logfile : logfile containing behavioural response. 
    """
    
    present_instruction('<Click to start trial>')
    present_fix()
    for word_i in [1,2]:
        word.setText(stimulus[trial]['word%s' %word_i])
        present_word(trigger=stimulus[trial]['trigger'])
    if run == 'prac':
        if stimulus[trial]['probe'] != '':
            response, _ = present_probe(stimulus[trial]['probe'])
            if response == stimulus[trial]['match']:
                word.setText('Correct!')
                present_feedback()
            else:
                word.setText('Incorrect!')
                present_feedback()
    elif run == 'expt':
        if trial['probe'] != '':
            response, RT = present_probe(trial['probe']) # presents probe
            logfile.addData('response', response)
            logfile.addData('RT', RT)
            if response == trial['response']: # record response
                logfile.addData('hit', 1)
            elif response != trial['response']:
                logfile.addData('hit', 0)
            elif response == 'q':
                core.quit()
        else:
            logfile.addData('hit', 'NaN')
    core.wait(random.uniform(0.2,0.7))
    pass 

# =============================================================================
# Experiment
# =============================================================================

# practice session
if expInfo['Run'] == 'prac':

    stimuli_fname = 'fakediamond_practice.csv'
    with open(stimuli_fname, 'rU') as f:
        stimuli = [i for i in csv.DictReader(f)]
    random.shuffle(stimuli) 

    instructions = ['Hi!\n\nThanks for taking part in our study!',
                    'In this study, you will be reading phrases.',
                    'They will be presented to you word-by-word at the centre of the screen.',
                    'Please read them carefully.',
                    'After some of the trials, you will be presented with a question.',
                    'Your task is to simply answer the question.',
                    'If your answer is YES, respond with your INDEX finger.\n\nIf your answer is NO, respond with your MIDDLE finger.',
                    'For practice, you will be given feedback on whether you got the task right.',
                    'We will start with a practice session!']
    for instruction in instructions:
        present_instruction(instruction)

    for trial in enumerate(stimuli):
        present_trial(stimuli, trial, run=expInfo['Run'], logfile=None)

    present_instruction('The practice session is now over. \n\n\nAny questions?')
    window.close()
    del stimuli, instructions
    
# experimental session
elif expInfo['Run'] == 'expt':

    # logfile: object created using psychopy
    logfile = data.ExperimentHandler(dataFileName='logs/{}_logfile'.format(expInfo['Subject']), autoLog=False, savePickle=False)

    # stimuli fully randomised
    stimuli_fname = 'fakediamond_stimuli.csv'
    with open(stimuli_fname, 'rU') as f:
        stimuli = [i for i in csv.DictReader(f)]
    random.shuffle(stimuli) 
    blocks = np.array_split(stimuli, 8) # split stimuli into blocks
    random.shuffle(blocks) 

    instructions = ['This is now the actual experiment!',
                    'Please read the phrases carefully, then answer some occasional questions.',
                    'If your answer is YES, respond with your INDEX finger.\n\nIf your answer is NO, respond with your MIDDLE finger.',
                    'This time, there will no feedback on your responses.',
                    'Every 2-3 minutes there will be a break.',
                    'Any questions?']
    for instruction in instructions:
        present_instruction(instruction)

    for i, block in enumerate(blocks):
        # introduce breaks between blocks
        if i > 0: 
            block_text = f'This will be block {str(i+1)} of {len(blocks)}. \n\nClick to begin.'
            print(block_text)
            present_instruction('\n\nFeel free to take a brief break now but try and keep your head still.'.join(block_text))

        for j, trial in enumerate(block): # Word-by-word presentation of trials
            present_trial(stimuli, trial, run=expInfo['Run'], logfile=logfile)

            # export all info to logfiles
            logfile.addData('ItemNum', trial['Item'])
            logfile.addData('TrialNum', j)
            logfile.addData('BlockNum', i)
            logfile.addData('Word1', trial['Word1'])
            logfile.addData('Word2', trial['Word2'])
            logfile.addData('Probe', trial['Probe'])
            logfile.nextEntry()

    present_instruction('The study is now over. \n\n\nPlease lie still while we finish the recording.')
    window.close()
    del stimuli, instructions
# core.quit()
