# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:12:38 2023

@author: rl05
"""

import numpy as np
import csv
import random
from psychopy import visual, core, data, event

from scansync.meg import MEGTriggerBox

MEG = MEGTriggerBox()
MEG.set_trigger_state(0) # reset trigger state

number_of_blocks = 5
show_photodiode = True

expInfo = {}
expInfo['Subject'] = input('Subject: ')
expInfo['Run'] = input('Run (prac/expt): ')
if expInfo['Run'] == 'expt':
    expInfo['Block'] = input('Block (1-5): ')

# =============================================================================
# Psychopy display settings
# =============================================================================

font = 'Courier New'
anchorHoriz = 'center'

window = visual.Window(units='pix', color='black', fullscr=True)
word = visual.TextStim(window, text='', font=font, color='lightgrey', anchorHoriz=anchorHoriz, height=30)
probe = visual.TextStim(window, text='', font=font, color='cyan', anchorHoriz=anchorHoriz, height=30)
word_instruction = visual.TextStim(window, text='', font=font, color='lightgrey', anchorHoriz=anchorHoriz, wrapWidth=800, height=30)
fixation = visual.TextStim(window, text='+', font=font, color='lightgrey', height=50)
photodiode = visual.Rect(window, units='height', width=0.15, height=0.05, pos=(-0.85,-0.45), fillColor='grey')

window.mouseVisible = False


# =============================================================================
# Functions
# =============================================================================

def clear_screen():
    word.setText('')
    word.draw()
    window.flip()

def present_fix():
    for frame in range (18): # refresh rate 60Hz; on 300ms
        if frame <= 18:
            fixation.draw()
        window.flip()

def present_word(trigger=None, show_photodiode=show_photodiode):
    if trigger is not None:
        window.callOnFlip(MEG.set_trigger_state, value=trigger, return_to_zero_ms=20)
    for frame in range(36): # presents each word: on 300ms, off 300ms
        if frame < 18:
            word.draw()
            if show_photodiode:
                photodiode.draw()
        window.flip()

def present_probe(text=''):
    probe.setText(text) # presents memory probe
    probe.draw()
    window.flip()
    button_pressed, RT = MEG.wait_for_button_press(allowed=['Rb','Ry'], timeout=3)
    return button_pressed, RT

def present_feedback(text=''):
    for frame in range (48): # presents feedback: on 500ms, off 300ms
        if frame <= 30:
            word.draw()
        window.flip()

def present_instruction_participant(text=''):
    word_instruction.setText(text)
    word_instruction.draw()
    window.flip()
    MEG.wait_for_button_press(allowed=['Rb','Ry'])

def present_instruction_experimenter(text='', show_photodiode=show_photodiode):
    word_instruction.setText(text)
    word_instruction.draw()
    if show_photodiode:
        photodiode.draw()
    window.flip()
    event.waitKeys(keyList=['1'])

def calculate_answer(trial, response):
    answer = ''
    if trial['answer'] == 'yes':
        answer = 'Rb'
    elif trial['answer'] == 'no':
        answer = 'Ry'
    return answer
    
import time 
def present_trial(trial, run='', logfile=None):
    
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

    present_fix()
    for word_i in [1,2]:
        word.setText(trial[f'word{word_i}'].lower())
        present_word(trigger=int(trial[f'trigger_word{word_i}']))
    core.wait(0.2) # for capturing late composition effects
    if run == 'prac':
        if trial['probe'] != '':
            response, RT = present_probe(trial['probe'])
            answer = calculate_answer(trial, response)
            if response == answer:
                word.setText('Correct :)')
                present_feedback()
            else:
                word.setText('Incorrect :(')
                present_feedback()
            del answer
    elif run == 'expt':
        if trial['probe'] != '':
            response, RT = present_probe(trial['probe'])
            answer = calculate_answer(trial, response)
            logfile.addData('response', response)
            logfile.addData('RT', RT)
            if response == answer: 
                logfile.addData('hit', 1)
            elif response != answer:
                logfile.addData('hit', 0)
            elif response == 'q':
                core.quit()
            clear_screen()
            core.wait(random.uniform(1.3,1.7))
        else: 
            clear_screen()
            core.wait(random.uniform(0.8,1.2))

# =============================================================================
# Experiment
# =============================================================================

# practice session
if expInfo['Run'] == 'prac':

    stimuli_fname = 'stimuli_practice.csv'
    with open(stimuli_fname, 'r') as f:
        stimuli = [i for i in csv.DictReader(f)]
    random.Random(expInfo['Subject']).shuffle(stimuli) 
    # stimuli=stimuli[:2] # for debugging
    
    instruction = 'In this study, you will read words and phrases word-by-word at the screen centre and answer yes/no questions. \n\nPlease fixate on the cross and read the words and phrases carefully. Try and answer the questions as quickly and accurately as you can. Press the BLUE button for YES, and YELLOW for NO. \n\nWe will start with a practice session. You will be given feedback. Any questions?'
    present_instruction_experimenter(instruction)

    present_instruction_participant(f'This will be a practice block.\n\n<Click to begin>')
    clear_screen()
    core.wait(1.5)

    for j, trial in enumerate(stimuli): 
        present_trial(trial=trial, run=expInfo['Run'], logfile=None)

        # abort experiment 
        for keys in event.getKeys():
            if keys[0] in ['escape','q']:
                window.close()
                core.quit()

    clear_screen()
    core.wait(1.5)
    present_instruction_experimenter('Any questions?')
    window.close()
    
# experimental session
elif expInfo['Run'] == 'expt':

    # logfile: object created using psychopy
    logfile = data.ExperimentHandler(dataFileName='logs/logfile_subject{}_block{}'.format(expInfo['Subject'],expInfo['Block']), autoLog=False, savePickle=False)

    # stimuli fully randomised
    stimuli_fname = 'stimuli_test.csv'
    # stimuli_fname = 'stimuli_debug.csv'
    with open(stimuli_fname, 'r') as f:
        stimuli = [i for i in csv.DictReader(f)]
    random.Random(int(expInfo['Subject'])).shuffle(stimuli) # set subject number as random.Random instance
    blocks = np.array_split(stimuli, number_of_blocks) 

    block_nr = int(expInfo['Block'])
    block = blocks[block_nr-1]

    if block_nr == 1:
        instruction = 'Read the phrases carefully and answer yes/no questions as quickly and accurately as you can. Press the BLUE button for YES, and YELLOW for NO. \n\n This time, there will no feedback on your responses. There will be a break every 10 minutes or so.\n\n Any questions?'
        present_instruction_experimenter(instruction)

    present_instruction_experimenter('Getting ready...')
    present_instruction_participant(f'This will be block {str(block_nr)} of {len(blocks)}.\n\n<Click to begin>')
    clear_screen()
    core.wait(1.5)

    for j, trial in enumerate(block): 
        present_trial(trial=trial, run=expInfo['Run'], logfile=logfile)
        # save trial info to logfile
        logfile.addData('block_nr', block_nr)
        logfile.addData('trial_nr', j+1)        
        logfile.addData('set_nr', trial['set_nr'])
        logfile.addData('item_nr', trial['item_nr'])
        logfile.addData('word1', trial['word1'])
        logfile.addData('word2', trial['word2'])
        logfile.addData('probe', trial['probe'])
        logfile.nextEntry()

        # abort experiment 
        for keys in event.getKeys():
            if keys[0] in ['escape','q']:
                window.close()
                core.quit()

    clear_screen()
    core.wait(1.5)
    if block_nr == number_of_blocks:
        present_instruction_experimenter('The study is now over. \n\nPlease keep still while we finish the recording.')
    else:
        present_instruction_experimenter(f'This is end of block {str(block_nr)} of {len(blocks)}.\n\nOne moment please.')
    window.close()
# core.quit()
