import numpy as np
import csv
import random
from psychopy import visual, core, data, event

font = 'Courier New'
anchorHoriz = 'center'

window = visual.Window(units='pix', color='black', fullscr=True)
word = visual.TextStim(window, text='', font=font, anchorHoriz=anchorHoriz, height=30)
photodiode = visual.Rect(window, units='height', width=0.08, height=0.05, pos=(-0.85,-0.45), fillColor='white')

word.setText('standardised')
word.draw()
photodiode.draw()
window.flip()

event.waitKeys(keyList=['q'])

# window.close()
# core.quit()
