#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Tue Sep 20 12:30:11 2022
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'templeton_exp2A_SD'  # from the Builder filename that created this script
expInfo = {'participant': '001', 'mapping': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s%s_%s_%s' % (expInfo['participant'], expName, expInfo['mapping'], 'practice', expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/haleyfrey/mountpoint_home/documents_and_experiment_code_Templeton2a/experiment_code/task-training_desc-noTrackerSmallerScannerImgs_d-09_m-19_y-22_var-main/templeton2A_SD_practice_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[2560, 1440], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "Intro"
IntroClock = core.Clock()
welcome = visual.TextStim(win=win, name='welcome',
    text='Welcome to the experiment!\n\n\nPress any key to continue.',
    font='Arial',
    units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
intro_continue = keyboard.Keyboard()

# Initialize components for Routine "instruction_1"
instruction_1Clock = core.Clock()
static_instruction = visual.TextStim(win=win, name='static_instruction',
    text='On each trial of the experiment you will see two groups of pictures, first one group and then the other a while later. \n\nYour task will involve reporting whether the pictures shown in the first group are the same as the pictures shown in the second group.',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1 = keyboard.Keyboard()
static_instruction_2 = visual.TextStim(win=win, name='static_instruction_2',
    text='Press any key to continue and  see an example of how these groups of pictures look.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "instruction_2"
instruction_2Clock = core.Clock()
static_instruction_3 = visual.TextStim(win=win, name='static_instruction_3',
    text='The pictures you’ll see will always be arranged in a circle like this. But the specific pictures shown may differ; this is only an example.',
    font='Arial',
    units='norm', pos=(0, 0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
static_key1_2 = keyboard.Keyboard()
static_instruction_4 = visual.TextStim(win=win, name='static_instruction_4',
    text='Please press any key for more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
array_image_1 = visual.ImageStim(
    win=win,
    name='array_image_1', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_2 = visual.ImageStim(
    win=win,
    name='array_image_2', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_3 = visual.ImageStim(
    win=win,
    name='array_image_3', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_4 = visual.ImageStim(
    win=win,
    name='array_image_4', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_5 = visual.ImageStim(
    win=win,
    name='array_image_5', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
array_image_6 = visual.ImageStim(
    win=win,
    name='array_image_6', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-9.0)
array_image_7 = visual.ImageStim(
    win=win,
    name='array_image_7', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-10.0)
array_image_8 = visual.ImageStim(
    win=win,
    name='array_image_8', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-11.0)
polygon_17 = visual.Polygon(
    win=win, name='polygon_17',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-12.0, interpolate=True)

# Initialize components for Routine "instruction_3"
instruction_3Clock = core.Clock()
static_instruction_15 = visual.TextStim(win=win, name='static_instruction_15',
    text='During each trial the first group of pictures will be shown only briefly, like the flash you just saw. After that no pictures will be on the screen for a while.',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
static_key1_6 = keyboard.Keyboard()
static_instruction_16 = visual.TextStim(win=win, name='static_instruction_16',
    text='Please press any key for more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
array_image_1b = visual.ImageStim(
    win=win,
    name='array_image_1b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_2b = visual.ImageStim(
    win=win,
    name='array_image_2b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_3b = visual.ImageStim(
    win=win,
    name='array_image_3b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_4b = visual.ImageStim(
    win=win,
    name='array_image_4b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_5b = visual.ImageStim(
    win=win,
    name='array_image_5b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
array_image_6b = visual.ImageStim(
    win=win,
    name='array_image_6b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-9.0)
array_image_7b = visual.ImageStim(
    win=win,
    name='array_image_7b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-10.0)
array_image_8b = visual.ImageStim(
    win=win,
    name='array_image_8b', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-11.0)
polygon_18 = visual.Polygon(
    win=win, name='polygon_18',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-12.0, interpolate=True)

# Initialize components for Routine "instruction_4"
instruction_4Clock = core.Clock()
static_instruction_17 = visual.TextStim(win=win, name='static_instruction_17',
    text='After that pause you will see the second group of pictures. This second group will stay on the screen for as long as you want, as shown here.',
    font='Arial',
    units='norm', pos=(0, 0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
static_key1_7 = keyboard.Keyboard()
static_instruction_18 = visual.TextStim(win=win, name='static_instruction_18',
    text='Please press any key for more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
array_image_1c = visual.ImageStim(
    win=win,
    name='array_image_1c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_2c = visual.ImageStim(
    win=win,
    name='array_image_2c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_3c = visual.ImageStim(
    win=win,
    name='array_image_3c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_4c = visual.ImageStim(
    win=win,
    name='array_image_4c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_5c = visual.ImageStim(
    win=win,
    name='array_image_5c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
array_image_6c = visual.ImageStim(
    win=win,
    name='array_image_6c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-9.0)
array_image_7c = visual.ImageStim(
    win=win,
    name='array_image_7c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-10.0)
array_image_8c = visual.ImageStim(
    win=win,
    name='array_image_8c', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-11.0)
polygon_19 = visual.Polygon(
    win=win, name='polygon_19',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-12.0, interpolate=True)

# Initialize components for Routine "instructions_4to5"
instructions_4to5Clock = core.Clock()
static_instruction_25 = visual.TextStim(win=win, name='static_instruction_25',
    text='Your task will always be the same:\n\nTo indicate whether, at one particular spot along the circle of pictures, the same picture was shown both times.\n\nIn other words, you will indicate whether the two groups of pictures both had the same picture in one particular location.',
    font='Arial',
    units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_11 = keyboard.Keyboard()
static_instruction_26 = visual.TextStim(win=win, name='static_instruction_26',
    text='Please press any key for more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "instruction_5"
instruction_5Clock = core.Clock()
static_instruction_19 = visual.TextStim(win=win, name='static_instruction_19',
    text='To let you know which spot along the circle you should be reporting about, there will be a red dotted line pointing to one of the eight spots, as shown here.\n\nSo in this example your task would be to identify whether the image to the right and slightly up stayed the same between the two groups of pictures or not, because the red dotted line is pointing to the right and slightly upward.',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_8 = keyboard.Keyboard()
static_instruction_20 = visual.TextStim(win=win, name='static_instruction_20',
    text='Please press any key for more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
polygon_20 = visual.Polygon(
    win=win, name='polygon_20',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
pointer_dot_0 = visual.Polygon(
    win=win, name='pointer_dot_0',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
pointer_dot_1 = visual.Polygon(
    win=win, name='pointer_dot_1',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-5.0, interpolate=True)
pointer_dot_2 = visual.Polygon(
    win=win, name='pointer_dot_2',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-6.0, interpolate=True)

# Initialize components for Routine "instruction_6"
instruction_6Clock = core.Clock()
static_instruction_27 = visual.TextStim(win=win, name='static_instruction_27',
    text='You will find that different trials will differ in details: the duration of the pause between the first and second group of pictures may vary, and the moment at which you see the red dotted line may vary, too.\n\nBut your task will always be the same: did the two groups of pictures both have the same picture at the location indicated by red dotted line, or not?',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_12 = keyboard.Keyboard()
static_instruction_28 = visual.TextStim(win=win, name='static_instruction_28',
    text='Please press any key to see more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
polygon_24 = visual.Polygon(
    win=win, name='polygon_24',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)

# Initialize components for Routine "instruction_67"
instruction_67Clock = core.Clock()
static_instruction_37 = visual.TextStim(win=win, name='static_instruction_37',
    text='default text',
    font='Arial',
    units='norm', pos=(0, 0.8), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_17 = keyboard.Keyboard()
static_instruction_38 = visual.TextStim(win=win, name='static_instruction_38',
    text='You will not be explicitly prompted when to give your response: you can simply press one of the two response keys as soon as you’ve seen both groups of pictures as well as the red dotted line indicating the location.\n\nPlease press any key to see more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.7), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
SD_image = visual.ImageStim(
    win=win,
    name='SD_image', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(0.4, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "instruction_67b"
instruction_67bClock = core.Clock()
static_instruction_39 = visual.TextStim(win=win, name='static_instruction_39',
    text='You will find that some trials are easy, yet that on other trials you have no idea and have to guess. That’s okay: immediately after each Same/Different response you get a chance to indicate how sure you were of that response.',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_18 = keyboard.Keyboard()
static_instruction_40 = visual.TextStim(win=win, name='static_instruction_40',
    text='Please press any key to see more instructions.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
polygon_37 = visual.Polygon(
    win=win, name='polygon_37',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)

# Initialize components for Routine "instruction_7"
instruction_7Clock = core.Clock()
static_instruction_31 = visual.TextStim(win=win, name='static_instruction_31',
    text='We will start with a few practice trials so that you can see for yourself.\n\nBefore starting the practice trials: here’s one final instruction. During the trials you will see a black dot at the screen center, just like the one you see there now. It is important to keep your eyes directed at that black dot during the trials. So no looking around while a trial is ongoing.',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
static_key1_14 = keyboard.Keyboard()
static_instruction_32 = visual.TextStim(win=win, name='static_instruction_32',
    text='Please press any key to begin the practice trials.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
polygon_29 = visual.Polygon(
    win=win, name='polygon_29',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)

# Initialize components for Routine "load_practice_trials"
load_practice_trialsClock = core.Clock()
practice_trial_load_number=0
practice_trial_image_list = []
retro_0_post_1_practice = []
mem_display_to_cue_times_practice = []
change_0_1_practice = []

# Initialize components for Routine "get_ready_practice"
get_ready_practiceClock = core.Clock()
static_instruction_29 = visual.TextStim(win=win, name='static_instruction_29',
    text='default text',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
static_key1_13 = keyboard.Keyboard()
static_instruction_30 = visual.TextStim(win=win, name='static_instruction_30',
    text='Please press any key to start.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
polygon_25 = visual.Polygon(
    win=win, name='polygon_25',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)

# Initialize components for Routine "practice_mem_display"
practice_mem_displayClock = core.Clock()
array_image_1d = visual.ImageStim(
    win=win,
    name='array_image_1d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-1.0)
array_image_2d = visual.ImageStim(
    win=win,
    name='array_image_2d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)
array_image_3d = visual.ImageStim(
    win=win,
    name='array_image_3d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
array_image_4d = visual.ImageStim(
    win=win,
    name='array_image_4d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_5d = visual.ImageStim(
    win=win,
    name='array_image_5d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_6d = visual.ImageStim(
    win=win,
    name='array_image_6d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_7d = visual.ImageStim(
    win=win,
    name='array_image_7d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_8d = visual.ImageStim(
    win=win,
    name='array_image_8d', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
polygon_26 = visual.Polygon(
    win=win, name='polygon_26',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-9.0, interpolate=True)

# Initialize components for Routine "retention_cue_and_test_intval_practice_2"
retention_cue_and_test_intval_practice_2Clock = core.Clock()
polygon_28 = visual.Polygon(
    win=win, name='polygon_28',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
array_image_1f = visual.ImageStim(
    win=win,
    name='array_image_1f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)
array_image_2f = visual.ImageStim(
    win=win,
    name='array_image_2f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
array_image_3f = visual.ImageStim(
    win=win,
    name='array_image_3f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_4f = visual.ImageStim(
    win=win,
    name='array_image_4f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_5f = visual.ImageStim(
    win=win,
    name='array_image_5f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_6f = visual.ImageStim(
    win=win,
    name='array_image_6f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_7f = visual.ImageStim(
    win=win,
    name='array_image_7f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
array_image_8f = visual.ImageStim(
    win=win,
    name='array_image_8f', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-9.0)
pointer_dot_0b = visual.Polygon(
    win=win, name='pointer_dot_0b',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-10.0, interpolate=True)
pointer_dot_1b = visual.Polygon(
    win=win, name='pointer_dot_1b',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-11.0, interpolate=True)
pointer_dot_2b = visual.Polygon(
    win=win, name='pointer_dot_2b',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-12.0, interpolate=True)
sd_practice_resp = keyboard.Keyboard()

# Initialize components for Routine "confidence_rating_practice"
confidence_rating_practiceClock = core.Clock()
text_10 = visual.TextStim(win=win, name='text_10',
    text='default text',
    font='Arial',
    units='norm', pos=(0, .65), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
conf_practice_resp = keyboard.Keyboard()
image = visual.ImageStim(
    win=win,
    name='image', 
    image='LH_buttonbox_practice.JPG', mask=None,
    ori=0, pos=(0, -0.1), size=(0.35, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)

# Initialize components for Routine "instructions_specific_training"
instructions_specific_trainingClock = core.Clock()
static_instruction_42 = visual.TextStim(win=win, name='static_instruction_42',
    text='default text',
    font='Arial',
    units='norm', pos=(0, 0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
polygon_36 = visual.Polygon(
    win=win, name='polygon_36',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
static_instruction_43 = visual.TextStim(win=win, name='static_instruction_43',
    text='Please press any key to continue.',
    font='Arial',
    units='norm', pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
static_key1_20 = keyboard.Keyboard()
text_with_or_without_eye_tracking="Good. Now that you know the basic idea, let's get better at the task. Here's a tip that helps people do well:\n\nAs soon as you see the red dotted line pointing to some spot, immediately try to recall or verbalize what occupied that spot in the first group of pictures. You’ll find that, the faster you try to recall it, the better you do.\n\nThis also means that, if the red dotted line appears earlier than the second group of pictures, don't wait for that second group to appear! Instead, try to recall what was at the indicated spot right away, before the second group of pictures appears. People who follow that strategy tend to be our best performers."

if False:
    text_with_or_without_eye_tracking=text_with_or_without_eye_tracking+"\n\nBefore we start the experiment, we will do a short calibration for the eye-tracking device. Please follow the black dots with your eyes."
    
static_instruction_42.setText(text_with_or_without_eye_tracking,False)

# Initialize components for Routine "load_training_trials"
load_training_trialsClock = core.Clock()
training_trial_load_number=0
training_trial_image_list = []
retro_0_post_1_training = []
change_0_1_training = []
consecutive_correct_retro=0
correct_retro_long_this_block=0
retro_trials_long_this_block=0
training_trial_index=0
training_trial_index_long_this_block=0

# Initialize components for Routine "get_ready_training_blockstart"
get_ready_training_blockstartClock = core.Clock()
static_instruction_44 = visual.TextStim(win=win, name='static_instruction_44',
    text='default text',
    font='Arial',
    units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
static_key1_21 = keyboard.Keyboard()

# Initialize components for Routine "training_mem_display"
training_mem_displayClock = core.Clock()
array_image_1l = visual.ImageStim(
    win=win,
    name='array_image_1l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-1.0)
array_image_2l = visual.ImageStim(
    win=win,
    name='array_image_2l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)
array_image_3l = visual.ImageStim(
    win=win,
    name='array_image_3l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
array_image_4l = visual.ImageStim(
    win=win,
    name='array_image_4l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_5l = visual.ImageStim(
    win=win,
    name='array_image_5l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_6l = visual.ImageStim(
    win=win,
    name='array_image_6l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_7l = visual.ImageStim(
    win=win,
    name='array_image_7l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_8l = visual.ImageStim(
    win=win,
    name='array_image_8l', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
polygon_44 = visual.Polygon(
    win=win, name='polygon_44',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-9.0, interpolate=True)

# Initialize components for Routine "retention_cue_and_test_intval_training"
retention_cue_and_test_intval_trainingClock = core.Clock()
polygon_46 = visual.Polygon(
    win=win, name='polygon_46',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[-1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
array_image_1m = visual.ImageStim(
    win=win,
    name='array_image_1m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-2.0)
array_image_2m = visual.ImageStim(
    win=win,
    name='array_image_2m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-3.0)
array_image_3m = visual.ImageStim(
    win=win,
    name='array_image_3m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-4.0)
array_image_4m = visual.ImageStim(
    win=win,
    name='array_image_4m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-5.0)
array_image_5m = visual.ImageStim(
    win=win,
    name='array_image_5m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-6.0)
array_image_6m = visual.ImageStim(
    win=win,
    name='array_image_6m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-7.0)
array_image_7m = visual.ImageStim(
    win=win,
    name='array_image_7m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-8.0)
array_image_8m = visual.ImageStim(
    win=win,
    name='array_image_8m', units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=512, interpolate=True, depth=-9.0)
pointer_dot_0d = visual.Polygon(
    win=win, name='pointer_dot_0d',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-10.0, interpolate=True)
pointer_dot_1d = visual.Polygon(
    win=win, name='pointer_dot_1d',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-11.0, interpolate=True)
pointer_dot_2d = visual.Polygon(
    win=win, name='pointer_dot_2d',units='pix', 
    edges=20, size=[1.0, 1.0],
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,-1,-1], lineColorSpace='rgb',
    fillColor=[1,-1,-1], fillColorSpace='rgb',
    opacity=1, depth=-12.0, interpolate=True)
reminder_training = visual.TextStim(win=win, name='reminder_training',
    text='Don’t forget to press S or D for ‘Same’ or ‘Different’. You’ve seen both groups of pictures, as well as the red dotted line indicating the location. If you don’t know the answer just guess; that’s OK.',
    font='Arial',
    pos=(0, 0.75), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-13.0);
sd_training_resp = keyboard.Keyboard()

# Initialize components for Routine "confidence_rating_training"
confidence_rating_trainingClock = core.Clock()
text_15 = visual.TextStim(win=win, name='text_15',
    text='default text',
    font='Arial',
    units='norm', pos=(0, .65), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
conf_training_resp = keyboard.Keyboard()

# Initialize components for Routine "short_pause"
short_pauseClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "End"
EndClock = core.Clock()
finish = visual.TextStim(win=win, name='finish',
    text='You’re done with the task!\n\nThank you for your participation.',
    font='Arial',
    units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "Intro"-------
continueRoutine = True
# update component parameters for each repeat
from numpy.random import random, shuffle
from numpy import tan, mod
from psychopy import core

timer  = core.Clock()
event.Mouse(visible=False)

x_scale = 27.83        #how many pixels per cm horizontally?
y_scale = 27.83        #how many pixels per cm vertically?
dist_cm = 57        #what is the screen distance in cm?

trials_per_block = 50
num_real_blocks = 10
num_real_trials = trials_per_block*num_real_blocks
real_trial_index = 0

blank_trial_start_dur_s=0.8
mem_display_duration_s=0.5

#----practice related-----
mem_display_to_cue_time_practice_s=1.6           #NB. test display can precede cue.
#-------------------------

cue_duration_s=0.5
retention_duration_s=0.8                #blank period following cue in retro-cue trials, before test display appears

#----training related-----
max_num_training_trials=300
consecutive_retro_correct_required=8  #8 #we're starting at the shortest delay (i.e. mem_display_to_cue_time_initial_training_s). Once this number of correct retrocue trials has been reached, we do some more training at a longer ones
retro_prop_correct_required=0.8       #at the longer retention intervals, this is the proportion correct retro cues that the participant needs to reach to finish the practice early
repeat_intructions_every=15
mem_display_to_cue_time_initial_training_s=0.75     #initially, both post and retro are 750 ms
mem_display_to_cue_time_retro_long_training_s=1.6        #in the second stage, retro are all 1.6 ms...
mem_display_to_cue_time_post_options_long_training_s=[1.6+0.2*counter for counter in range(11)]  #... whereas post are drawn from this set
short_block_done = 0
trials_long_perblock = 50
#-------------------------

num_images_per_display=8
image_index_list_for_shuffling=list(range(0,num_images_per_display))

num_images_available=10
num_practice_trials=7

#rand_val_for_image_selection=int(round(random()*19.))

permutation_data_file_string_practice='limited_permutations.xlsx'

image_side_dva=1.027    #2.0
image_eccentr_dva=2.3   #4.5

outline_side_relative_to_image_eccentricity=2.5
pointer_length_relative_to_image_eccentricity=.25

x_y_gains_per_angle=[[0.3826834323650898, 0.9238795325112867], [0.9238795325112867, 0.38268343236508984],
                    [0.9238795325112867, -0.3826834323650897], [0.38268343236508984, -0.9238795325112867],
                    [-0.3826834323650892, -0.923879532511287], [-0.9238795325112868, -0.3826834323650895], 
                    [-0.9238795325112866, 0.38268343236509], [-0.38268343236508956, 0.9238795325112868]]

dot_diam_dva=.2

#commented out stuff related to measuring screen distance online:
#max_height_accepted_cm=203.5
#min_height_accepted_cm=141.5
#
#height_offset_desktop_cm=37.42             #for curve of natural distance vs. height
#height_slope_desktop_cm_per_cm=0.1349
#
#height_offset_laptop_cm=-18.48
#height_slope_laptop_cm_per_cm=0.4119
#
#blindspot_entry_offset_cm=11.42             #for curve of natural distance vs. blind spot entry point
#blind_spot_entry_slope_cm_per_cm=3.569
#
#max_distance_accepted_laptop_cm=81.0        #min and max based on what was not excluded as outlier in earlier data
#min_distance_accepted_laptop_cm=22.0
#
#max_distance_accepted_desktop_cm=90.0
#min_distance_accepted_desktop_cm=29.0
#
#showDevice=1;
#showHeight=1;

#----------------------------
#set the things that differ between the 23 and the 32 key-response mapping version:
my_map = int(expInfo['mapping'])
if my_map == 23:
    my_map_0_or_1 = 0
elif my_map == 32:
    my_map_0_or_1 = 1
key_response_mapping_option=my_map_0_or_1       #0 for 23; 1 for 32; left = 2, down = 3; 23 = 67, 32 = 76
key_mapping_instruction_text_options=['To indicate your response you’ll use your right hand; press the left arrow with your index finger if the two groups of pictures both had the same picture at the location indicated by the red dotted line, or press the down arrow with your middle finger if two different pictures were shown at that location. See the image below:','To indicate your response you’ll use your right hand; press the down arrow with your middle finger if the two groups of pictures both had the same picture at the location indicated by the red dotted line, or press the left arrow with your index finger if two different pictures were shown at that location. See the image below:']
key_mapping_instruction_image_options=['RH_buttonbox_V1_practice.jpg','RH_buttonbox_V2_practice.jpg']
response_string_pair_options=[['left','down'],['down','left']]
same_finger_string_options=['index','middle']
different_finger_string_options=['middle','index']

key_mapping_instruction_text=key_mapping_instruction_text_options[key_response_mapping_option]
key_mapping_instruction_image=key_mapping_instruction_image_options[key_response_mapping_option]
response_string_pair=response_string_pair_options[key_response_mapping_option]
same_finger_string=same_finger_string_options[key_response_mapping_option]
different_finger_string=different_finger_string_options[key_response_mapping_option]
#-------------------------------

all_pic_names = ['bow.png','clown.png','car.png','book.png','bed.png','moon.png','strawberry.png','lemon.png','sun.png','flag.png']
all_pic_names_training=['bow.png','clown.png','car.png','book.png',
'bed.png','moon.png','strawberry.png','lemon.png','sun.png','flag.png']
intro_continue.keys = []
intro_continue.rt = []
_intro_continue_allKeys = []
# keep track of which components have finished
IntroComponents = [welcome, intro_continue]
for thisComponent in IntroComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
IntroClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Intro"-------
while continueRoutine:
    # get current time
    t = IntroClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=IntroClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *welcome* updates
    if welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome.frameNStart = frameN  # exact frame index
        welcome.tStart = t  # local t and not account for scr refresh
        welcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome, 'tStartRefresh')  # time at next scr refresh
        welcome.setAutoDraw(True)
    
    # *intro_continue* updates
    waitOnFlip = False
    if intro_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        intro_continue.frameNStart = frameN  # exact frame index
        intro_continue.tStart = t  # local t and not account for scr refresh
        intro_continue.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(intro_continue, 'tStartRefresh')  # time at next scr refresh
        intro_continue.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(intro_continue.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(intro_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if intro_continue.status == STARTED and not waitOnFlip:
        theseKeys = intro_continue.getKeys(keyList=None, waitRelease=False)
        _intro_continue_allKeys.extend(theseKeys)
        if len(_intro_continue_allKeys):
            intro_continue.keys = _intro_continue_allKeys[-1].name  # just the last key pressed
            intro_continue.rt = _intro_continue_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in IntroComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Intro"-------
for thisComponent in IntroComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('welcome.started', welcome.tStartRefresh)
thisExp.addData('welcome.stopped', welcome.tStopRefresh)
# the Routine "Intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_1"-------
continueRoutine = True
# update component parameters for each repeat
static_key1.keys = []
static_key1.rt = []
_static_key1_allKeys = []
# keep track of which components have finished
instruction_1Components = [static_instruction, static_key1, static_instruction_2]
for thisComponent in instruction_1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_1"-------
while continueRoutine:
    # get current time
    t = instruction_1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction* updates
    if static_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction.frameNStart = frameN  # exact frame index
        static_instruction.tStart = t  # local t and not account for scr refresh
        static_instruction.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction, 'tStartRefresh')  # time at next scr refresh
        static_instruction.setAutoDraw(True)
    
    # *static_key1* updates
    waitOnFlip = False
    if static_key1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1.frameNStart = frameN  # exact frame index
        static_key1.tStart = t  # local t and not account for scr refresh
        static_key1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1, 'tStartRefresh')  # time at next scr refresh
        static_key1.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1.status == STARTED and not waitOnFlip:
        theseKeys = static_key1.getKeys(keyList=None, waitRelease=False)
        _static_key1_allKeys.extend(theseKeys)
        if len(_static_key1_allKeys):
            static_key1.keys = _static_key1_allKeys[-1].name  # just the last key pressed
            static_key1.rt = _static_key1_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_2* updates
    if static_instruction_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_2.frameNStart = frameN  # exact frame index
        static_instruction_2.tStart = t  # local t and not account for scr refresh
        static_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_2, 'tStartRefresh')  # time at next scr refresh
        static_instruction_2.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_1"-------
for thisComponent in instruction_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_2"-------
continueRoutine = True
# update component parameters for each repeat
distance_cm = dist_cm

image_eccentr_cm = tan(3.14159*image_eccentr_dva/180.) * distance_cm
image_side_cm = 2. * tan(0.5*3.14159*image_side_dva/180.) * distance_cm

random_pic_names=all_pic_names_training[:]
shuffle(random_pic_names)
pic_1_name=random_pic_names[0]
array_image_1.setImage(pic_1_name, False)
array_image_1.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
array_image_1.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_2_name=random_pic_names[1]
array_image_2.setImage(pic_2_name, False)
array_image_2.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
array_image_2.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_3_name=random_pic_names[2]
array_image_3.setImage(pic_3_name, False)
array_image_3.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
array_image_3.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_4_name=random_pic_names[3]
array_image_4.setImage(pic_4_name, False)
array_image_4.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
array_image_4.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_5_name=random_pic_names[4]
array_image_5.setImage(pic_5_name, False)
array_image_5.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
array_image_5.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_6_name=random_pic_names[5]
array_image_6.setImage(pic_6_name, False)
array_image_6.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
array_image_6.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_7_name=random_pic_names[6]
array_image_7.setImage(pic_7_name, False)
array_image_7.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
array_image_7.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

pic_8_name=random_pic_names[7]
array_image_8.setImage(pic_8_name, False)
array_image_8.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
array_image_8.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
static_key1_2.keys = []
static_key1_2.rt = []
_static_key1_2_allKeys = []
array_image_1.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
array_image_1.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_1.setImage(pic_1_name)
array_image_2.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
array_image_2.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_2.setImage(pic_2_name)
array_image_3.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
array_image_3.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_3.setImage(pic_3_name)
array_image_4.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
array_image_4.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_4.setImage(pic_4_name)
array_image_5.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
array_image_5.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_5.setImage(pic_5_name)
array_image_6.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
array_image_6.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_6.setImage(pic_6_name)
array_image_7.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
array_image_7.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_7.setImage(pic_7_name)
array_image_8.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
array_image_8.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_8.setImage(pic_8_name)
polygon_17.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_2Components = [static_instruction_3, static_key1_2, static_instruction_4, array_image_1, array_image_2, array_image_3, array_image_4, array_image_5, array_image_6, array_image_7, array_image_8, polygon_17]
for thisComponent in instruction_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_2"-------
while continueRoutine:
    # get current time
    t = instruction_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_3* updates
    if static_instruction_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_3.frameNStart = frameN  # exact frame index
        static_instruction_3.tStart = t  # local t and not account for scr refresh
        static_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_3, 'tStartRefresh')  # time at next scr refresh
        static_instruction_3.setAutoDraw(True)
    
    # *static_key1_2* updates
    waitOnFlip = False
    if static_key1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_2.frameNStart = frameN  # exact frame index
        static_key1_2.tStart = t  # local t and not account for scr refresh
        static_key1_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_2, 'tStartRefresh')  # time at next scr refresh
        static_key1_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_2.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_2.getKeys(keyList=None, waitRelease=False)
        _static_key1_2_allKeys.extend(theseKeys)
        if len(_static_key1_2_allKeys):
            static_key1_2.keys = _static_key1_2_allKeys[-1].name  # just the last key pressed
            static_key1_2.rt = _static_key1_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_4* updates
    if static_instruction_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_4.frameNStart = frameN  # exact frame index
        static_instruction_4.tStart = t  # local t and not account for scr refresh
        static_instruction_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_4, 'tStartRefresh')  # time at next scr refresh
        static_instruction_4.setAutoDraw(True)
    
    # *array_image_1* updates
    if array_image_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_1.frameNStart = frameN  # exact frame index
        array_image_1.tStart = t  # local t and not account for scr refresh
        array_image_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_1, 'tStartRefresh')  # time at next scr refresh
        array_image_1.setAutoDraw(True)
    
    # *array_image_2* updates
    if array_image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_2.frameNStart = frameN  # exact frame index
        array_image_2.tStart = t  # local t and not account for scr refresh
        array_image_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_2, 'tStartRefresh')  # time at next scr refresh
        array_image_2.setAutoDraw(True)
    
    # *array_image_3* updates
    if array_image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_3.frameNStart = frameN  # exact frame index
        array_image_3.tStart = t  # local t and not account for scr refresh
        array_image_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_3, 'tStartRefresh')  # time at next scr refresh
        array_image_3.setAutoDraw(True)
    
    # *array_image_4* updates
    if array_image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_4.frameNStart = frameN  # exact frame index
        array_image_4.tStart = t  # local t and not account for scr refresh
        array_image_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_4, 'tStartRefresh')  # time at next scr refresh
        array_image_4.setAutoDraw(True)
    
    # *array_image_5* updates
    if array_image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_5.frameNStart = frameN  # exact frame index
        array_image_5.tStart = t  # local t and not account for scr refresh
        array_image_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_5, 'tStartRefresh')  # time at next scr refresh
        array_image_5.setAutoDraw(True)
    
    # *array_image_6* updates
    if array_image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_6.frameNStart = frameN  # exact frame index
        array_image_6.tStart = t  # local t and not account for scr refresh
        array_image_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_6, 'tStartRefresh')  # time at next scr refresh
        array_image_6.setAutoDraw(True)
    
    # *array_image_7* updates
    if array_image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_7.frameNStart = frameN  # exact frame index
        array_image_7.tStart = t  # local t and not account for scr refresh
        array_image_7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_7, 'tStartRefresh')  # time at next scr refresh
        array_image_7.setAutoDraw(True)
    
    # *array_image_8* updates
    if array_image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        array_image_8.frameNStart = frameN  # exact frame index
        array_image_8.tStart = t  # local t and not account for scr refresh
        array_image_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_8, 'tStartRefresh')  # time at next scr refresh
        array_image_8.setAutoDraw(True)
    
    # *polygon_17* updates
    if polygon_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_17.frameNStart = frameN  # exact frame index
        polygon_17.tStart = t  # local t and not account for scr refresh
        polygon_17.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_17, 'tStartRefresh')  # time at next scr refresh
        polygon_17.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_2"-------
for thisComponent in instruction_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_3"-------
continueRoutine = True
# update component parameters for each repeat
array_image_1b.setImage(pic_1_name, False)
array_image_1b.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
array_image_1b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_2b.setImage(pic_2_name, False)
array_image_2b.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
array_image_2b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_3b.setImage(pic_3_name, False)
array_image_3b.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
array_image_3b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_4b.setImage(pic_4_name, False)
array_image_4b.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
array_image_4b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_5b.setImage(pic_5_name, False)
array_image_5b.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
array_image_5b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_6b.setImage(pic_6_name, False)
array_image_6b.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
array_image_6b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_7b.setImage(pic_7_name, False)
array_image_7b.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
array_image_7b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_8b.setImage(pic_8_name, False)
array_image_8b.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
array_image_8b.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
static_key1_6.keys = []
static_key1_6.rt = []
_static_key1_6_allKeys = []
array_image_1b.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
array_image_1b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_1b.setImage(pic_1_name)
array_image_2b.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
array_image_2b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_2b.setImage(pic_2_name)
array_image_3b.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
array_image_3b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_3b.setImage(pic_3_name)
array_image_4b.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
array_image_4b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_4b.setImage(pic_4_name)
array_image_5b.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
array_image_5b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_5b.setImage(pic_5_name)
array_image_6b.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
array_image_6b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_6b.setImage(pic_6_name)
array_image_7b.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
array_image_7b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_7b.setImage(pic_7_name)
array_image_8b.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
array_image_8b.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_8b.setImage(pic_8_name)
polygon_18.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_3Components = [static_instruction_15, static_key1_6, static_instruction_16, array_image_1b, array_image_2b, array_image_3b, array_image_4b, array_image_5b, array_image_6b, array_image_7b, array_image_8b, polygon_18]
for thisComponent in instruction_3Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_3Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_3"-------
while continueRoutine:
    # get current time
    t = instruction_3Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_3Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_15* updates
    if static_instruction_15.status == NOT_STARTED and tThisFlip >= 1.-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_15.frameNStart = frameN  # exact frame index
        static_instruction_15.tStart = t  # local t and not account for scr refresh
        static_instruction_15.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_15, 'tStartRefresh')  # time at next scr refresh
        static_instruction_15.setAutoDraw(True)
    
    # *static_key1_6* updates
    waitOnFlip = False
    if static_key1_6.status == NOT_STARTED and tThisFlip >= 1.-frameTolerance:
        # keep track of start time/frame for later
        static_key1_6.frameNStart = frameN  # exact frame index
        static_key1_6.tStart = t  # local t and not account for scr refresh
        static_key1_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_6, 'tStartRefresh')  # time at next scr refresh
        static_key1_6.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_6.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_6.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_6.getKeys(keyList=None, waitRelease=False)
        _static_key1_6_allKeys.extend(theseKeys)
        if len(_static_key1_6_allKeys):
            static_key1_6.keys = _static_key1_6_allKeys[-1].name  # just the last key pressed
            static_key1_6.rt = _static_key1_6_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_16* updates
    if static_instruction_16.status == NOT_STARTED and tThisFlip >= 1.-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_16.frameNStart = frameN  # exact frame index
        static_instruction_16.tStart = t  # local t and not account for scr refresh
        static_instruction_16.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_16, 'tStartRefresh')  # time at next scr refresh
        static_instruction_16.setAutoDraw(True)
    
    # *array_image_1b* updates
    if array_image_1b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_1b.frameNStart = frameN  # exact frame index
        array_image_1b.tStart = t  # local t and not account for scr refresh
        array_image_1b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_1b, 'tStartRefresh')  # time at next scr refresh
        array_image_1b.setAutoDraw(True)
    if array_image_1b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_1b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_1b.tStop = t  # not accounting for scr refresh
            array_image_1b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_1b, 'tStopRefresh')  # time at next scr refresh
            array_image_1b.setAutoDraw(False)
    
    # *array_image_2b* updates
    if array_image_2b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_2b.frameNStart = frameN  # exact frame index
        array_image_2b.tStart = t  # local t and not account for scr refresh
        array_image_2b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_2b, 'tStartRefresh')  # time at next scr refresh
        array_image_2b.setAutoDraw(True)
    if array_image_2b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_2b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_2b.tStop = t  # not accounting for scr refresh
            array_image_2b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_2b, 'tStopRefresh')  # time at next scr refresh
            array_image_2b.setAutoDraw(False)
    
    # *array_image_3b* updates
    if array_image_3b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_3b.frameNStart = frameN  # exact frame index
        array_image_3b.tStart = t  # local t and not account for scr refresh
        array_image_3b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_3b, 'tStartRefresh')  # time at next scr refresh
        array_image_3b.setAutoDraw(True)
    if array_image_3b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_3b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_3b.tStop = t  # not accounting for scr refresh
            array_image_3b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_3b, 'tStopRefresh')  # time at next scr refresh
            array_image_3b.setAutoDraw(False)
    
    # *array_image_4b* updates
    if array_image_4b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_4b.frameNStart = frameN  # exact frame index
        array_image_4b.tStart = t  # local t and not account for scr refresh
        array_image_4b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_4b, 'tStartRefresh')  # time at next scr refresh
        array_image_4b.setAutoDraw(True)
    if array_image_4b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_4b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_4b.tStop = t  # not accounting for scr refresh
            array_image_4b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_4b, 'tStopRefresh')  # time at next scr refresh
            array_image_4b.setAutoDraw(False)
    
    # *array_image_5b* updates
    if array_image_5b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_5b.frameNStart = frameN  # exact frame index
        array_image_5b.tStart = t  # local t and not account for scr refresh
        array_image_5b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_5b, 'tStartRefresh')  # time at next scr refresh
        array_image_5b.setAutoDraw(True)
    if array_image_5b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_5b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_5b.tStop = t  # not accounting for scr refresh
            array_image_5b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_5b, 'tStopRefresh')  # time at next scr refresh
            array_image_5b.setAutoDraw(False)
    
    # *array_image_6b* updates
    if array_image_6b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_6b.frameNStart = frameN  # exact frame index
        array_image_6b.tStart = t  # local t and not account for scr refresh
        array_image_6b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_6b, 'tStartRefresh')  # time at next scr refresh
        array_image_6b.setAutoDraw(True)
    if array_image_6b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_6b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_6b.tStop = t  # not accounting for scr refresh
            array_image_6b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_6b, 'tStopRefresh')  # time at next scr refresh
            array_image_6b.setAutoDraw(False)
    
    # *array_image_7b* updates
    if array_image_7b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_7b.frameNStart = frameN  # exact frame index
        array_image_7b.tStart = t  # local t and not account for scr refresh
        array_image_7b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_7b, 'tStartRefresh')  # time at next scr refresh
        array_image_7b.setAutoDraw(True)
    if array_image_7b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_7b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_7b.tStop = t  # not accounting for scr refresh
            array_image_7b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_7b, 'tStopRefresh')  # time at next scr refresh
            array_image_7b.setAutoDraw(False)
    
    # *array_image_8b* updates
    if array_image_8b.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
        # keep track of start time/frame for later
        array_image_8b.frameNStart = frameN  # exact frame index
        array_image_8b.tStart = t  # local t and not account for scr refresh
        array_image_8b.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_8b, 'tStartRefresh')  # time at next scr refresh
        array_image_8b.setAutoDraw(True)
    if array_image_8b.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > array_image_8b.tStartRefresh + .5-frameTolerance:
            # keep track of stop time/frame for later
            array_image_8b.tStop = t  # not accounting for scr refresh
            array_image_8b.frameNStop = frameN  # exact frame index
            win.timeOnFlip(array_image_8b, 'tStopRefresh')  # time at next scr refresh
            array_image_8b.setAutoDraw(False)
    
    # *polygon_18* updates
    if polygon_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_18.frameNStart = frameN  # exact frame index
        polygon_18.tStart = t  # local t and not account for scr refresh
        polygon_18.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_18, 'tStartRefresh')  # time at next scr refresh
        polygon_18.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_3Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_3"-------
for thisComponent in instruction_3Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_3" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_4"-------
continueRoutine = True
# update component parameters for each repeat
array_image_1c.setImage(pic_1_name, False)
array_image_1c.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
array_image_1c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_2c.setImage(pic_2_name, False)
array_image_2c.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
array_image_2c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_3c.setImage(pic_3_name, False)
array_image_3c.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
array_image_3c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_4c.setImage(pic_4_name, False)
array_image_4c.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
array_image_4c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_5c.setImage(pic_5_name, False)
array_image_5c.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
array_image_5c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_6c.setImage(pic_6_name, False)
array_image_6c.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
array_image_6c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_7c.setImage(pic_7_name, False)
array_image_7c.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
array_image_7c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

array_image_8c.setImage(pic_8_name, False)
array_image_8c.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
array_image_8c.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)

static_key1_7.keys = []
static_key1_7.rt = []
_static_key1_7_allKeys = []
array_image_1c.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
array_image_1c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_1c.setImage(pic_1_name)
array_image_2c.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
array_image_2c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_2c.setImage(pic_2_name)
array_image_3c.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
array_image_3c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_3c.setImage(pic_3_name)
array_image_4c.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
array_image_4c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_4c.setImage(pic_4_name)
array_image_5c.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
array_image_5c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_5c.setImage(pic_5_name)
array_image_6c.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
array_image_6c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_6c.setImage(pic_6_name)
array_image_7c.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
array_image_7c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_7c.setImage(pic_7_name)
array_image_8c.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
array_image_8c.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
array_image_8c.setImage(pic_8_name)
polygon_19.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_4Components = [static_instruction_17, static_key1_7, static_instruction_18, array_image_1c, array_image_2c, array_image_3c, array_image_4c, array_image_5c, array_image_6c, array_image_7c, array_image_8c, polygon_19]
for thisComponent in instruction_4Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_4Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_4"-------
while continueRoutine:
    # get current time
    t = instruction_4Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_4Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_17* updates
    if static_instruction_17.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_17.frameNStart = frameN  # exact frame index
        static_instruction_17.tStart = t  # local t and not account for scr refresh
        static_instruction_17.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_17, 'tStartRefresh')  # time at next scr refresh
        static_instruction_17.setAutoDraw(True)
    
    # *static_key1_7* updates
    waitOnFlip = False
    if static_key1_7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_7.frameNStart = frameN  # exact frame index
        static_key1_7.tStart = t  # local t and not account for scr refresh
        static_key1_7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_7, 'tStartRefresh')  # time at next scr refresh
        static_key1_7.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_7.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_7.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_7.getKeys(keyList=None, waitRelease=False)
        _static_key1_7_allKeys.extend(theseKeys)
        if len(_static_key1_7_allKeys):
            static_key1_7.keys = _static_key1_7_allKeys[-1].name  # just the last key pressed
            static_key1_7.rt = _static_key1_7_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_18* updates
    if static_instruction_18.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_18.frameNStart = frameN  # exact frame index
        static_instruction_18.tStart = t  # local t and not account for scr refresh
        static_instruction_18.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_18, 'tStartRefresh')  # time at next scr refresh
        static_instruction_18.setAutoDraw(True)
    
    # *array_image_1c* updates
    if array_image_1c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_1c.frameNStart = frameN  # exact frame index
        array_image_1c.tStart = t  # local t and not account for scr refresh
        array_image_1c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_1c, 'tStartRefresh')  # time at next scr refresh
        array_image_1c.setAutoDraw(True)
    
    # *array_image_2c* updates
    if array_image_2c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_2c.frameNStart = frameN  # exact frame index
        array_image_2c.tStart = t  # local t and not account for scr refresh
        array_image_2c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_2c, 'tStartRefresh')  # time at next scr refresh
        array_image_2c.setAutoDraw(True)
    
    # *array_image_3c* updates
    if array_image_3c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_3c.frameNStart = frameN  # exact frame index
        array_image_3c.tStart = t  # local t and not account for scr refresh
        array_image_3c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_3c, 'tStartRefresh')  # time at next scr refresh
        array_image_3c.setAutoDraw(True)
    
    # *array_image_4c* updates
    if array_image_4c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_4c.frameNStart = frameN  # exact frame index
        array_image_4c.tStart = t  # local t and not account for scr refresh
        array_image_4c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_4c, 'tStartRefresh')  # time at next scr refresh
        array_image_4c.setAutoDraw(True)
    
    # *array_image_5c* updates
    if array_image_5c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_5c.frameNStart = frameN  # exact frame index
        array_image_5c.tStart = t  # local t and not account for scr refresh
        array_image_5c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_5c, 'tStartRefresh')  # time at next scr refresh
        array_image_5c.setAutoDraw(True)
    
    # *array_image_6c* updates
    if array_image_6c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_6c.frameNStart = frameN  # exact frame index
        array_image_6c.tStart = t  # local t and not account for scr refresh
        array_image_6c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_6c, 'tStartRefresh')  # time at next scr refresh
        array_image_6c.setAutoDraw(True)
    
    # *array_image_7c* updates
    if array_image_7c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_7c.frameNStart = frameN  # exact frame index
        array_image_7c.tStart = t  # local t and not account for scr refresh
        array_image_7c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_7c, 'tStartRefresh')  # time at next scr refresh
        array_image_7c.setAutoDraw(True)
    
    # *array_image_8c* updates
    if array_image_8c.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        array_image_8c.frameNStart = frameN  # exact frame index
        array_image_8c.tStart = t  # local t and not account for scr refresh
        array_image_8c.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(array_image_8c, 'tStartRefresh')  # time at next scr refresh
        array_image_8c.setAutoDraw(True)
    
    # *polygon_19* updates
    if polygon_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_19.frameNStart = frameN  # exact frame index
        polygon_19.tStart = t  # local t and not account for scr refresh
        polygon_19.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_19, 'tStartRefresh')  # time at next scr refresh
        polygon_19.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_4Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_4"-------
for thisComponent in instruction_4Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_4" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instructions_4to5"-------
continueRoutine = True
# update component parameters for each repeat
static_key1_11.keys = []
static_key1_11.rt = []
_static_key1_11_allKeys = []
# keep track of which components have finished
instructions_4to5Components = [static_instruction_25, static_key1_11, static_instruction_26]
for thisComponent in instructions_4to5Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructions_4to5Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions_4to5"-------
while continueRoutine:
    # get current time
    t = instructions_4to5Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructions_4to5Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_25* updates
    if static_instruction_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_25.frameNStart = frameN  # exact frame index
        static_instruction_25.tStart = t  # local t and not account for scr refresh
        static_instruction_25.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_25, 'tStartRefresh')  # time at next scr refresh
        static_instruction_25.setAutoDraw(True)
    
    # *static_key1_11* updates
    waitOnFlip = False
    if static_key1_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_11.frameNStart = frameN  # exact frame index
        static_key1_11.tStart = t  # local t and not account for scr refresh
        static_key1_11.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_11, 'tStartRefresh')  # time at next scr refresh
        static_key1_11.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_11.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_11.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_11.getKeys(keyList=None, waitRelease=False)
        _static_key1_11_allKeys.extend(theseKeys)
        if len(_static_key1_11_allKeys):
            static_key1_11.keys = _static_key1_11_allKeys[-1].name  # just the last key pressed
            static_key1_11.rt = _static_key1_11_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_26* updates
    if static_instruction_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_26.frameNStart = frameN  # exact frame index
        static_instruction_26.tStart = t  # local t and not account for scr refresh
        static_instruction_26.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_26, 'tStartRefresh')  # time at next scr refresh
        static_instruction_26.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructions_4to5Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions_4to5"-------
for thisComponent in instructions_4to5Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instructions_4to5" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_5"-------
continueRoutine = True
# update component parameters for each repeat
static_key1_8.keys = []
static_key1_8.rt = []
_static_key1_8_allKeys = []
polygon_20.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
pointer_dot_0.setPos((x_y_gains_per_angle[1][0]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*y_scale))
pointer_dot_0.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
pointer_dot_1.setPos((x_y_gains_per_angle[1][0]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*x_scale*.66667, x_y_gains_per_angle[1][1]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*y_scale*.66667))
pointer_dot_1.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
pointer_dot_2.setPos((x_y_gains_per_angle[1][0]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*x_scale*.33333, x_y_gains_per_angle[1][1]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*y_scale*.33333))
pointer_dot_2.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
# keep track of which components have finished
instruction_5Components = [static_instruction_19, static_key1_8, static_instruction_20, polygon_20, pointer_dot_0, pointer_dot_1, pointer_dot_2]
for thisComponent in instruction_5Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_5Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_5"-------
while continueRoutine:
    # get current time
    t = instruction_5Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_5Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_19* updates
    if static_instruction_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_19.frameNStart = frameN  # exact frame index
        static_instruction_19.tStart = t  # local t and not account for scr refresh
        static_instruction_19.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_19, 'tStartRefresh')  # time at next scr refresh
        static_instruction_19.setAutoDraw(True)
    
    # *static_key1_8* updates
    waitOnFlip = False
    if static_key1_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_8.frameNStart = frameN  # exact frame index
        static_key1_8.tStart = t  # local t and not account for scr refresh
        static_key1_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_8, 'tStartRefresh')  # time at next scr refresh
        static_key1_8.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_8.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_8.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_8.getKeys(keyList=None, waitRelease=False)
        _static_key1_8_allKeys.extend(theseKeys)
        if len(_static_key1_8_allKeys):
            static_key1_8.keys = _static_key1_8_allKeys[-1].name  # just the last key pressed
            static_key1_8.rt = _static_key1_8_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_20* updates
    if static_instruction_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_20.frameNStart = frameN  # exact frame index
        static_instruction_20.tStart = t  # local t and not account for scr refresh
        static_instruction_20.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_20, 'tStartRefresh')  # time at next scr refresh
        static_instruction_20.setAutoDraw(True)
    
    # *polygon_20* updates
    if polygon_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_20.frameNStart = frameN  # exact frame index
        polygon_20.tStart = t  # local t and not account for scr refresh
        polygon_20.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_20, 'tStartRefresh')  # time at next scr refresh
        polygon_20.setAutoDraw(True)
    
    # *pointer_dot_0* updates
    if pointer_dot_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        pointer_dot_0.frameNStart = frameN  # exact frame index
        pointer_dot_0.tStart = t  # local t and not account for scr refresh
        pointer_dot_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(pointer_dot_0, 'tStartRefresh')  # time at next scr refresh
        pointer_dot_0.setAutoDraw(True)
    
    # *pointer_dot_1* updates
    if pointer_dot_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        pointer_dot_1.frameNStart = frameN  # exact frame index
        pointer_dot_1.tStart = t  # local t and not account for scr refresh
        pointer_dot_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(pointer_dot_1, 'tStartRefresh')  # time at next scr refresh
        pointer_dot_1.setAutoDraw(True)
    
    # *pointer_dot_2* updates
    if pointer_dot_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        pointer_dot_2.frameNStart = frameN  # exact frame index
        pointer_dot_2.tStart = t  # local t and not account for scr refresh
        pointer_dot_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(pointer_dot_2, 'tStartRefresh')  # time at next scr refresh
        pointer_dot_2.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_5Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_5"-------
for thisComponent in instruction_5Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('pointer_dot_0.started', pointer_dot_0.tStartRefresh)
thisExp.addData('pointer_dot_0.stopped', pointer_dot_0.tStopRefresh)
thisExp.addData('pointer_dot_1.started', pointer_dot_1.tStartRefresh)
thisExp.addData('pointer_dot_1.stopped', pointer_dot_1.tStopRefresh)
thisExp.addData('pointer_dot_2.started', pointer_dot_2.tStartRefresh)
thisExp.addData('pointer_dot_2.stopped', pointer_dot_2.tStopRefresh)
# the Routine "instruction_5" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_6"-------
continueRoutine = True
# update component parameters for each repeat
static_key1_12.keys = []
static_key1_12.rt = []
_static_key1_12_allKeys = []
polygon_24.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_6Components = [static_instruction_27, static_key1_12, static_instruction_28, polygon_24]
for thisComponent in instruction_6Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_6Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_6"-------
while continueRoutine:
    # get current time
    t = instruction_6Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_6Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_27* updates
    if static_instruction_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_27.frameNStart = frameN  # exact frame index
        static_instruction_27.tStart = t  # local t and not account for scr refresh
        static_instruction_27.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_27, 'tStartRefresh')  # time at next scr refresh
        static_instruction_27.setAutoDraw(True)
    
    # *static_key1_12* updates
    waitOnFlip = False
    if static_key1_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_12.frameNStart = frameN  # exact frame index
        static_key1_12.tStart = t  # local t and not account for scr refresh
        static_key1_12.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_12, 'tStartRefresh')  # time at next scr refresh
        static_key1_12.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_12.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_12.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_12.getKeys(keyList=None, waitRelease=False)
        _static_key1_12_allKeys.extend(theseKeys)
        if len(_static_key1_12_allKeys):
            static_key1_12.keys = _static_key1_12_allKeys[-1].name  # just the last key pressed
            static_key1_12.rt = _static_key1_12_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_28* updates
    if static_instruction_28.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_28.frameNStart = frameN  # exact frame index
        static_instruction_28.tStart = t  # local t and not account for scr refresh
        static_instruction_28.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_28, 'tStartRefresh')  # time at next scr refresh
        static_instruction_28.setAutoDraw(True)
    
    # *polygon_24* updates
    if polygon_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_24.frameNStart = frameN  # exact frame index
        polygon_24.tStart = t  # local t and not account for scr refresh
        polygon_24.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_24, 'tStartRefresh')  # time at next scr refresh
        polygon_24.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_6Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_6"-------
for thisComponent in instruction_6Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_6" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_67"-------
continueRoutine = True
# update component parameters for each repeat
static_instruction_37.setText(key_mapping_instruction_text)
static_key1_17.keys = []
static_key1_17.rt = []
_static_key1_17_allKeys = []
SD_image.setImage(key_mapping_instruction_image)
# keep track of which components have finished
instruction_67Components = [static_instruction_37, static_key1_17, static_instruction_38, SD_image]
for thisComponent in instruction_67Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_67Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_67"-------
while continueRoutine:
    # get current time
    t = instruction_67Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_67Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_37* updates
    if static_instruction_37.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_37.frameNStart = frameN  # exact frame index
        static_instruction_37.tStart = t  # local t and not account for scr refresh
        static_instruction_37.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_37, 'tStartRefresh')  # time at next scr refresh
        static_instruction_37.setAutoDraw(True)
    
    # *static_key1_17* updates
    waitOnFlip = False
    if static_key1_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_17.frameNStart = frameN  # exact frame index
        static_key1_17.tStart = t  # local t and not account for scr refresh
        static_key1_17.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_17, 'tStartRefresh')  # time at next scr refresh
        static_key1_17.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_17.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_17.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_17.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_17.getKeys(keyList=None, waitRelease=False)
        _static_key1_17_allKeys.extend(theseKeys)
        if len(_static_key1_17_allKeys):
            static_key1_17.keys = _static_key1_17_allKeys[-1].name  # just the last key pressed
            static_key1_17.rt = _static_key1_17_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_38* updates
    if static_instruction_38.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_38.frameNStart = frameN  # exact frame index
        static_instruction_38.tStart = t  # local t and not account for scr refresh
        static_instruction_38.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_38, 'tStartRefresh')  # time at next scr refresh
        static_instruction_38.setAutoDraw(True)
    
    # *SD_image* updates
    if SD_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        SD_image.frameNStart = frameN  # exact frame index
        SD_image.tStart = t  # local t and not account for scr refresh
        SD_image.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(SD_image, 'tStartRefresh')  # time at next scr refresh
        SD_image.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_67Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_67"-------
for thisComponent in instruction_67Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('SD_image.started', SD_image.tStartRefresh)
thisExp.addData('SD_image.stopped', SD_image.tStopRefresh)
# the Routine "instruction_67" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_67b"-------
continueRoutine = True
# update component parameters for each repeat
static_key1_18.keys = []
static_key1_18.rt = []
_static_key1_18_allKeys = []
polygon_37.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_67bComponents = [static_instruction_39, static_key1_18, static_instruction_40, polygon_37]
for thisComponent in instruction_67bComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_67bClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_67b"-------
while continueRoutine:
    # get current time
    t = instruction_67bClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_67bClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_39* updates
    if static_instruction_39.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_39.frameNStart = frameN  # exact frame index
        static_instruction_39.tStart = t  # local t and not account for scr refresh
        static_instruction_39.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_39, 'tStartRefresh')  # time at next scr refresh
        static_instruction_39.setAutoDraw(True)
    
    # *static_key1_18* updates
    waitOnFlip = False
    if static_key1_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_18.frameNStart = frameN  # exact frame index
        static_key1_18.tStart = t  # local t and not account for scr refresh
        static_key1_18.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_18, 'tStartRefresh')  # time at next scr refresh
        static_key1_18.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_18.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_18.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_18.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_18.getKeys(keyList=None, waitRelease=False)
        _static_key1_18_allKeys.extend(theseKeys)
        if len(_static_key1_18_allKeys):
            static_key1_18.keys = _static_key1_18_allKeys[-1].name  # just the last key pressed
            static_key1_18.rt = _static_key1_18_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_40* updates
    if static_instruction_40.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_40.frameNStart = frameN  # exact frame index
        static_instruction_40.tStart = t  # local t and not account for scr refresh
        static_instruction_40.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_40, 'tStartRefresh')  # time at next scr refresh
        static_instruction_40.setAutoDraw(True)
    
    # *polygon_37* updates
    if polygon_37.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_37.frameNStart = frameN  # exact frame index
        polygon_37.tStart = t  # local t and not account for scr refresh
        polygon_37.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_37, 'tStartRefresh')  # time at next scr refresh
        polygon_37.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_67bComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_67b"-------
for thisComponent in instruction_67bComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_67b" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_7"-------
continueRoutine = True
# update component parameters for each repeat
static_key1_14.keys = []
static_key1_14.rt = []
_static_key1_14_allKeys = []
polygon_29.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
# keep track of which components have finished
instruction_7Components = [static_instruction_31, static_key1_14, static_instruction_32, polygon_29]
for thisComponent in instruction_7Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_7Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_7"-------
while continueRoutine:
    # get current time
    t = instruction_7Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_7Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_31* updates
    if static_instruction_31.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_31.frameNStart = frameN  # exact frame index
        static_instruction_31.tStart = t  # local t and not account for scr refresh
        static_instruction_31.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_31, 'tStartRefresh')  # time at next scr refresh
        static_instruction_31.setAutoDraw(True)
    
    # *static_key1_14* updates
    waitOnFlip = False
    if static_key1_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_14.frameNStart = frameN  # exact frame index
        static_key1_14.tStart = t  # local t and not account for scr refresh
        static_key1_14.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_14, 'tStartRefresh')  # time at next scr refresh
        static_key1_14.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_14.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_14.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_14.getKeys(keyList=None, waitRelease=False)
        _static_key1_14_allKeys.extend(theseKeys)
        if len(_static_key1_14_allKeys):
            static_key1_14.keys = _static_key1_14_allKeys[-1].name  # just the last key pressed
            static_key1_14.rt = _static_key1_14_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *static_instruction_32* updates
    if static_instruction_32.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_32.frameNStart = frameN  # exact frame index
        static_instruction_32.tStart = t  # local t and not account for scr refresh
        static_instruction_32.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_32, 'tStartRefresh')  # time at next scr refresh
        static_instruction_32.setAutoDraw(True)
    
    # *polygon_29* updates
    if polygon_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_29.frameNStart = frameN  # exact frame index
        polygon_29.tStart = t  # local t and not account for scr refresh
        polygon_29.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_29, 'tStartRefresh')  # time at next scr refresh
        polygon_29.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_7Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_7"-------
for thisComponent in instruction_7Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction_7" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials_2 = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(permutation_data_file_string_practice),
    seed=None, name='trials_2')
thisExp.addLoop(trials_2)  # add the loop to the experiment
thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
if thisTrial_2 != None:
    for paramName in thisTrial_2:
        exec('{} = thisTrial_2[paramName]'.format(paramName))

for thisTrial_2 in trials_2:
    currentLoop = trials_2
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "load_practice_trials"-------
    continueRoutine = True
    # update component parameters for each repeat
    
    practice_trial_image_list.append([Used_image_1,Used_image_2,Used_image_3,Used_image_4,Used_image_5,Used_image_6,Used_image_7,Used_image_8,Unused_image_1,Unused_image_2])
    if (practice_trial_load_number+1)<=num_practice_trials/2:
        retro_0_post_1_practice.append(0)
        change_0_1_practice.append(0)
    else:
        retro_0_post_1_practice.append(1)
        change_0_1_practice.append(1)
        
    mem_display_to_cue_times_practice.append(mem_display_to_cue_time_practice_s)
    
    # keep track of which components have finished
    load_practice_trialsComponents = []
    for thisComponent in load_practice_trialsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    load_practice_trialsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "load_practice_trials"-------
    while continueRoutine:
        # get current time
        t = load_practice_trialsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=load_practice_trialsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in load_practice_trialsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "load_practice_trials"-------
    for thisComponent in load_practice_trialsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_trial_load_number=practice_trial_load_number+1
    
    if practice_trial_load_number>=num_practice_trials:
        trials_2.finished = True
        
        shuffle(retro_0_post_1_practice)
        shuffle(mem_display_to_cue_times_practice)
        shuffle(change_0_1_practice)
        
        practice_trial_index=0
        confidence_onset_time_s=0.
        confidence_image_onset_time_s=0.
     
    # the Routine "load_practice_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'trials_2'


# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=num_practice_trials, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "get_ready_practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    if practice_trial_index == 0:
        practice_trial_text = "Get ready for practice trial 1 out of " + str(num_practice_trials)+".\n\nPlease don't forget to keep your eyes on the dot in the middle, and to use your right hand to indicate 'Same' or 'Different' using your "+same_finger_string+" and "+different_finger_string+" finger, respectively."
    elif practice_trial_index >= num_practice_trials/2.:
        practice_trial_text = "Get ready for practice trial " + str(practice_trial_index+1) + " out of " + str(num_practice_trials)
    else:
        practice_trial_text = "Get ready for practice trial " + str(practice_trial_index+1) + " out of " + str(num_practice_trials)+".\n\nPlease don't forget to keep your eyes on the dot in the middle, and to use your right hand to indicate 'Same' or 'Different' with your "+same_finger_string+" and "+different_finger_string+" finger, respectively, and then your left hand to indicate your confidence in that choice (ring finger=certain, middle finger=doubt, index finger=guess)."
    
    static_instruction_29.setText(practice_trial_text,False);
    
    
    static_instruction_29.setText(practice_trial_text)
    static_key1_13.keys = []
    static_key1_13.rt = []
    _static_key1_13_allKeys = []
    polygon_25.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
    # keep track of which components have finished
    get_ready_practiceComponents = [static_instruction_29, static_key1_13, static_instruction_30, polygon_25]
    for thisComponent in get_ready_practiceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    get_ready_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "get_ready_practice"-------
    while continueRoutine:
        # get current time
        t = get_ready_practiceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=get_ready_practiceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *static_instruction_29* updates
        if static_instruction_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            static_instruction_29.frameNStart = frameN  # exact frame index
            static_instruction_29.tStart = t  # local t and not account for scr refresh
            static_instruction_29.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(static_instruction_29, 'tStartRefresh')  # time at next scr refresh
            static_instruction_29.setAutoDraw(True)
        
        # *static_key1_13* updates
        waitOnFlip = False
        if static_key1_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            static_key1_13.frameNStart = frameN  # exact frame index
            static_key1_13.tStart = t  # local t and not account for scr refresh
            static_key1_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(static_key1_13, 'tStartRefresh')  # time at next scr refresh
            static_key1_13.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(static_key1_13.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(static_key1_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if static_key1_13.status == STARTED and not waitOnFlip:
            theseKeys = static_key1_13.getKeys(keyList=None, waitRelease=False)
            _static_key1_13_allKeys.extend(theseKeys)
            if len(_static_key1_13_allKeys):
                static_key1_13.keys = _static_key1_13_allKeys[-1].name  # just the last key pressed
                static_key1_13.rt = _static_key1_13_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *static_instruction_30* updates
        if static_instruction_30.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            static_instruction_30.frameNStart = frameN  # exact frame index
            static_instruction_30.tStart = t  # local t and not account for scr refresh
            static_instruction_30.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(static_instruction_30, 'tStartRefresh')  # time at next scr refresh
            static_instruction_30.setAutoDraw(True)
        
        # *polygon_25* updates
        if polygon_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_25.frameNStart = frameN  # exact frame index
            polygon_25.tStart = t  # local t and not account for scr refresh
            polygon_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_25, 'tStartRefresh')  # time at next scr refresh
            polygon_25.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in get_ready_practiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "get_ready_practice"-------
    for thisComponent in get_ready_practiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "get_ready_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "practice_mem_display"-------
    continueRoutine = True
    # update component parameters for each repeat
    Image_1=all_pic_names_training[practice_trial_image_list[practice_trial_index][0]]
    array_image_1d.setImage(Image_1, False)
    array_image_1d.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
    array_image_1d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_2=all_pic_names_training[practice_trial_image_list[practice_trial_index][1]]
    array_image_2d.setImage(Image_2, False)
    array_image_2d.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
    array_image_2d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_3=all_pic_names_training[practice_trial_image_list[practice_trial_index][2]]
    array_image_3d.setImage(Image_3, False)
    array_image_3d.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
    array_image_3d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_4=all_pic_names_training[practice_trial_image_list[practice_trial_index][3]]
    array_image_4d.setImage(Image_4, False)
    array_image_4d.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
    array_image_4d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_5=all_pic_names_training[practice_trial_image_list[practice_trial_index][4]]
    array_image_5d.setImage(Image_5, False)
    array_image_5d.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
    array_image_5d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_6=all_pic_names_training[practice_trial_image_list[practice_trial_index][5]]
    array_image_6d.setImage(Image_6, False)
    array_image_6d.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
    array_image_6d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_7=all_pic_names_training[practice_trial_image_list[practice_trial_index][6]]
    array_image_7d.setImage(Image_7, False)
    array_image_7d.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
    array_image_7d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_8=all_pic_names_training[practice_trial_image_list[practice_trial_index][7]]
    array_image_8d.setImage(Image_8, False)
    array_image_8d.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
    array_image_8d.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    thisExp.addData('memDisplayOnTimePractice',timer.getTime())
    
    #storing variables specifying trial
    for image_index in range(8):
        thisExp.addData('mem_image_'+str(image_index)+'_practice',practice_trial_image_list[practice_trial_index][image_index])
    array_image_1d.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
    array_image_1d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_1d.setImage(Image_1)
    array_image_2d.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
    array_image_2d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_2d.setImage(Image_2)
    array_image_3d.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
    array_image_3d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_3d.setImage(Image_3)
    array_image_4d.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
    array_image_4d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_4d.setImage(Image_4)
    array_image_5d.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
    array_image_5d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_5d.setImage(Image_5)
    array_image_6d.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
    array_image_6d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_6d.setImage(Image_6)
    array_image_7d.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
    array_image_7d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_7d.setImage(Image_7)
    array_image_8d.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
    array_image_8d.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_8d.setImage(Image_8)
    polygon_26.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
    # keep track of which components have finished
    practice_mem_displayComponents = [array_image_1d, array_image_2d, array_image_3d, array_image_4d, array_image_5d, array_image_6d, array_image_7d, array_image_8d, polygon_26]
    for thisComponent in practice_mem_displayComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    practice_mem_displayClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "practice_mem_display"-------
    while continueRoutine:
        # get current time
        t = practice_mem_displayClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=practice_mem_displayClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *array_image_1d* updates
        if array_image_1d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_1d.frameNStart = frameN  # exact frame index
            array_image_1d.tStart = t  # local t and not account for scr refresh
            array_image_1d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_1d, 'tStartRefresh')  # time at next scr refresh
            array_image_1d.setAutoDraw(True)
        if array_image_1d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_1d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_1d.tStop = t  # not accounting for scr refresh
                array_image_1d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_1d, 'tStopRefresh')  # time at next scr refresh
                array_image_1d.setAutoDraw(False)
        
        # *array_image_2d* updates
        if array_image_2d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_2d.frameNStart = frameN  # exact frame index
            array_image_2d.tStart = t  # local t and not account for scr refresh
            array_image_2d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_2d, 'tStartRefresh')  # time at next scr refresh
            array_image_2d.setAutoDraw(True)
        if array_image_2d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_2d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_2d.tStop = t  # not accounting for scr refresh
                array_image_2d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_2d, 'tStopRefresh')  # time at next scr refresh
                array_image_2d.setAutoDraw(False)
        
        # *array_image_3d* updates
        if array_image_3d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_3d.frameNStart = frameN  # exact frame index
            array_image_3d.tStart = t  # local t and not account for scr refresh
            array_image_3d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_3d, 'tStartRefresh')  # time at next scr refresh
            array_image_3d.setAutoDraw(True)
        if array_image_3d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_3d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_3d.tStop = t  # not accounting for scr refresh
                array_image_3d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_3d, 'tStopRefresh')  # time at next scr refresh
                array_image_3d.setAutoDraw(False)
        
        # *array_image_4d* updates
        if array_image_4d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_4d.frameNStart = frameN  # exact frame index
            array_image_4d.tStart = t  # local t and not account for scr refresh
            array_image_4d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_4d, 'tStartRefresh')  # time at next scr refresh
            array_image_4d.setAutoDraw(True)
        if array_image_4d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_4d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_4d.tStop = t  # not accounting for scr refresh
                array_image_4d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_4d, 'tStopRefresh')  # time at next scr refresh
                array_image_4d.setAutoDraw(False)
        
        # *array_image_5d* updates
        if array_image_5d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_5d.frameNStart = frameN  # exact frame index
            array_image_5d.tStart = t  # local t and not account for scr refresh
            array_image_5d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_5d, 'tStartRefresh')  # time at next scr refresh
            array_image_5d.setAutoDraw(True)
        if array_image_5d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_5d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_5d.tStop = t  # not accounting for scr refresh
                array_image_5d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_5d, 'tStopRefresh')  # time at next scr refresh
                array_image_5d.setAutoDraw(False)
        
        # *array_image_6d* updates
        if array_image_6d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_6d.frameNStart = frameN  # exact frame index
            array_image_6d.tStart = t  # local t and not account for scr refresh
            array_image_6d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_6d, 'tStartRefresh')  # time at next scr refresh
            array_image_6d.setAutoDraw(True)
        if array_image_6d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_6d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_6d.tStop = t  # not accounting for scr refresh
                array_image_6d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_6d, 'tStopRefresh')  # time at next scr refresh
                array_image_6d.setAutoDraw(False)
        
        # *array_image_7d* updates
        if array_image_7d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_7d.frameNStart = frameN  # exact frame index
            array_image_7d.tStart = t  # local t and not account for scr refresh
            array_image_7d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_7d, 'tStartRefresh')  # time at next scr refresh
            array_image_7d.setAutoDraw(True)
        if array_image_7d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_7d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_7d.tStop = t  # not accounting for scr refresh
                array_image_7d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_7d, 'tStopRefresh')  # time at next scr refresh
                array_image_7d.setAutoDraw(False)
        
        # *array_image_8d* updates
        if array_image_8d.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_8d.frameNStart = frameN  # exact frame index
            array_image_8d.tStart = t  # local t and not account for scr refresh
            array_image_8d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_8d, 'tStartRefresh')  # time at next scr refresh
            array_image_8d.setAutoDraw(True)
        if array_image_8d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_8d.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_8d.tStop = t  # not accounting for scr refresh
                array_image_8d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_8d, 'tStopRefresh')  # time at next scr refresh
                array_image_8d.setAutoDraw(False)
        
        # *polygon_26* updates
        if polygon_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_26.frameNStart = frameN  # exact frame index
            polygon_26.tStart = t  # local t and not account for scr refresh
            polygon_26.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_26, 'tStartRefresh')  # time at next scr refresh
            polygon_26.setAutoDraw(True)
        if polygon_26.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > polygon_26.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                polygon_26.tStop = t  # not accounting for scr refresh
                polygon_26.frameNStop = frameN  # exact frame index
                win.timeOnFlip(polygon_26, 'tStopRefresh')  # time at next scr refresh
                polygon_26.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_mem_displayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "practice_mem_display"-------
    for thisComponent in practice_mem_displayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    mem_display_to_cue_duration_s=mem_display_to_cue_times_practice[practice_trial_index]
    
    if retro_0_post_1_practice[practice_trial_index]:
        mem_display_to_test_array_duration_s=mem_display_to_cue_duration_s-.1
        routine_duration_s=mem_display_to_cue_duration_s+cue_duration_s
        response_time_s = routine_duration_s - cue_duration_s
    else:
        mem_display_to_test_array_duration_s=mem_display_to_cue_duration_s+cue_duration_s+retention_duration_s
        routine_duration_s=mem_display_to_cue_duration_s+cue_duration_s+retention_duration_s
        response_time_s = routine_duration_s
    
    shuffle(image_index_list_for_shuffling)
    change_from_index_practice=image_index_list_for_shuffling[0]  #create a change_from_index, even when there is no change. So you have somewhere to point the pointer.
    
    image_index_list_in_order=list(range(0,num_images_per_display))
    
    change_or_no_practice=change_0_1_practice[practice_trial_index]
    if change_or_no_practice:
        change_to_index_practice=num_images_per_display
        image_index_list_in_order[change_from_index_practice]=change_to_index_practice
    
    thisExp.addData('retention_interval_practice',mem_display_to_cue_duration_s)
    thisExp.addData('retro_0_post_1_practice',retro_0_post_1_practice[practice_trial_index])
    thisExp.addData('change_or_no_practice',change_or_no_practice)
    thisExp.addData('cue_location_practice',change_from_index_practice)
    
    thisExp.addData('mem_display_to_test_array_duration_s',mem_display_to_test_array_duration_s)
    thisExp.addData('mem_display_to_cue_duration_s',mem_display_to_cue_duration_s)
    thisExp.addData('routine_duration_s',routine_duration_s)
    # the Routine "practice_mem_display" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "retention_cue_and_test_intval_practice_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    Image_1=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[0]]]
    array_image_1f.setImage(Image_1, False)
    array_image_1f.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
    array_image_1f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_2=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[1]]]
    array_image_2f.setImage(Image_2, False)
    array_image_2f.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
    array_image_2f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_3=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[2]]]
    array_image_3f.setImage(Image_3, False)
    array_image_3f.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
    array_image_3f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_4=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[3]]]
    array_image_4f.setImage(Image_4, False)
    array_image_4f.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
    array_image_4f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_5=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[4]]]
    array_image_5f.setImage(Image_5, False)
    array_image_5f.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
    array_image_5f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_6=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[5]]]
    array_image_6f.setImage(Image_6, False)
    array_image_6f.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
    array_image_6f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_7=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[6]]]
    array_image_7f.setImage(Image_7, False)
    array_image_7f.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
    array_image_7f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_8=all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[7]]]
    array_image_8f.setImage(Image_8, False)
    array_image_8f.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
    array_image_8f.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    x_pointer_0=x_y_gains_per_angle[change_from_index_practice][0]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*x_scale
    y_pointer_0=x_y_gains_per_angle[change_from_index_practice][1]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*y_scale
    pointer_dot_0b.setPos((x_pointer_0, y_pointer_0), False)
    
    x_pointer_1=x_pointer_0*.66667
    y_pointer_1=y_pointer_0*.66667
    pointer_dot_1b.setPos((x_pointer_1, y_pointer_1), False)
    
    x_pointer_2=x_pointer_0*.33333
    y_pointer_2=y_pointer_0*.33333
    pointer_dot_2b.setPos((x_pointer_2, y_pointer_2), False)
    
    polygon_28.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
    array_image_1f.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
    array_image_1f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_1f.setImage(Image_1)
    array_image_2f.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
    array_image_2f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_2f.setImage(Image_2)
    array_image_3f.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
    array_image_3f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_3f.setImage(Image_3)
    array_image_4f.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
    array_image_4f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_4f.setImage(Image_4)
    array_image_5f.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
    array_image_5f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_5f.setImage(Image_5)
    array_image_6f.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
    array_image_6f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_6f.setImage(Image_6)
    array_image_7f.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
    array_image_7f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_7f.setImage(Image_7)
    array_image_8f.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
    array_image_8f.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_8f.setImage(Image_8)
    pointer_dot_0b.setPos((x_pointer_0,y_pointer_0))
    pointer_dot_0b.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    pointer_dot_1b.setPos((x_pointer_1,y_pointer_1))
    pointer_dot_1b.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    pointer_dot_2b.setPos((x_pointer_2,y_pointer_2))
    pointer_dot_2b.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    sd_practice_resp.keys = []
    sd_practice_resp.rt = []
    _sd_practice_resp_allKeys = []
    # keep track of which components have finished
    retention_cue_and_test_intval_practice_2Components = [polygon_28, array_image_1f, array_image_2f, array_image_3f, array_image_4f, array_image_5f, array_image_6f, array_image_7f, array_image_8f, pointer_dot_0b, pointer_dot_1b, pointer_dot_2b, sd_practice_resp]
    for thisComponent in retention_cue_and_test_intval_practice_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    retention_cue_and_test_intval_practice_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "retention_cue_and_test_intval_practice_2"-------
    while continueRoutine:
        # get current time
        t = retention_cue_and_test_intval_practice_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=retention_cue_and_test_intval_practice_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon_28* updates
        if polygon_28.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_28.frameNStart = frameN  # exact frame index
            polygon_28.tStart = t  # local t and not account for scr refresh
            polygon_28.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_28, 'tStartRefresh')  # time at next scr refresh
            polygon_28.setAutoDraw(True)
        
        # *array_image_1f* updates
        if array_image_1f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_1f.frameNStart = frameN  # exact frame index
            array_image_1f.tStart = t  # local t and not account for scr refresh
            array_image_1f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_1f, 'tStartRefresh')  # time at next scr refresh
            array_image_1f.setAutoDraw(True)
        
        # *array_image_2f* updates
        if array_image_2f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_2f.frameNStart = frameN  # exact frame index
            array_image_2f.tStart = t  # local t and not account for scr refresh
            array_image_2f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_2f, 'tStartRefresh')  # time at next scr refresh
            array_image_2f.setAutoDraw(True)
        
        # *array_image_3f* updates
        if array_image_3f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_3f.frameNStart = frameN  # exact frame index
            array_image_3f.tStart = t  # local t and not account for scr refresh
            array_image_3f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_3f, 'tStartRefresh')  # time at next scr refresh
            array_image_3f.setAutoDraw(True)
        
        # *array_image_4f* updates
        if array_image_4f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_4f.frameNStart = frameN  # exact frame index
            array_image_4f.tStart = t  # local t and not account for scr refresh
            array_image_4f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_4f, 'tStartRefresh')  # time at next scr refresh
            array_image_4f.setAutoDraw(True)
        
        # *array_image_5f* updates
        if array_image_5f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_5f.frameNStart = frameN  # exact frame index
            array_image_5f.tStart = t  # local t and not account for scr refresh
            array_image_5f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_5f, 'tStartRefresh')  # time at next scr refresh
            array_image_5f.setAutoDraw(True)
        
        # *array_image_6f* updates
        if array_image_6f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_6f.frameNStart = frameN  # exact frame index
            array_image_6f.tStart = t  # local t and not account for scr refresh
            array_image_6f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_6f, 'tStartRefresh')  # time at next scr refresh
            array_image_6f.setAutoDraw(True)
        
        # *array_image_7f* updates
        if array_image_7f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_7f.frameNStart = frameN  # exact frame index
            array_image_7f.tStart = t  # local t and not account for scr refresh
            array_image_7f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_7f, 'tStartRefresh')  # time at next scr refresh
            array_image_7f.setAutoDraw(True)
        
        # *array_image_8f* updates
        if array_image_8f.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_8f.frameNStart = frameN  # exact frame index
            array_image_8f.tStart = t  # local t and not account for scr refresh
            array_image_8f.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_8f, 'tStartRefresh')  # time at next scr refresh
            array_image_8f.setAutoDraw(True)
        
        # *pointer_dot_0b* updates
        if pointer_dot_0b.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_0b.frameNStart = frameN  # exact frame index
            pointer_dot_0b.tStart = t  # local t and not account for scr refresh
            pointer_dot_0b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_0b, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_0b.setAutoDraw(True)
        if pointer_dot_0b.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_0b.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_0b.tStop = t  # not accounting for scr refresh
                pointer_dot_0b.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_0b, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_0b.setAutoDraw(False)
        
        # *pointer_dot_1b* updates
        if pointer_dot_1b.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_1b.frameNStart = frameN  # exact frame index
            pointer_dot_1b.tStart = t  # local t and not account for scr refresh
            pointer_dot_1b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_1b, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_1b.setAutoDraw(True)
        if pointer_dot_1b.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_1b.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_1b.tStop = t  # not accounting for scr refresh
                pointer_dot_1b.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_1b, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_1b.setAutoDraw(False)
        
        # *pointer_dot_2b* updates
        if pointer_dot_2b.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_2b.frameNStart = frameN  # exact frame index
            pointer_dot_2b.tStart = t  # local t and not account for scr refresh
            pointer_dot_2b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_2b, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_2b.setAutoDraw(True)
        if pointer_dot_2b.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_2b.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_2b.tStop = t  # not accounting for scr refresh
                pointer_dot_2b.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_2b, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_2b.setAutoDraw(False)
        
        # *sd_practice_resp* updates
        waitOnFlip = False
        if sd_practice_resp.status == NOT_STARTED and tThisFlip >= response_time_s-frameTolerance:
            # keep track of start time/frame for later
            sd_practice_resp.frameNStart = frameN  # exact frame index
            sd_practice_resp.tStart = t  # local t and not account for scr refresh
            sd_practice_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sd_practice_resp, 'tStartRefresh')  # time at next scr refresh
            sd_practice_resp.status = STARTED
            # AllowedKeys looks like a variable named `response_string_pair`
            if not type(response_string_pair) in [list, tuple, np.ndarray]:
                if not isinstance(response_string_pair, str):
                    logging.error('AllowedKeys variable `response_string_pair` is not string- or list-like.')
                    core.quit()
                elif not ',' in response_string_pair:
                    response_string_pair = (response_string_pair,)
                else:
                    response_string_pair = eval(response_string_pair)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sd_practice_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sd_practice_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sd_practice_resp.status == STARTED and not waitOnFlip:
            theseKeys = sd_practice_resp.getKeys(keyList=list(response_string_pair), waitRelease=False)
            _sd_practice_resp_allKeys.extend(theseKeys)
            if len(_sd_practice_resp_allKeys):
                sd_practice_resp.keys = _sd_practice_resp_allKeys[-1].name  # just the last key pressed
                sd_practice_resp.rt = _sd_practice_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in retention_cue_and_test_intval_practice_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "retention_cue_and_test_intval_practice_2"-------
    for thisComponent in retention_cue_and_test_intval_practice_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('array_image_1f.started', array_image_1f.tStartRefresh)
    trials.addData('array_image_1f.stopped', array_image_1f.tStopRefresh)
    trials.addData('array_image_2f.started', array_image_2f.tStartRefresh)
    trials.addData('array_image_2f.stopped', array_image_2f.tStopRefresh)
    trials.addData('array_image_3f.started', array_image_3f.tStartRefresh)
    trials.addData('array_image_3f.stopped', array_image_3f.tStopRefresh)
    trials.addData('array_image_4f.started', array_image_4f.tStartRefresh)
    trials.addData('array_image_4f.stopped', array_image_4f.tStopRefresh)
    trials.addData('array_image_5f.started', array_image_5f.tStartRefresh)
    trials.addData('array_image_5f.stopped', array_image_5f.tStopRefresh)
    trials.addData('array_image_6f.started', array_image_6f.tStartRefresh)
    trials.addData('array_image_6f.stopped', array_image_6f.tStopRefresh)
    trials.addData('array_image_7f.started', array_image_7f.tStartRefresh)
    trials.addData('array_image_7f.stopped', array_image_7f.tStopRefresh)
    trials.addData('array_image_8f.started', array_image_8f.tStartRefresh)
    trials.addData('array_image_8f.stopped', array_image_8f.tStopRefresh)
    trials.addData('pointer_dot_0b.started', pointer_dot_0b.tStartRefresh)
    trials.addData('pointer_dot_0b.stopped', pointer_dot_0b.tStopRefresh)
    trials.addData('pointer_dot_1b.started', pointer_dot_1b.tStartRefresh)
    trials.addData('pointer_dot_1b.stopped', pointer_dot_1b.tStopRefresh)
    trials.addData('pointer_dot_2b.started', pointer_dot_2b.tStartRefresh)
    trials.addData('pointer_dot_2b.stopped', pointer_dot_2b.tStopRefresh)
    # check responses
    if sd_practice_resp.keys in ['', [], None]:  # No response was made
        sd_practice_resp.keys = None
    trials.addData('sd_practice_resp.keys',sd_practice_resp.keys)
    if sd_practice_resp.keys != None:  # we had a response
        trials.addData('sd_practice_resp.rt', sd_practice_resp.rt)
    trials.addData('sd_practice_resp.started', sd_practice_resp.tStartRefresh)
    trials.addData('sd_practice_resp.stopped', sd_practice_resp.tStopRefresh)
    # the Routine "retention_cue_and_test_intval_practice_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "confidence_rating_practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    if practice_trial_index==0:
        confidence_reminder_text='Excellent. At this point, right after each Same/Different choice, you’re expected to indicate how confident you are that the choice you just made was correct.\n\nPlease use your left hand and use your ring finger if you are certain (\'z\'), your middle finger if you have some doubt (\'x\'), or your index finger if it was just a guess (\'c\'). See image below and make your response.'
    else:
        confidence_reminder_text='Don\'t forget: after each Same/Different choice, you should indicate how confident you are in the choice you just made.\n\nPlease use your left hand to press either ring finger (I\'m certain, \'z\'), middle finger (I have some doubt, \'x\'), or index finger (it was just a guess, \'c\'.)'
    text_10.setText(confidence_reminder_text,False)
    text_10.setText(confidence_reminder_text)
    conf_practice_resp.keys = []
    conf_practice_resp.rt = []
    _conf_practice_resp_allKeys = []
    # keep track of which components have finished
    confidence_rating_practiceComponents = [text_10, conf_practice_resp, image]
    for thisComponent in confidence_rating_practiceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    confidence_rating_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "confidence_rating_practice"-------
    while continueRoutine:
        # get current time
        t = confidence_rating_practiceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=confidence_rating_practiceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_10* updates
        if text_10.status == NOT_STARTED and tThisFlip >= confidence_onset_time_s-frameTolerance:
            # keep track of start time/frame for later
            text_10.frameNStart = frameN  # exact frame index
            text_10.tStart = t  # local t and not account for scr refresh
            text_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
            text_10.setAutoDraw(True)
        
        # *conf_practice_resp* updates
        waitOnFlip = False
        if conf_practice_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            conf_practice_resp.frameNStart = frameN  # exact frame index
            conf_practice_resp.tStart = t  # local t and not account for scr refresh
            conf_practice_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(conf_practice_resp, 'tStartRefresh')  # time at next scr refresh
            conf_practice_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(conf_practice_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(conf_practice_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if conf_practice_resp.status == STARTED and not waitOnFlip:
            theseKeys = conf_practice_resp.getKeys(keyList=['z', 'x', 'c'], waitRelease=False)
            _conf_practice_resp_allKeys.extend(theseKeys)
            if len(_conf_practice_resp_allKeys):
                conf_practice_resp.keys = _conf_practice_resp_allKeys[-1].name  # just the last key pressed
                conf_practice_resp.rt = _conf_practice_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *image* updates
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            image.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidence_rating_practiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "confidence_rating_practice"-------
    for thisComponent in confidence_rating_practiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_trial_index = practice_trial_index + 1
    trials.addData('text_10.started', text_10.tStartRefresh)
    trials.addData('text_10.stopped', text_10.tStopRefresh)
    # check responses
    if conf_practice_resp.keys in ['', [], None]:  # No response was made
        conf_practice_resp.keys = None
    trials.addData('conf_practice_resp.keys',conf_practice_resp.keys)
    if conf_practice_resp.keys != None:  # we had a response
        trials.addData('conf_practice_resp.rt', conf_practice_resp.rt)
    trials.addData('conf_practice_resp.started', conf_practice_resp.tStartRefresh)
    trials.addData('conf_practice_resp.stopped', conf_practice_resp.tStopRefresh)
    trials.addData('image.started', image.tStartRefresh)
    trials.addData('image.stopped', image.tStopRefresh)
    # the Routine "confidence_rating_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed num_practice_trials repeats of 'trials'


# ------Prepare to start Routine "instructions_specific_training"-------
continueRoutine = True
# update component parameters for each repeat
static_instruction_42.setText(text_with_or_without_eye_tracking)
polygon_36.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
static_key1_20.keys = []
static_key1_20.rt = []
_static_key1_20_allKeys = []
# keep track of which components have finished
instructions_specific_trainingComponents = [static_instruction_42, polygon_36, static_instruction_43, static_key1_20]
for thisComponent in instructions_specific_trainingComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructions_specific_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions_specific_training"-------
while continueRoutine:
    # get current time
    t = instructions_specific_trainingClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructions_specific_trainingClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *static_instruction_42* updates
    if static_instruction_42.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_42.frameNStart = frameN  # exact frame index
        static_instruction_42.tStart = t  # local t and not account for scr refresh
        static_instruction_42.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_42, 'tStartRefresh')  # time at next scr refresh
        static_instruction_42.setAutoDraw(True)
    
    # *polygon_36* updates
    if polygon_36.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_36.frameNStart = frameN  # exact frame index
        polygon_36.tStart = t  # local t and not account for scr refresh
        polygon_36.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_36, 'tStartRefresh')  # time at next scr refresh
        polygon_36.setAutoDraw(True)
    
    # *static_instruction_43* updates
    if static_instruction_43.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_instruction_43.frameNStart = frameN  # exact frame index
        static_instruction_43.tStart = t  # local t and not account for scr refresh
        static_instruction_43.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_instruction_43, 'tStartRefresh')  # time at next scr refresh
        static_instruction_43.setAutoDraw(True)
    
    # *static_key1_20* updates
    waitOnFlip = False
    if static_key1_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        static_key1_20.frameNStart = frameN  # exact frame index
        static_key1_20.tStart = t  # local t and not account for scr refresh
        static_key1_20.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(static_key1_20, 'tStartRefresh')  # time at next scr refresh
        static_key1_20.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(static_key1_20.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(static_key1_20.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if static_key1_20.status == STARTED and not waitOnFlip:
        theseKeys = static_key1_20.getKeys(keyList=None, waitRelease=False)
        _static_key1_20_allKeys.extend(theseKeys)
        if len(_static_key1_20_allKeys):
            static_key1_20.keys = _static_key1_20_allKeys[-1].name  # just the last key pressed
            static_key1_20.rt = _static_key1_20_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructions_specific_trainingComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions_specific_training"-------
for thisComponent in instructions_specific_trainingComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instructions_specific_training" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials_4 = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(permutation_data_file_string_practice),
    seed=None, name='trials_4')
thisExp.addLoop(trials_4)  # add the loop to the experiment
thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
if thisTrial_4 != None:
    for paramName in thisTrial_4:
        exec('{} = thisTrial_4[paramName]'.format(paramName))

for thisTrial_4 in trials_4:
    currentLoop = trials_4
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
    if thisTrial_4 != None:
        for paramName in thisTrial_4:
            exec('{} = thisTrial_4[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "load_training_trials"-------
    continueRoutine = True
    # update component parameters for each repeat
    training_trial_image_list.append([Used_image_1,Used_image_2,Used_image_3,Used_image_4,Used_image_5,Used_image_6,Used_image_7,Used_image_8,Unused_image_1,Unused_image_2])
    if (training_trial_load_number+1)<=max_num_training_trials/2:
        retro_0_post_1_training.append(0)
        change_0_1_training.append(0)
    else:
        retro_0_post_1_training.append(1)
        change_0_1_training.append(1)
    
    # keep track of which components have finished
    load_training_trialsComponents = []
    for thisComponent in load_training_trialsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    load_training_trialsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "load_training_trials"-------
    while continueRoutine:
        # get current time
        t = load_training_trialsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=load_training_trialsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in load_training_trialsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "load_training_trials"-------
    for thisComponent in load_training_trialsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training_trial_load_number=training_trial_load_number+1
    
    if training_trial_load_number>=max_num_training_trials:
        trials_4.finished = True
        shuffle(retro_0_post_1_training)
        shuffle(change_0_1_training)
    
    confidence_onset_time_s=0.
    confidence_image_onset_time_s=0.
    # the Routine "load_training_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 1 repeats of 'trials_4'


# set up handler to look after randomisation of conditions etc
trials_10 = data.TrialHandler(nReps=max_num_training_trials, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_10')
thisExp.addLoop(trials_10)  # add the loop to the experiment
thisTrial_10 = trials_10.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_10.rgb)
if thisTrial_10 != None:
    for paramName in thisTrial_10:
        exec('{} = thisTrial_10[paramName]'.format(paramName))

for thisTrial_10 in trials_10:
    currentLoop = trials_10
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_10.rgb)
    if thisTrial_10 != None:
        for paramName in thisTrial_10:
            exec('{} = thisTrial_10[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "get_ready_training_blockstart"-------
    continueRoutine = True
    # update component parameters for each repeat
    if (training_trial_index>0) and (mod(training_trial_index,repeat_intructions_every) == 0):
        block_break_text="You are doing quite well! Don't forget this tip from our best performers:\n\nAs soon as the red dotted line points to a spot, immediately try to recall what was in that spot in the first group of pictures. So if the dotted line appears before the second group of pictures, don't wait for the second group of pictures to appear before recalling what it was.\n\nPlease press any key to continue your training." 
        continueRoutine=True
    else:
        block_break_text = ""
        continueRoutine=False
        
    static_instruction_44.setText(block_break_text,False)
    
    thisExp.addData('training_trial_index',training_trial_index)
    
    if short_block_done:
        thisExp.addData('training_trial_index_long_this_block',training_trial_index_long_this_block)
    static_instruction_44.setText(block_break_text)
    static_key1_21.keys = []
    static_key1_21.rt = []
    _static_key1_21_allKeys = []
    # keep track of which components have finished
    get_ready_training_blockstartComponents = [static_instruction_44, static_key1_21]
    for thisComponent in get_ready_training_blockstartComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    get_ready_training_blockstartClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "get_ready_training_blockstart"-------
    while continueRoutine:
        # get current time
        t = get_ready_training_blockstartClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=get_ready_training_blockstartClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *static_instruction_44* updates
        if static_instruction_44.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            static_instruction_44.frameNStart = frameN  # exact frame index
            static_instruction_44.tStart = t  # local t and not account for scr refresh
            static_instruction_44.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(static_instruction_44, 'tStartRefresh')  # time at next scr refresh
            static_instruction_44.setAutoDraw(True)
        
        # *static_key1_21* updates
        waitOnFlip = False
        if static_key1_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            static_key1_21.frameNStart = frameN  # exact frame index
            static_key1_21.tStart = t  # local t and not account for scr refresh
            static_key1_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(static_key1_21, 'tStartRefresh')  # time at next scr refresh
            static_key1_21.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(static_key1_21.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(static_key1_21.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if static_key1_21.status == STARTED and not waitOnFlip:
            theseKeys = static_key1_21.getKeys(keyList=None, waitRelease=False)
            _static_key1_21_allKeys.extend(theseKeys)
            if len(_static_key1_21_allKeys):
                static_key1_21.keys = _static_key1_21_allKeys[-1].name  # just the last key pressed
                static_key1_21.rt = _static_key1_21_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in get_ready_training_blockstartComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "get_ready_training_blockstart"-------
    for thisComponent in get_ready_training_blockstartComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "get_ready_training_blockstart" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "training_mem_display"-------
    continueRoutine = True
    # update component parameters for each repeat
    Image_1=all_pic_names_training[training_trial_image_list[training_trial_index][0]]
    array_image_1l.setImage(Image_1, False)
    array_image_1l.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
    array_image_1l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_2=all_pic_names_training[training_trial_image_list[training_trial_index][1]]
    array_image_2l.setImage(Image_2, False)
    array_image_2l.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
    array_image_2l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_3=all_pic_names_training[training_trial_image_list[training_trial_index][2]]
    array_image_3l.setImage(Image_3, False)
    array_image_3l.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
    array_image_3l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_4=all_pic_names_training[training_trial_image_list[training_trial_index][3]]
    array_image_4l.setImage(Image_4, False)
    array_image_4l.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
    array_image_4l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_5=all_pic_names_training[training_trial_image_list[training_trial_index][4]]
    array_image_5l.setImage(Image_5, False)
    array_image_5l.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
    array_image_5l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_6=all_pic_names_training[training_trial_image_list[training_trial_index][5]]
    array_image_6l.setImage(Image_6, False)
    array_image_6l.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
    array_image_6l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_7=all_pic_names_training[training_trial_image_list[training_trial_index][6]]
    array_image_7l.setImage(Image_7, False)
    array_image_7l.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
    array_image_7l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_8=all_pic_names_training[training_trial_image_list[training_trial_index][7]]
    array_image_8l.setImage(Image_8, False)
    array_image_8l.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
    array_image_8l.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    thisExp.addData('memDisplayOnTimeTraining',timer.getTime())
    
    #storing variables specifying trial
    for image_index in range(num_images_per_display):
        thisExp.addData('mem_image_'+str(image_index)+'_training',training_trial_image_list[training_trial_index][image_index])
    array_image_1l.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
    array_image_1l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_1l.setImage(Image_1)
    array_image_2l.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
    array_image_2l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_2l.setImage(Image_2)
    array_image_3l.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
    array_image_3l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_3l.setImage(Image_3)
    array_image_4l.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
    array_image_4l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_4l.setImage(Image_4)
    array_image_5l.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
    array_image_5l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_5l.setImage(Image_5)
    array_image_6l.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
    array_image_6l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_6l.setImage(Image_6)
    array_image_7l.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
    array_image_7l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_7l.setImage(Image_7)
    array_image_8l.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
    array_image_8l.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_8l.setImage(Image_8)
    polygon_44.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
    # keep track of which components have finished
    training_mem_displayComponents = [array_image_1l, array_image_2l, array_image_3l, array_image_4l, array_image_5l, array_image_6l, array_image_7l, array_image_8l, polygon_44]
    for thisComponent in training_mem_displayComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    training_mem_displayClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "training_mem_display"-------
    while continueRoutine:
        # get current time
        t = training_mem_displayClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=training_mem_displayClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *array_image_1l* updates
        if array_image_1l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_1l.frameNStart = frameN  # exact frame index
            array_image_1l.tStart = t  # local t and not account for scr refresh
            array_image_1l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_1l, 'tStartRefresh')  # time at next scr refresh
            array_image_1l.setAutoDraw(True)
        if array_image_1l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_1l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_1l.tStop = t  # not accounting for scr refresh
                array_image_1l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_1l, 'tStopRefresh')  # time at next scr refresh
                array_image_1l.setAutoDraw(False)
        
        # *array_image_2l* updates
        if array_image_2l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_2l.frameNStart = frameN  # exact frame index
            array_image_2l.tStart = t  # local t and not account for scr refresh
            array_image_2l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_2l, 'tStartRefresh')  # time at next scr refresh
            array_image_2l.setAutoDraw(True)
        if array_image_2l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_2l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_2l.tStop = t  # not accounting for scr refresh
                array_image_2l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_2l, 'tStopRefresh')  # time at next scr refresh
                array_image_2l.setAutoDraw(False)
        
        # *array_image_3l* updates
        if array_image_3l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_3l.frameNStart = frameN  # exact frame index
            array_image_3l.tStart = t  # local t and not account for scr refresh
            array_image_3l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_3l, 'tStartRefresh')  # time at next scr refresh
            array_image_3l.setAutoDraw(True)
        if array_image_3l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_3l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_3l.tStop = t  # not accounting for scr refresh
                array_image_3l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_3l, 'tStopRefresh')  # time at next scr refresh
                array_image_3l.setAutoDraw(False)
        
        # *array_image_4l* updates
        if array_image_4l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_4l.frameNStart = frameN  # exact frame index
            array_image_4l.tStart = t  # local t and not account for scr refresh
            array_image_4l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_4l, 'tStartRefresh')  # time at next scr refresh
            array_image_4l.setAutoDraw(True)
        if array_image_4l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_4l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_4l.tStop = t  # not accounting for scr refresh
                array_image_4l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_4l, 'tStopRefresh')  # time at next scr refresh
                array_image_4l.setAutoDraw(False)
        
        # *array_image_5l* updates
        if array_image_5l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_5l.frameNStart = frameN  # exact frame index
            array_image_5l.tStart = t  # local t and not account for scr refresh
            array_image_5l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_5l, 'tStartRefresh')  # time at next scr refresh
            array_image_5l.setAutoDraw(True)
        if array_image_5l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_5l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_5l.tStop = t  # not accounting for scr refresh
                array_image_5l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_5l, 'tStopRefresh')  # time at next scr refresh
                array_image_5l.setAutoDraw(False)
        
        # *array_image_6l* updates
        if array_image_6l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_6l.frameNStart = frameN  # exact frame index
            array_image_6l.tStart = t  # local t and not account for scr refresh
            array_image_6l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_6l, 'tStartRefresh')  # time at next scr refresh
            array_image_6l.setAutoDraw(True)
        if array_image_6l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_6l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_6l.tStop = t  # not accounting for scr refresh
                array_image_6l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_6l, 'tStopRefresh')  # time at next scr refresh
                array_image_6l.setAutoDraw(False)
        
        # *array_image_7l* updates
        if array_image_7l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_7l.frameNStart = frameN  # exact frame index
            array_image_7l.tStart = t  # local t and not account for scr refresh
            array_image_7l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_7l, 'tStartRefresh')  # time at next scr refresh
            array_image_7l.setAutoDraw(True)
        if array_image_7l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_7l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_7l.tStop = t  # not accounting for scr refresh
                array_image_7l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_7l, 'tStopRefresh')  # time at next scr refresh
                array_image_7l.setAutoDraw(False)
        
        # *array_image_8l* updates
        if array_image_8l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            array_image_8l.frameNStart = frameN  # exact frame index
            array_image_8l.tStart = t  # local t and not account for scr refresh
            array_image_8l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_8l, 'tStartRefresh')  # time at next scr refresh
            array_image_8l.setAutoDraw(True)
        if array_image_8l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > array_image_8l.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                array_image_8l.tStop = t  # not accounting for scr refresh
                array_image_8l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(array_image_8l, 'tStopRefresh')  # time at next scr refresh
                array_image_8l.setAutoDraw(False)
        
        # *polygon_44* updates
        if polygon_44.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_44.frameNStart = frameN  # exact frame index
            polygon_44.tStart = t  # local t and not account for scr refresh
            polygon_44.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_44, 'tStartRefresh')  # time at next scr refresh
            polygon_44.setAutoDraw(True)
        if polygon_44.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > polygon_44.tStartRefresh + mem_display_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                polygon_44.tStop = t  # not accounting for scr refresh
                polygon_44.frameNStop = frameN  # exact frame index
                win.timeOnFlip(polygon_44, 'tStopRefresh')  # time at next scr refresh
                polygon_44.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_mem_displayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "training_mem_display"-------
    for thisComponent in training_mem_displayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    if not(short_block_done):
    
        mem_display_to_cue_duration_s=mem_display_to_cue_time_initial_training_s
        
    elif retro_0_post_1_training[training_trial_index]:
        
        shuffle(mem_display_to_cue_time_post_options_long_training_s)
        mem_display_to_cue_duration_s = mem_display_to_cue_time_post_options_long_training_s[0]
        
    else:
        
        mem_display_to_cue_duration_s=mem_display_to_cue_time_retro_long_training_s
            
    if retro_0_post_1_training[training_trial_index]:
        mem_display_to_test_array_duration_s=mem_display_to_cue_duration_s-.1
        routine_duration_s=mem_display_to_cue_duration_s+cue_duration_s
        response_time_s = routine_duration_s - cue_duration_s
    else:
        mem_display_to_test_array_duration_s=mem_display_to_cue_duration_s+cue_duration_s+retention_duration_s
        routine_duration_s=mem_display_to_cue_duration_s+cue_duration_s+retention_duration_s
        response_time_s = routine_duration_s
    
    shuffle(image_index_list_for_shuffling)
    change_from_index_training=image_index_list_for_shuffling[0]  #create a change_from_index, even when there is no change. So you have somewhere to point the pointer.
    
    image_index_list_in_order=list(range(0,num_images_per_display))
    
    change_or_no_training=change_0_1_training[training_trial_index]
    if change_or_no_training:
        change_to_index_training=num_images_per_display
        image_index_list_in_order[change_from_index_training]=change_to_index_training
    
    thisExp.addData('retro_0_post_1_training',retro_0_post_1_training[training_trial_index])
    thisExp.addData('change_or_no_training',change_or_no_training)
    thisExp.addData('cue_location_training',change_from_index_training)
    
    thisExp.addData('mem_display_to_test_array_duration_s',mem_display_to_test_array_duration_s)
    thisExp.addData('mem_display_to_cue_duration_s',mem_display_to_cue_duration_s)
    thisExp.addData('routine_duration_s',routine_duration_s)
    trials_10.addData('array_image_1l.started', array_image_1l.tStartRefresh)
    trials_10.addData('array_image_1l.stopped', array_image_1l.tStopRefresh)
    # the Routine "training_mem_display" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "retention_cue_and_test_intval_training"-------
    continueRoutine = True
    # update component parameters for each repeat
    Image_1=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[0]]]
    array_image_1m.setImage(Image_1, False)
    array_image_1m.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale), False)
    array_image_1m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_2=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[1]]]
    array_image_2m.setImage(Image_2, False)
    array_image_2m.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale), False)
    array_image_2m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_3=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[2]]]
    array_image_3m.setImage(Image_3, False)
    array_image_3m.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale), False)
    array_image_3m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_4=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[3]]]
    array_image_4m.setImage(Image_4, False)
    array_image_4m.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale), False)
    array_image_4m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_5=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[4]]]
    array_image_5m.setImage(Image_5, False)
    array_image_5m.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale), False)
    array_image_5m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_6=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[5]]]
    array_image_6m.setImage(Image_6, False)
    array_image_6m.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale), False)
    array_image_6m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_7=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[6]]]
    array_image_7m.setImage(Image_7, False)
    array_image_7m.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale), False)
    array_image_7m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    Image_8=all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[7]]]
    array_image_8m.setImage(Image_8, False)
    array_image_8m.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale), False)
    array_image_8m.setSize((image_side_cm*x_scale,image_side_cm*y_scale),False)
    
    x_pointer_0=x_y_gains_per_angle[change_from_index_training][0]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*x_scale
    y_pointer_0=x_y_gains_per_angle[change_from_index_training][1]*pointer_length_relative_to_image_eccentricity*image_eccentr_cm*y_scale
    pointer_dot_0d.setPos((x_pointer_0, y_pointer_0), False)
    
    x_pointer_1=x_pointer_0*.66667
    y_pointer_1=y_pointer_0*.66667
    pointer_dot_1d.setPos((x_pointer_1, y_pointer_1), False)
    
    x_pointer_2=x_pointer_0*.33333
    y_pointer_2=y_pointer_0*.33333
    pointer_dot_2d.setPos((x_pointer_2, y_pointer_2), False)
    
    #storing variables specifying trial
    for image_index in range(8):
        thisExp.addData('test_image_'+str(image_index)+'_training',training_trial_image_list[training_trial_index][image_index_list_in_order[image_index]])
    polygon_46.setSize((dot_diam_dva*x_scale, dot_diam_dva*y_scale))
    array_image_1m.setPos((x_y_gains_per_angle[0][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[0][1]*image_eccentr_cm*y_scale))
    array_image_1m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_1m.setImage(Image_1)
    array_image_2m.setPos((x_y_gains_per_angle[1][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[1][1]*image_eccentr_cm*y_scale))
    array_image_2m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_2m.setImage(Image_2)
    array_image_3m.setPos((x_y_gains_per_angle[2][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[2][1]*image_eccentr_cm*y_scale))
    array_image_3m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_3m.setImage(Image_3)
    array_image_4m.setPos((x_y_gains_per_angle[3][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[3][1]*image_eccentr_cm*y_scale))
    array_image_4m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_4m.setImage(Image_4)
    array_image_5m.setPos((x_y_gains_per_angle[4][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[4][1]*image_eccentr_cm*y_scale))
    array_image_5m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_5m.setImage(Image_5)
    array_image_6m.setPos((x_y_gains_per_angle[5][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[5][1]*image_eccentr_cm*y_scale))
    array_image_6m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_6m.setImage(Image_6)
    array_image_7m.setPos((x_y_gains_per_angle[6][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[6][1]*image_eccentr_cm*y_scale))
    array_image_7m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_7m.setImage(Image_7)
    array_image_8m.setPos((x_y_gains_per_angle[7][0]*image_eccentr_cm*x_scale, x_y_gains_per_angle[7][1]*image_eccentr_cm*y_scale))
    array_image_8m.setSize((image_side_cm*x_scale,image_side_cm*y_scale))
    array_image_8m.setImage(Image_8)
    pointer_dot_0d.setPos((x_pointer_0,y_pointer_0))
    pointer_dot_0d.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    pointer_dot_1d.setPos((x_pointer_1,y_pointer_1))
    pointer_dot_1d.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    pointer_dot_2d.setPos((x_pointer_2,y_pointer_2))
    pointer_dot_2d.setSize((dot_diam_dva*x_scale/2., dot_diam_dva*y_scale/2.))
    sd_training_resp.keys = []
    sd_training_resp.rt = []
    _sd_training_resp_allKeys = []
    # keep track of which components have finished
    retention_cue_and_test_intval_trainingComponents = [polygon_46, array_image_1m, array_image_2m, array_image_3m, array_image_4m, array_image_5m, array_image_6m, array_image_7m, array_image_8m, pointer_dot_0d, pointer_dot_1d, pointer_dot_2d, reminder_training, sd_training_resp]
    for thisComponent in retention_cue_and_test_intval_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    retention_cue_and_test_intval_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "retention_cue_and_test_intval_training"-------
    while continueRoutine:
        # get current time
        t = retention_cue_and_test_intval_trainingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=retention_cue_and_test_intval_trainingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon_46* updates
        if polygon_46.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_46.frameNStart = frameN  # exact frame index
            polygon_46.tStart = t  # local t and not account for scr refresh
            polygon_46.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_46, 'tStartRefresh')  # time at next scr refresh
            polygon_46.setAutoDraw(True)
        
        # *array_image_1m* updates
        if array_image_1m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_1m.frameNStart = frameN  # exact frame index
            array_image_1m.tStart = t  # local t and not account for scr refresh
            array_image_1m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_1m, 'tStartRefresh')  # time at next scr refresh
            array_image_1m.setAutoDraw(True)
        
        # *array_image_2m* updates
        if array_image_2m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_2m.frameNStart = frameN  # exact frame index
            array_image_2m.tStart = t  # local t and not account for scr refresh
            array_image_2m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_2m, 'tStartRefresh')  # time at next scr refresh
            array_image_2m.setAutoDraw(True)
        
        # *array_image_3m* updates
        if array_image_3m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_3m.frameNStart = frameN  # exact frame index
            array_image_3m.tStart = t  # local t and not account for scr refresh
            array_image_3m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_3m, 'tStartRefresh')  # time at next scr refresh
            array_image_3m.setAutoDraw(True)
        
        # *array_image_4m* updates
        if array_image_4m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_4m.frameNStart = frameN  # exact frame index
            array_image_4m.tStart = t  # local t and not account for scr refresh
            array_image_4m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_4m, 'tStartRefresh')  # time at next scr refresh
            array_image_4m.setAutoDraw(True)
        
        # *array_image_5m* updates
        if array_image_5m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_5m.frameNStart = frameN  # exact frame index
            array_image_5m.tStart = t  # local t and not account for scr refresh
            array_image_5m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_5m, 'tStartRefresh')  # time at next scr refresh
            array_image_5m.setAutoDraw(True)
        
        # *array_image_6m* updates
        if array_image_6m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_6m.frameNStart = frameN  # exact frame index
            array_image_6m.tStart = t  # local t and not account for scr refresh
            array_image_6m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_6m, 'tStartRefresh')  # time at next scr refresh
            array_image_6m.setAutoDraw(True)
        
        # *array_image_7m* updates
        if array_image_7m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_7m.frameNStart = frameN  # exact frame index
            array_image_7m.tStart = t  # local t and not account for scr refresh
            array_image_7m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_7m, 'tStartRefresh')  # time at next scr refresh
            array_image_7m.setAutoDraw(True)
        
        # *array_image_8m* updates
        if array_image_8m.status == NOT_STARTED and tThisFlip >= mem_display_to_test_array_duration_s-frameTolerance:
            # keep track of start time/frame for later
            array_image_8m.frameNStart = frameN  # exact frame index
            array_image_8m.tStart = t  # local t and not account for scr refresh
            array_image_8m.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(array_image_8m, 'tStartRefresh')  # time at next scr refresh
            array_image_8m.setAutoDraw(True)
        
        # *pointer_dot_0d* updates
        if pointer_dot_0d.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_0d.frameNStart = frameN  # exact frame index
            pointer_dot_0d.tStart = t  # local t and not account for scr refresh
            pointer_dot_0d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_0d, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_0d.setAutoDraw(True)
        if pointer_dot_0d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_0d.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_0d.tStop = t  # not accounting for scr refresh
                pointer_dot_0d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_0d, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_0d.setAutoDraw(False)
        
        # *pointer_dot_1d* updates
        if pointer_dot_1d.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_1d.frameNStart = frameN  # exact frame index
            pointer_dot_1d.tStart = t  # local t and not account for scr refresh
            pointer_dot_1d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_1d, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_1d.setAutoDraw(True)
        if pointer_dot_1d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_1d.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_1d.tStop = t  # not accounting for scr refresh
                pointer_dot_1d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_1d, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_1d.setAutoDraw(False)
        
        # *pointer_dot_2d* updates
        if pointer_dot_2d.status == NOT_STARTED and tThisFlip >= mem_display_to_cue_duration_s-frameTolerance:
            # keep track of start time/frame for later
            pointer_dot_2d.frameNStart = frameN  # exact frame index
            pointer_dot_2d.tStart = t  # local t and not account for scr refresh
            pointer_dot_2d.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pointer_dot_2d, 'tStartRefresh')  # time at next scr refresh
            pointer_dot_2d.setAutoDraw(True)
        if pointer_dot_2d.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pointer_dot_2d.tStartRefresh + cue_duration_s-frameTolerance:
                # keep track of stop time/frame for later
                pointer_dot_2d.tStop = t  # not accounting for scr refresh
                pointer_dot_2d.frameNStop = frameN  # exact frame index
                win.timeOnFlip(pointer_dot_2d, 'tStopRefresh')  # time at next scr refresh
                pointer_dot_2d.setAutoDraw(False)
        
        # *reminder_training* updates
        if reminder_training.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
            # keep track of start time/frame for later
            reminder_training.frameNStart = frameN  # exact frame index
            reminder_training.tStart = t  # local t and not account for scr refresh
            reminder_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reminder_training, 'tStartRefresh')  # time at next scr refresh
            reminder_training.setAutoDraw(True)
        
        # *sd_training_resp* updates
        waitOnFlip = False
        if sd_training_resp.status == NOT_STARTED and tThisFlip >= response_time_s-frameTolerance:
            # keep track of start time/frame for later
            sd_training_resp.frameNStart = frameN  # exact frame index
            sd_training_resp.tStart = t  # local t and not account for scr refresh
            sd_training_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sd_training_resp, 'tStartRefresh')  # time at next scr refresh
            sd_training_resp.status = STARTED
            # AllowedKeys looks like a variable named `response_string_pair`
            if not type(response_string_pair) in [list, tuple, np.ndarray]:
                if not isinstance(response_string_pair, str):
                    logging.error('AllowedKeys variable `response_string_pair` is not string- or list-like.')
                    core.quit()
                elif not ',' in response_string_pair:
                    response_string_pair = (response_string_pair,)
                else:
                    response_string_pair = eval(response_string_pair)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sd_training_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sd_training_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sd_training_resp.status == STARTED and not waitOnFlip:
            theseKeys = sd_training_resp.getKeys(keyList=list(response_string_pair), waitRelease=False)
            _sd_training_resp_allKeys.extend(theseKeys)
            if len(_sd_training_resp_allKeys):
                sd_training_resp.keys = _sd_training_resp_allKeys[-1].name  # just the last key pressed
                sd_training_resp.rt = _sd_training_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        if len(sd_training_resp.keys):
            if(short_block_done == 0):
                if(retro_0_post_1_training[training_trial_index] == 0):
                    if (change_or_no_training == 0) and (sd_training_resp.keys == response_string_pair[0]):
                        consecutive_correct_retro=consecutive_correct_retro+1
                    else:
                        if (change_or_no_training == 1) and (sd_training_resp.keys == response_string_pair[1]):
                            consecutive_correct_retro=consecutive_correct_retro+1
                        else:
                            consecutive_correct_retro=0
                            
                thisExp.addData('consecutive_correct_retro_short',consecutive_correct_retro)
                
            elif(short_block_done == 1):
                
                if(retro_0_post_1_training[training_trial_index] == 0):
                    retro_trials_long_this_block = retro_trials_long_this_block + 1
                    if (change_or_no_training == 0) and (sd_training_resp.keys == response_string_pair[0]):
                        correct_retro_long_this_block=correct_retro_long_this_block+1
                    elif (change_or_no_training == 1) and (sd_training_resp.keys == response_string_pair[1]):
                        correct_retro_long_this_block=correct_retro_long_this_block+1
        
                thisExp.addData('correct_retro_long_this_block',correct_retro_long_this_block)
                thisExp.addData('retro_trials_long_this_block',retro_trials_long_this_block)
               
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in retention_cue_and_test_intval_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "retention_cue_and_test_intval_training"-------
    for thisComponent in retention_cue_and_test_intval_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_10.addData('array_image_1m.started', array_image_1m.tStartRefresh)
    trials_10.addData('array_image_1m.stopped', array_image_1m.tStopRefresh)
    trials_10.addData('array_image_2m.started', array_image_2m.tStartRefresh)
    trials_10.addData('array_image_2m.stopped', array_image_2m.tStopRefresh)
    trials_10.addData('array_image_3m.started', array_image_3m.tStartRefresh)
    trials_10.addData('array_image_3m.stopped', array_image_3m.tStopRefresh)
    trials_10.addData('array_image_4m.started', array_image_4m.tStartRefresh)
    trials_10.addData('array_image_4m.stopped', array_image_4m.tStopRefresh)
    trials_10.addData('array_image_5m.started', array_image_5m.tStartRefresh)
    trials_10.addData('array_image_5m.stopped', array_image_5m.tStopRefresh)
    trials_10.addData('array_image_6m.started', array_image_6m.tStartRefresh)
    trials_10.addData('array_image_6m.stopped', array_image_6m.tStopRefresh)
    trials_10.addData('array_image_7m.started', array_image_7m.tStartRefresh)
    trials_10.addData('array_image_7m.stopped', array_image_7m.tStopRefresh)
    trials_10.addData('array_image_8m.started', array_image_8m.tStartRefresh)
    trials_10.addData('array_image_8m.stopped', array_image_8m.tStopRefresh)
    trials_10.addData('pointer_dot_0d.started', pointer_dot_0d.tStartRefresh)
    trials_10.addData('pointer_dot_0d.stopped', pointer_dot_0d.tStopRefresh)
    trials_10.addData('pointer_dot_1d.started', pointer_dot_1d.tStartRefresh)
    trials_10.addData('pointer_dot_1d.stopped', pointer_dot_1d.tStopRefresh)
    trials_10.addData('pointer_dot_2d.started', pointer_dot_2d.tStartRefresh)
    trials_10.addData('pointer_dot_2d.stopped', pointer_dot_2d.tStopRefresh)
    trials_10.addData('reminder_training.started', reminder_training.tStartRefresh)
    trials_10.addData('reminder_training.stopped', reminder_training.tStopRefresh)
    # check responses
    if sd_training_resp.keys in ['', [], None]:  # No response was made
        sd_training_resp.keys = None
    trials_10.addData('sd_training_resp.keys',sd_training_resp.keys)
    if sd_training_resp.keys != None:  # we had a response
        trials_10.addData('sd_training_resp.rt', sd_training_resp.rt)
    trials_10.addData('sd_training_resp.started', sd_training_resp.tStartRefresh)
    trials_10.addData('sd_training_resp.stopped', sd_training_resp.tStopRefresh)
    # the Routine "retention_cue_and_test_intval_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "confidence_rating_training"-------
    continueRoutine = True
    # update component parameters for each repeat
    text_15.setText("Don't forget: after each Same/Different choice, you should indicate how confident you are in the choice you just made.\n\nPlease use your left hand, using your ring finger if you're certain (z), your middle finger if you have doubt (x), or your index finger if it was just a guess (c).")
    conf_training_resp.keys = []
    conf_training_resp.rt = []
    _conf_training_resp_allKeys = []
    # keep track of which components have finished
    confidence_rating_trainingComponents = [text_15, conf_training_resp]
    for thisComponent in confidence_rating_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    confidence_rating_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "confidence_rating_training"-------
    while continueRoutine:
        # get current time
        t = confidence_rating_trainingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=confidence_rating_trainingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_15* updates
        if text_15.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
            # keep track of start time/frame for later
            text_15.frameNStart = frameN  # exact frame index
            text_15.tStart = t  # local t and not account for scr refresh
            text_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
            text_15.setAutoDraw(True)
        
        # *conf_training_resp* updates
        waitOnFlip = False
        if conf_training_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            conf_training_resp.frameNStart = frameN  # exact frame index
            conf_training_resp.tStart = t  # local t and not account for scr refresh
            conf_training_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(conf_training_resp, 'tStartRefresh')  # time at next scr refresh
            conf_training_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(conf_training_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(conf_training_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if conf_training_resp.status == STARTED and not waitOnFlip:
            theseKeys = conf_training_resp.getKeys(keyList=['z', 'x', 'c'], waitRelease=False)
            _conf_training_resp_allKeys.extend(theseKeys)
            if len(_conf_training_resp_allKeys):
                conf_training_resp.keys = _conf_training_resp_allKeys[-1].name  # just the last key pressed
                conf_training_resp.rt = _conf_training_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidence_rating_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "confidence_rating_training"-------
    for thisComponent in confidence_rating_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training_trial_index = training_trial_index + 1
    if short_block_done:
        training_trial_index_long_this_block = training_trial_index_long_this_block+1
    
    if (short_block_done == 0) and (consecutive_correct_retro == consecutive_retro_correct_required):
        thisExp.addData('short_interval_training_finished',1)
        short_block_done = 1
       
    if short_block_done:
        
        if training_trial_index_long_this_block==trials_long_perblock:
            
            thisExp.addData('long_interval_block_finished',1)
            
            long_block_retro_acc = float(correct_retro_long_this_block)/float(retro_trials_long_this_block)
            thisExp.addData('long_block_retro_acc',long_block_retro_acc)
            
            if(long_block_retro_acc >= retro_prop_correct_required):
                #trials_10.finished = True
                thisExp.addData('reached_criterion',1)
            else:
                training_trial_index_long_this_block = 0
                correct_retro_long_this_block = 0
                retro_trials_long_this_block = 0
                thisExp.addData('reached_criterion',0)
                
        elif training_trial_index>=max_num_training_trials:
            trials_10.finished = True
            thisExp.addData('run_out_of_trials',1)
    #        win.close()
    #        core.quit()
    trials_10.addData('text_15.started', text_15.tStartRefresh)
    trials_10.addData('text_15.stopped', text_15.tStopRefresh)
    # check responses
    if conf_training_resp.keys in ['', [], None]:  # No response was made
        conf_training_resp.keys = None
    trials_10.addData('conf_training_resp.keys',conf_training_resp.keys)
    if conf_training_resp.keys != None:  # we had a response
        trials_10.addData('conf_training_resp.rt', conf_training_resp.rt)
    trials_10.addData('conf_training_resp.started', conf_training_resp.tStartRefresh)
    trials_10.addData('conf_training_resp.stopped', conf_training_resp.tStopRefresh)
    # the Routine "confidence_rating_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "short_pause"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    short_pauseComponents = [text]
    for thisComponent in short_pauseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    short_pauseClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "short_pause"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = short_pauseClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=short_pauseClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + .5-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in short_pauseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "short_pause"-------
    for thisComponent in short_pauseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_10.addData('text.started', text.tStartRefresh)
    trials_10.addData('text.stopped', text.tStopRefresh)
    thisExp.nextEntry()
    
# completed max_num_training_trials repeats of 'trials_10'


# ------Prepare to start Routine "End"-------
continueRoutine = True
routineTimer.add(8.000000)
# update component parameters for each repeat
# keep track of which components have finished
EndComponents = [finish]
for thisComponent in EndComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
EndClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "End"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = EndClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=EndClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *finish* updates
    if finish.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        finish.frameNStart = frameN  # exact frame index
        finish.tStart = t  # local t and not account for scr refresh
        finish.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(finish, 'tStartRefresh')  # time at next scr refresh
        finish.setAutoDraw(True)
    if finish.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > finish.tStartRefresh + 8-frameTolerance:
            # keep track of stop time/frame for later
            finish.tStop = t  # not accounting for scr refresh
            finish.frameNStop = frameN  # exact frame index
            win.timeOnFlip(finish, 'tStopRefresh')  # time at next scr refresh
            finish.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "End"-------
for thisComponent in EndComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
