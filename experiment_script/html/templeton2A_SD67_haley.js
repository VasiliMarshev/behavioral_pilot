/******************************* 
 * Templeton2A_Sd67_Haley Test *
 *******************************/

import { PsychoJS } from './lib/core-2020.2.js';
import * as core from './lib/core-2020.2.js';
import { TrialHandler } from './lib/data-2020.2.js';
import { Scheduler } from './lib/util-2020.2.js';
import * as visual from './lib/visual-2020.2.js';
import * as sound from './lib/sound-2020.2.js';
import * as util from './lib/util-2020.2.js';
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([1, 1, 1]),
  units: 'height',
  waitBlanking: true
});

// store info about the experiment session:
let expName = 'templeton2A_SD67_haley';  // from the Builder filename that created this script
let expInfo = {'participant': '001', 'EyeTrack': False, 'mapping': '0'};

// Start code blocks for 'Before Experiment'
import * as pylink from 'pylink';
import {EyeLinkCoreGraphicsPsychoPy} from 'EyeLinkCoreGraphicsPsychoPy';

// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(IntroRoutineBegin());
flowScheduler.add(IntroRoutineEachFrame());
flowScheduler.add(IntroRoutineEnd());
flowScheduler.add(instruction_1RoutineBegin());
flowScheduler.add(instruction_1RoutineEachFrame());
flowScheduler.add(instruction_1RoutineEnd());
flowScheduler.add(instruction_2RoutineBegin());
flowScheduler.add(instruction_2RoutineEachFrame());
flowScheduler.add(instruction_2RoutineEnd());
flowScheduler.add(instruction_3RoutineBegin());
flowScheduler.add(instruction_3RoutineEachFrame());
flowScheduler.add(instruction_3RoutineEnd());
flowScheduler.add(instruction_4RoutineBegin());
flowScheduler.add(instruction_4RoutineEachFrame());
flowScheduler.add(instruction_4RoutineEnd());
flowScheduler.add(instructions_4to5RoutineBegin());
flowScheduler.add(instructions_4to5RoutineEachFrame());
flowScheduler.add(instructions_4to5RoutineEnd());
flowScheduler.add(instruction_5RoutineBegin());
flowScheduler.add(instruction_5RoutineEachFrame());
flowScheduler.add(instruction_5RoutineEnd());
flowScheduler.add(instruction_6RoutineBegin());
flowScheduler.add(instruction_6RoutineEachFrame());
flowScheduler.add(instruction_6RoutineEnd());
flowScheduler.add(instruction_67RoutineBegin());
flowScheduler.add(instruction_67RoutineEachFrame());
flowScheduler.add(instruction_67RoutineEnd());
flowScheduler.add(instruction_67bRoutineBegin());
flowScheduler.add(instruction_67bRoutineEachFrame());
flowScheduler.add(instruction_67bRoutineEnd());
flowScheduler.add(instruction_7RoutineBegin());
flowScheduler.add(instruction_7RoutineEachFrame());
flowScheduler.add(instruction_7RoutineEnd());
const trials_2LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_2LoopBegin, trials_2LoopScheduler);
flowScheduler.add(trials_2LoopScheduler);
flowScheduler.add(trials_2LoopEnd);
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin, trialsLoopScheduler);
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);
flowScheduler.add(instructions_specific_trainingRoutineBegin());
flowScheduler.add(instructions_specific_trainingRoutineEachFrame());
flowScheduler.add(instructions_specific_trainingRoutineEnd());
flowScheduler.add(eyetracker_calibRoutineBegin());
flowScheduler.add(eyetracker_calibRoutineEachFrame());
flowScheduler.add(eyetracker_calibRoutineEnd());
const trials_4LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_4LoopBegin, trials_4LoopScheduler);
flowScheduler.add(trials_4LoopScheduler);
flowScheduler.add(trials_4LoopEnd);
const trials_10LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_10LoopBegin, trials_10LoopScheduler);
flowScheduler.add(trials_10LoopScheduler);
flowScheduler.add(trials_10LoopEnd);
flowScheduler.add(instruction_8RoutineBegin());
flowScheduler.add(instruction_8RoutineEachFrame());
flowScheduler.add(instruction_8RoutineEnd());
const main_exp_blocksLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(main_exp_blocksLoopBegin, main_exp_blocksLoopScheduler);
flowScheduler.add(main_exp_blocksLoopScheduler);
flowScheduler.add(main_exp_blocksLoopEnd);
flowScheduler.add(EndRoutineBegin());
flowScheduler.add(EndRoutineEachFrame());
flowScheduler.add(EndRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  });

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.DEBUG);


var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2020.2.10';
  expInfo['OS'] = window.navigator.platform;

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  psychoJS.setRedirectUrls('https://app.prolific.co/submissions/complete?cc=7CD28C93breakthis', '');

  return Scheduler.Event.NEXT;
}


var IntroClock;
var win;
var shuffle;
var tan;
var mod;
var range;
var welcome;
var intro_continue;
var instruction_1Clock;
var static_instruction;
var static_key1;
var static_instruction_2;
var instruction_2Clock;
var static_instruction_3;
var static_key1_2;
var static_instruction_4;
var array_image_1;
var array_image_2;
var array_image_3;
var array_image_4;
var array_image_5;
var array_image_6;
var array_image_7;
var array_image_8;
var polygon_17;
var instruction_3Clock;
var static_instruction_15;
var static_key1_6;
var static_instruction_16;
var array_image_1b;
var array_image_2b;
var array_image_3b;
var array_image_4b;
var array_image_5b;
var array_image_6b;
var array_image_7b;
var array_image_8b;
var polygon_18;
var instruction_4Clock;
var static_instruction_17;
var static_key1_7;
var static_instruction_18;
var array_image_1c;
var array_image_2c;
var array_image_3c;
var array_image_4c;
var array_image_5c;
var array_image_6c;
var array_image_7c;
var array_image_8c;
var polygon_19;
var instructions_4to5Clock;
var static_instruction_25;
var static_key1_11;
var static_instruction_26;
var instruction_5Clock;
var static_instruction_19;
var static_key1_8;
var static_instruction_20;
var polygon_20;
var pointer_dot_0;
var pointer_dot_1;
var pointer_dot_2;
var instruction_6Clock;
var static_instruction_27;
var static_key1_12;
var static_instruction_28;
var polygon_24;
var instruction_67Clock;
var static_instruction_37;
var static_key1_17;
var static_instruction_38;
var SD_image;
var instruction_67bClock;
var static_instruction_39;
var static_key1_18;
var static_instruction_40;
var polygon_37;
var instruction_7Clock;
var static_instruction_31;
var static_key1_14;
var static_instruction_32;
var polygon_29;
var load_practice_trialsClock;
var practice_trial_load_number;
var practice_trial_image_list;
var retro_0_post_1_practice;
var mem_display_to_cue_times_practice;
var change_0_1_practice;
var get_ready_practiceClock;
var static_instruction_29;
var static_key1_13;
var static_instruction_30;
var polygon_25;
var blank_at_trial_startClock;
var polygon_39;
var practice_mem_displayClock;
var array_image_1d;
var array_image_2d;
var array_image_3d;
var array_image_4d;
var array_image_5d;
var array_image_6d;
var array_image_7d;
var array_image_8d;
var polygon_26;
var retention_cue_and_test_intval_practice_2Clock;
var polygon_28;
var array_image_1f;
var array_image_2f;
var array_image_3f;
var array_image_4f;
var array_image_5f;
var array_image_6f;
var array_image_7f;
var array_image_8f;
var pointer_dot_0b;
var pointer_dot_1b;
var pointer_dot_2b;
var sd_practice_resp;
var confidence_rating_practiceClock;
var text_10;
var conf_practice_resp;
var image;
var instructions_specific_trainingClock;
var static_instruction_42;
var polygon_36;
var static_instruction_43;
var static_key1_20;
var eyetracker_calibClock;
var variable_not_used;
var dataFileName;
var load_training_trialsClock;
var training_trial_load_number;
var training_trial_image_list;
var retro_0_post_1_training;
var change_0_1_training;
var consecutive_correct_retro;
var correct_retro_long_perblock;
var retro_trials_long_perblock;
var get_ready_training_blockstartClock;
var static_instruction_44;
var static_key1_21;
var training_mem_displayClock;
var array_image_1l;
var array_image_2l;
var array_image_3l;
var array_image_4l;
var array_image_5l;
var array_image_6l;
var array_image_7l;
var array_image_8l;
var polygon_44;
var retention_cue_and_test_intval_trainingClock;
var polygon_46;
var array_image_1m;
var array_image_2m;
var array_image_3m;
var array_image_4m;
var array_image_5m;
var array_image_6m;
var array_image_7m;
var array_image_8m;
var pointer_dot_0d;
var pointer_dot_1d;
var pointer_dot_2d;
var reminder_training;
var sd_training_resp;
var confidence_rating_trainingClock;
var text_15;
var conf_training_resp;
var instruction_8Clock;
var static_instruction_33;
var static_key1_15;
var static_instruction_34;
var polygon_31;
var load_real_trialsClock;
var trial_load_number;
var real_trial_image_list;
var retro_0_post_1_real;
var change_0_1_real;
var cued_locations_real;
var retint_short_0_long_1;
var get_ready_real_blockstartClock;
var static_instruction_41;
var static_key1_19;
var real_mem_displayClock;
var array_image_1h;
var array_image_2h;
var array_image_3h;
var array_image_4h;
var array_image_5h;
var array_image_6h;
var array_image_7h;
var array_image_8h;
var polygon_33;
var retention_cue_and_test_intval_realClock;
var polygon_34;
var array_image_1j;
var array_image_2j;
var array_image_3j;
var array_image_4j;
var array_image_5j;
var array_image_6j;
var array_image_7j;
var array_image_8j;
var pointer_dot_0c;
var pointer_dot_1c;
var pointer_dot_2c;
var sd_resp_real;
var reminder_real;
var confidence_rating_real_2Clock;
var text_14;
var conf_real_resp;
var EndClock;
var finish;
var globalClock;
var routineTimer;
function experimentInit() {
  // Initialize components for Routine "Intro"
  IntroClock = new util.Clock();
  win=psychoJS.window;
  Array.prototype.append = [].push;
  
  window.addEventListener("keydown", (event) =>
  {
      event.preventDefault();
      event.stopPropagation();
      return false;
  });
  
  //shuffle = util.shuffle;
  
  //shuffle = function(a) {
  //  var j, x, i;
  //  for (i = a.length - 1; i > 0; i--) {
  //      j = Math.floor(Math.random() * (i + 1));
  //      x = a[i];
  //      a[i] = a[j];
  //      a[j] = x;
  //  }
  //  return a;
  //}
  
  shuffle= function(array) {
      for (let i = array.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [array[i], array[j]] = [array[j], array[i]];
      }
  }
  
  tan = function(a) {
      var value;
      value = Math.tan(a);
      return value;
  }
  
  mod = function(a,b) {
      var value;
      value = a % b;
      return value;
  }
  
  range = function(start, end) {
      var ans = [];
      for (let i = start; i < end; i++) {
          ans.push(i);
      }
      return ans;
  }
  
  welcome = new visual.TextStim({
    win: psychoJS.window,
    name: 'welcome',
    text: 'Welcome to the experiment!\n\n\nPress any key to continue.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  intro_continue = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instruction_1"
  instruction_1Clock = new util.Clock();
  static_instruction = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction',
    text: 'On each trial of the experiment you will see two groups of pictures, first one group and then the other a while later. \n\nYour task will involve reporting whether the pictures shown in the first group are the same as the pictures shown in the second group.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_2',
    text: 'Press any key to continue and  see an example of how these groups of pictures look.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  // Initialize components for Routine "instruction_2"
  instruction_2Clock = new util.Clock();
  static_instruction_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_3',
    text: 'The pictures you’ll see will always be arranged in a circle like this. But the specific pictures shown may differ; this is only an example.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.75], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_4',
    text: 'Please press any key for more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.75)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  array_image_1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_4 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_5 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_6 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  array_image_7 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -10.0 
  });
  array_image_8 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -11.0 
  });
  polygon_17 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_17', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  // Initialize components for Routine "instruction_3"
  instruction_3Clock = new util.Clock();
  static_instruction_15 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_15',
    text: 'During each trial the first group of pictures will be shown only briefly, like the flash you just saw. After that no pictures will be on the screen for a while.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_6 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_16 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_16',
    text: 'Please press any key for more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  array_image_1b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_2b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_3b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_4b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_5b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_6b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  array_image_7b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -10.0 
  });
  array_image_8b = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8b', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -11.0 
  });
  polygon_18 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_18', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  // Initialize components for Routine "instruction_4"
  instruction_4Clock = new util.Clock();
  static_instruction_17 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_17',
    text: 'After that pause you will see the second group of pictures. This second group will stay on the screen for as long as you want, as shown here.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.75], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_7 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_18 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_18',
    text: 'Please press any key for more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.75)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  array_image_1c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_2c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_3c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_4c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_5c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_6c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  array_image_7c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -10.0 
  });
  array_image_8c = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8c', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -11.0 
  });
  polygon_19 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_19', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  // Initialize components for Routine "instructions_4to5"
  instructions_4to5Clock = new util.Clock();
  static_instruction_25 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_25',
    text: 'Your task will always be the same:\n\nTo indicate whether, at one particular spot along the circle of pictures, the same picture was shown both times.\n\nIn other words, you will indicate whether the two groups of pictures both had the same picture in one particular location.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_11 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_26 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_26',
    text: 'Please press any key for more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.75)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  // Initialize components for Routine "instruction_5"
  instruction_5Clock = new util.Clock();
  static_instruction_19 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_19',
    text: 'To let you know which spot along the circle you should be reporting about, there will be a red dotted line pointing to one of the eight spots, as shown here.\n\nSo in this example your task would be to identify whether the image to the right and slightly up stayed the same between the two groups of pictures or not, because the red dotted line is pointing to the right and slightly upward.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_8 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_20 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_20',
    text: 'Please press any key for more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  polygon_20 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_20', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -3, interpolate: true,
  });
  
  pointer_dot_0 = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_0', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -4, interpolate: true,
  });
  
  pointer_dot_1 = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_1', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -5, interpolate: true,
  });
  
  pointer_dot_2 = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_2', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -6, interpolate: true,
  });
  
  // Initialize components for Routine "instruction_6"
  instruction_6Clock = new util.Clock();
  static_instruction_27 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_27',
    text: 'You will find that different trials will differ in details: the duration of the pause between the first and second group of pictures may vary, and the moment at which you see the red dotted line may vary, too.\n\nBut your task will always be the same: did the two groups of pictures both have the same picture at the location indicated by red dotted line, or not?',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_12 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_28 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_28',
    text: 'Please press any key to see more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  polygon_24 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_24', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -3, interpolate: true,
  });
  
  // Initialize components for Routine "instruction_67"
  instruction_67Clock = new util.Clock();
  static_instruction_37 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_37',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.8], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_17 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_38 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_38',
    text: 'You will not be explicitly prompted when to give your response: you can simply press one of the two response keys as soon as you’ve seen both groups of pictures as well as the red dotted line indicating the location.\n\nPlease press any key to see more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.7)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  SD_image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'SD_image', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.4, 0.5],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  // Initialize components for Routine "instruction_67b"
  instruction_67bClock = new util.Clock();
  static_instruction_39 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_39',
    text: 'You will find that some trials are easy, yet that on other trials you have no idea and have to guess. That’s okay: immediately after each Same/Different response you get a chance to indicate how sure you were of that response.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_18 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_40 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_40',
    text: 'Please press any key to see more instructions.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  polygon_37 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_37', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -3, interpolate: true,
  });
  
  // Initialize components for Routine "instruction_7"
  instruction_7Clock = new util.Clock();
  static_instruction_31 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_31',
    text: 'We will start with a few practice trials so that you can see for yourself.\n\nBefore starting the practice trials: here’s one final instruction. During the trials you will see a black dot at the screen center, just like the one you see there now. It is important to keep your eyes directed at that black dot during the trials. So no looking around while a trial is ongoing.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  static_key1_14 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_32 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_32',
    text: 'Please press any key to begin the practice trials.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  polygon_29 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_29', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -3, interpolate: true,
  });
  
  // Initialize components for Routine "load_practice_trials"
  load_practice_trialsClock = new util.Clock();
  practice_trial_load_number = 0;
  practice_trial_image_list = [];
  retro_0_post_1_practice = [];
  mem_display_to_cue_times_practice = [];
  change_0_1_practice = [];
  
  // Initialize components for Routine "get_ready_practice"
  get_ready_practiceClock = new util.Clock();
  static_instruction_29 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_29',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_13 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_30 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_30',
    text: 'Please press any key to start.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  polygon_25 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_25', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -4, interpolate: true,
  });
  
  // Initialize components for Routine "blank_at_trial_start"
  blank_at_trial_startClock = new util.Clock();
  polygon_39 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_39', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  // Initialize components for Routine "practice_mem_display"
  practice_mem_displayClock = new util.Clock();
  array_image_1d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -1.0 
  });
  array_image_2d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_3d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_4d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_5d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_6d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_7d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_8d = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8d', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  polygon_26 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_26', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -9, interpolate: true,
  });
  
  // Initialize components for Routine "retention_cue_and_test_intval_practice_2"
  retention_cue_and_test_intval_practice_2Clock = new util.Clock();
  polygon_28 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_28', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -1, interpolate: true,
  });
  
  array_image_1f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_2f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_3f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_4f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_5f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_6f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_7f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_8f = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8f', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  pointer_dot_0b = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_0b', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -10, interpolate: true,
  });
  
  pointer_dot_1b = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_1b', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -11, interpolate: true,
  });
  
  pointer_dot_2b = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_2b', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  sd_practice_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "confidence_rating_practice"
  confidence_rating_practiceClock = new util.Clock();
  text_10 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_10',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.65], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  conf_practice_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image', units : undefined, 
    image : 'LH_buttonbox.JPG', mask : undefined,
    ori : 0, pos : [0, (- 0.1)], size : [0.35, 0.5],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -3.0 
  });
  // Initialize components for Routine "instructions_specific_training"
  instructions_specific_trainingClock = new util.Clock();
  static_instruction_42 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_42',
    text: 'Good. Now that you know the basic idea, let’s get better at the task. Here’s a tip that helps people do well:\n\nAs soon as you see the red dotted line pointing to some spot, immediately try to recall or verbalize what occupied that spot in the first group of pictures. You’ll find that, the faster you try to recall it, the better you do.\n\nThis also means that, if the red dotted line appears earlier than the second group of pictures, don’t wait for that second group to appear! Instead, try to recall what was at the indicated spot right away, before the second group of pictures appears. People who follow that strategy tend to be our best performers.\n\nBefore we start the experiment, we will do a short calibration for the eye-tracking device. Please follow the black dots with your eyes.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  polygon_36 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_36', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -1, interpolate: true,
  });
  
  static_instruction_43 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_43',
    text: 'Please press any key to start the calibration then to try out this strategy during a few training trials.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  static_key1_20 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "eyetracker_calib"
  eyetracker_calibClock = new util.Clock();
  if (expInfo["EyeTrack"]) {
      variable_not_used = new pylink.EyeLink();
  } else {
      pylink.EyeLink.dummy_open;
      variable_not_used = new pylink.EyeLink(null);
  }
  [scnWidth, scnHeight] = [1920, 1080];
  dataFileName = (expInfo["participant"] + ".EDF");
  pylink.getEYELINK().openDataFile(dataFileName);
  pylink.getEYELINK().sendCommand("add_file_preamble_text 'Templeton2A_Exp'");
  
  // Initialize components for Routine "load_training_trials"
  load_training_trialsClock = new util.Clock();
  training_trial_load_number = 0;
  training_trial_image_list = [];
  retro_0_post_1_training = [];
  change_0_1_training = [];
  consecutive_correct_retro = 0;
  correct_retro_long_perblock = 0;
  retro_trials_long_perblock = 0;
  
  // Initialize components for Routine "get_ready_training_blockstart"
  get_ready_training_blockstartClock = new util.Clock();
  static_instruction_44 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_44',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_21 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "blank_at_trial_start"
  blank_at_trial_startClock = new util.Clock();
  polygon_39 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_39', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  // Initialize components for Routine "training_mem_display"
  training_mem_displayClock = new util.Clock();
  array_image_1l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -1.0 
  });
  array_image_2l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_3l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_4l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_5l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_6l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_7l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_8l = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8l', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  polygon_44 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_44', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -9, interpolate: true,
  });
  
  // Initialize components for Routine "retention_cue_and_test_intval_training"
  retention_cue_and_test_intval_trainingClock = new util.Clock();
  polygon_46 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_46', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -1, interpolate: true,
  });
  
  array_image_1m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_2m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_3m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_4m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_5m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_6m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_7m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_8m = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8m', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  pointer_dot_0d = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_0d', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -10, interpolate: true,
  });
  
  pointer_dot_1d = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_1d', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -11, interpolate: true,
  });
  
  pointer_dot_2d = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_2d', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  reminder_training = new visual.TextStim({
    win: psychoJS.window,
    name: 'reminder_training',
    text: 'Don’t forget to press S or D for ‘Same’ or ‘Different’. You’ve seen both groups of pictures, as well as the red dotted line indicating the location. If you don’t know the answer just guess; that’s OK.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.75], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -13.0 
  });
  
  sd_training_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "confidence_rating_training"
  confidence_rating_trainingClock = new util.Clock();
  text_15 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_15',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.65], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  conf_training_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instruction_8"
  instruction_8Clock = new util.Clock();
  static_instruction_33 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_33',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.5], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_15 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  static_instruction_34 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_34',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.5)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  polygon_31 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_31', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -4, interpolate: true,
  });
  
  // Initialize components for Routine "load_real_trials"
  load_real_trialsClock = new util.Clock();
  trial_load_number = 0;
  real_trial_image_list = [];
  retro_0_post_1_real = [];
  change_0_1_real = [];
  cued_locations_real = [];
  retint_short_0_long_1 = [];
  
  // Initialize components for Routine "get_ready_real_blockstart"
  get_ready_real_blockstartClock = new util.Clock();
  static_instruction_41 = new visual.TextStim({
    win: psychoJS.window,
    name: 'static_instruction_41',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  static_key1_19 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "blank_at_trial_start"
  blank_at_trial_startClock = new util.Clock();
  polygon_39 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_39', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  // Initialize components for Routine "real_mem_display"
  real_mem_displayClock = new util.Clock();
  array_image_1h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -1.0 
  });
  array_image_2h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_3h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_4h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_5h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_6h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_7h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_8h = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8h', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  polygon_33 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_33', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: -9, interpolate: true,
  });
  
  // Initialize components for Routine "retention_cue_and_test_intval_real"
  retention_cue_and_test_intval_realClock = new util.Clock();
  polygon_34 = new visual.Polygon ({
    win: psychoJS.window, name: 'polygon_34', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1), (- 1), (- 1)]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  array_image_1j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_1j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  array_image_2j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_2j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  array_image_3j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_3j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -4.0 
  });
  array_image_4j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_4j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -5.0 
  });
  array_image_5j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_5j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -6.0 
  });
  array_image_6j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_6j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -7.0 
  });
  array_image_7j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_7j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -8.0 
  });
  array_image_8j = new visual.ImageStim({
    win : psychoJS.window,
    name : 'array_image_8j', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -9.0 
  });
  pointer_dot_0c = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_0c', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -10, interpolate: true,
  });
  
  pointer_dot_1c = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_1c', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -11, interpolate: true,
  });
  
  pointer_dot_2c = new visual.Polygon ({
    win: psychoJS.window, name: 'pointer_dot_2c', units : 'pix', 
    edges: 20, size:[1.0, 1.0],
    ori: 0, pos: [0, 0],
    lineWidth: 1, lineColor: new util.Color([1, (- 1), (- 1)]),
    fillColor: new util.Color([1, (- 1), (- 1)]),
    opacity: 1, depth: -12, interpolate: true,
  });
  
  sd_resp_real = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  reminder_real = new visual.TextStim({
    win: psychoJS.window,
    name: 'reminder_real',
    text: 'Don’t forget to press S or D for ‘Same’ or ‘Different’. You’ve seen both groups of pictures, as well as the red dotted line indicating the location. If you don’t know the answer just guess; that’s OK.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.75], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -14.0 
  });
  
  // Initialize components for Routine "confidence_rating_real_2"
  confidence_rating_real_2Clock = new util.Clock();
  text_14 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_14',
    text: 'default text',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.65], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  conf_real_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "End"
  EndClock = new util.Clock();
  finish = new visual.TextStim({
    win: psychoJS.window,
    name: 'finish',
    text: 'You’re done with the task!\n\nThank you for your participation.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var random;
var timer;
var thisExp;
var event;
var x_scale;
var y_scale;
var dist_cm;
var trials_per_block;
var num_real_blocks;
var num_real_trials;
var real_trial_index;
var blank_trial_start_dur_s;
var mem_display_duration_s;
var mem_display_to_cue_time_s;
var cue_duration_s;
var retention_duration_s;
var max_num_training_trials;
var consecutive_retro_correct_required;
var repeat_intructions_every;
var mem_display_to_cue_times_training_s;
var short_block_done;
var trials_long_perblock;
var num_images_per_display;
var image_index_list_for_shuffling;
var num_images_available;
var num_practice_trials;
var permutation_data_file_string_practice;
var image_side_dva;
var image_eccentr_dva;
var outline_side_relative_to_image_eccentricity;
var pointer_length_relative_to_image_eccentricity;
var x_y_gains_per_angle;
var dot_diam_dva;
var my_map;
var key_response_mapping_option;
var key_mapping_instruction_text_options;
var key_mapping_instruction_image_options;
var response_string_pair_options;
var key_mapping_instruction_text;
var key_mapping_instruction_image;
var response_string_pair;
var all_pic_names;
var all_pic_names_training;
var _intro_continue_allKeys;
var IntroComponents;
function IntroRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'Intro'-------
    t = 0;
    IntroClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    random = Math.random;
    
    //Math.seedrandom();
    
    //round = function(num, n=0) {    
    //    return +(Math.round(num + ("e+" + n))  + ("e-" + n));
    //};
    
    timer  = new util.Clock();
    thisExp = psychoJS.experiment;
    event = psychoJS.eventManager;
    
    
    
            // add-on: list(s: string): string[]
            function list(s) {
                // if s is a string, we return a list of its characters
                if (typeof s === 'string')
                    return s.split('');
                else
                    // otherwise we return s:
                    return s;
            }
    
            x_scale = 50;
    y_scale = 50;
    dist_cm = 60;
    trials_per_block = 50;
    num_real_blocks = 10;
    num_real_trials = (trials_per_block * num_real_blocks);
    real_trial_index = 0;
    blank_trial_start_dur_s = 0.8;
    mem_display_duration_s = 0.5;
    mem_display_to_cue_time_s = 2.0;
    cue_duration_s = 0.5;
    retention_duration_s = 0.8;
    max_num_training_trials = 300;
    consecutive_retro_correct_required = 8;
    repeat_intructions_every = 15;
    mem_display_to_cue_times_training_s = [0.75, 2.6];
    short_block_done = 0;
    trials_long_perblock = 50;
    num_images_per_display = 8;
    image_index_list_for_shuffling = list(range(0, num_images_per_display));
    num_images_available = 10;
    num_practice_trials = 7;
    permutation_data_file_string_practice = "limited_permutations.xlsx";
    image_side_dva = 2.0;
    image_eccentr_dva = 4.5;
    outline_side_relative_to_image_eccentricity = 2.5;
    pointer_length_relative_to_image_eccentricity = 0.25;
    x_y_gains_per_angle = [[0.3826834323650898, 0.9238795325112867], [0.9238795325112867, 0.38268343236508984], [0.9238795325112867, (- 0.3826834323650897)], [0.38268343236508984, (- 0.9238795325112867)], [(- 0.3826834323650892), (- 0.923879532511287)], [(- 0.9238795325112868), (- 0.3826834323650895)], [(- 0.9238795325112866), 0.38268343236509], [(- 0.38268343236508956), 0.9238795325112868]];
    dot_diam_dva = 0.2;
    my_map = Number.parseInt(expInfo["mapping"]);
    key_response_mapping_option = my_map;
    key_mapping_instruction_text_options = ["To indicate your response you\u2019ll use your right hand; press the left button with your index finger if the two groups of pictures both had the same picture at the location indicated by the red dotted line, or press the right button with your middle finger if two different pictures were shown at that location. See the image below:", "To indicate your response you\u2019ll use your right hand; press the right button with your middle finger if the two groups of pictures both had the same picture at the location indicated by the red dotted line, or press the left button with your index finger if two different pictures were shown at that location. See the image below:"];
    key_mapping_instruction_image_options = ["RH_buttonbox_V1.jpg", "RH_buttonbox_V2.jpg"];
    response_string_pair_options = [["6", "7"], ["7", "6"]];
    key_mapping_instruction_text = key_mapping_instruction_text_options[key_response_mapping_option];
    key_mapping_instruction_image = key_mapping_instruction_image_options[key_response_mapping_option];
    response_string_pair = response_string_pair_options[key_response_mapping_option];
    all_pic_names = ["bow.png", "clown.png", "car.png", "book.png", "bed.png", "moon.png", "strawberry.png", "lemon.png", "sun.png", "flag.png"];
    all_pic_names_training = ["accordion.png", "celery.png", "anchor.png", "football_helmet.png", "arm.png", "ax.png", "bike.png", "corn.png", "cat.png", "chair.png"];
    
    intro_continue.keys = undefined;
    intro_continue.rt = undefined;
    _intro_continue_allKeys = [];
    // keep track of which components have finished
    IntroComponents = [];
    IntroComponents.push(welcome);
    IntroComponents.push(intro_continue);
    
    for (const thisComponent of IntroComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function IntroRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'Intro'-------
    // get current time
    t = IntroClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *welcome* updates
    if (t >= 0.0 && welcome.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      welcome.tStart = t;  // (not accounting for frame time here)
      welcome.frameNStart = frameN;  // exact frame index
      
      welcome.setAutoDraw(true);
    }

    
    // *intro_continue* updates
    if (t >= 0.0 && intro_continue.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      intro_continue.tStart = t;  // (not accounting for frame time here)
      intro_continue.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { intro_continue.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { intro_continue.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { intro_continue.clearEvents(); });
    }

    if (intro_continue.status === PsychoJS.Status.STARTED) {
      let theseKeys = intro_continue.getKeys({keyList: [], waitRelease: false});
      _intro_continue_allKeys = _intro_continue_allKeys.concat(theseKeys);
      if (_intro_continue_allKeys.length > 0) {
        intro_continue.keys = _intro_continue_allKeys[_intro_continue_allKeys.length - 1].name;  // just the last key pressed
        intro_continue.rt = _intro_continue_allKeys[_intro_continue_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of IntroComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function IntroRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'Intro'-------
    for (const thisComponent of IntroComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_allKeys;
var instruction_1Components;
function instruction_1RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_1'-------
    t = 0;
    instruction_1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1.keys = undefined;
    static_key1.rt = undefined;
    _static_key1_allKeys = [];
    // keep track of which components have finished
    instruction_1Components = [];
    instruction_1Components.push(static_instruction);
    instruction_1Components.push(static_key1);
    instruction_1Components.push(static_instruction_2);
    
    for (const thisComponent of instruction_1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_1RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_1'-------
    // get current time
    t = instruction_1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction* updates
    if (t >= 0.0 && static_instruction.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction.tStart = t;  // (not accounting for frame time here)
      static_instruction.frameNStart = frameN;  // exact frame index
      
      static_instruction.setAutoDraw(true);
    }

    
    // *static_key1* updates
    if (t >= 0.0 && static_key1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1.tStart = t;  // (not accounting for frame time here)
      static_key1.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1.clearEvents(); });
    }

    if (static_key1.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1.getKeys({keyList: [], waitRelease: false});
      _static_key1_allKeys = _static_key1_allKeys.concat(theseKeys);
      if (_static_key1_allKeys.length > 0) {
        static_key1.keys = _static_key1_allKeys[_static_key1_allKeys.length - 1].name;  // just the last key pressed
        static_key1.rt = _static_key1_allKeys[_static_key1_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_2* updates
    if (t >= 0.0 && static_instruction_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_2.tStart = t;  // (not accounting for frame time here)
      static_instruction_2.frameNStart = frameN;  // exact frame index
      
      static_instruction_2.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_1RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_1'-------
    for (const thisComponent of instruction_1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var distance_cm;
var image_eccentr_cm;
var image_side_cm;
var random_pic_names;
var pic_1_name;
var pic_2_name;
var pic_3_name;
var pic_4_name;
var pic_5_name;
var pic_6_name;
var pic_7_name;
var pic_8_name;
var _static_key1_2_allKeys;
var instruction_2Components;
function instruction_2RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_2'-------
    t = 0;
    instruction_2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    distance_cm = dist_cm;
    image_eccentr_cm = (tan(((3.14159 * image_eccentr_dva) / 180.0)) * distance_cm);
    image_side_cm = ((2.0 * tan((((0.5 * 3.14159) * image_side_dva) / 180.0))) * distance_cm);
    random_pic_names = all_pic_names_training.slice(0);
    shuffle(random_pic_names);
    pic_1_name = random_pic_names[0];
    array_image_1.setImage(pic_1_name, false);
    array_image_1.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_2_name = random_pic_names[1];
    array_image_2.setImage(pic_2_name, false);
    array_image_2.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_3_name = random_pic_names[2];
    array_image_3.setImage(pic_3_name, false);
    array_image_3.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_4_name = random_pic_names[3];
    array_image_4.setImage(pic_4_name, false);
    array_image_4.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_5_name = random_pic_names[4];
    array_image_5.setImage(pic_5_name, false);
    array_image_5.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_6_name = random_pic_names[5];
    array_image_6.setImage(pic_6_name, false);
    array_image_6.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_7_name = random_pic_names[6];
    array_image_7.setImage(pic_7_name, false);
    array_image_7.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    pic_8_name = random_pic_names[7];
    array_image_8.setImage(pic_8_name, false);
    array_image_8.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    
    static_key1_2.keys = undefined;
    static_key1_2.rt = undefined;
    _static_key1_2_allKeys = [];
    array_image_1.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1.setImage(pic_1_name);
    array_image_2.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2.setImage(pic_2_name);
    array_image_3.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3.setImage(pic_3_name);
    array_image_4.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4.setImage(pic_4_name);
    array_image_5.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5.setImage(pic_5_name);
    array_image_6.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6.setImage(pic_6_name);
    array_image_7.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7.setImage(pic_7_name);
    array_image_8.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8.setImage(pic_8_name);
    polygon_17.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_2Components = [];
    instruction_2Components.push(static_instruction_3);
    instruction_2Components.push(static_key1_2);
    instruction_2Components.push(static_instruction_4);
    instruction_2Components.push(array_image_1);
    instruction_2Components.push(array_image_2);
    instruction_2Components.push(array_image_3);
    instruction_2Components.push(array_image_4);
    instruction_2Components.push(array_image_5);
    instruction_2Components.push(array_image_6);
    instruction_2Components.push(array_image_7);
    instruction_2Components.push(array_image_8);
    instruction_2Components.push(polygon_17);
    
    for (const thisComponent of instruction_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_2RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_2'-------
    // get current time
    t = instruction_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_3* updates
    if (t >= 0.0 && static_instruction_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_3.tStart = t;  // (not accounting for frame time here)
      static_instruction_3.frameNStart = frameN;  // exact frame index
      
      static_instruction_3.setAutoDraw(true);
    }

    
    // *static_key1_2* updates
    if (t >= 0.0 && static_key1_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_2.tStart = t;  // (not accounting for frame time here)
      static_key1_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_2.clearEvents(); });
    }

    if (static_key1_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_2.getKeys({keyList: [], waitRelease: false});
      _static_key1_2_allKeys = _static_key1_2_allKeys.concat(theseKeys);
      if (_static_key1_2_allKeys.length > 0) {
        static_key1_2.keys = _static_key1_2_allKeys[_static_key1_2_allKeys.length - 1].name;  // just the last key pressed
        static_key1_2.rt = _static_key1_2_allKeys[_static_key1_2_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_4* updates
    if (t >= 0.0 && static_instruction_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_4.tStart = t;  // (not accounting for frame time here)
      static_instruction_4.frameNStart = frameN;  // exact frame index
      
      static_instruction_4.setAutoDraw(true);
    }

    
    // *array_image_1* updates
    if (t >= 0.0 && array_image_1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1.tStart = t;  // (not accounting for frame time here)
      array_image_1.frameNStart = frameN;  // exact frame index
      
      array_image_1.setAutoDraw(true);
    }

    
    // *array_image_2* updates
    if (t >= 0.0 && array_image_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2.tStart = t;  // (not accounting for frame time here)
      array_image_2.frameNStart = frameN;  // exact frame index
      
      array_image_2.setAutoDraw(true);
    }

    
    // *array_image_3* updates
    if (t >= 0.0 && array_image_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3.tStart = t;  // (not accounting for frame time here)
      array_image_3.frameNStart = frameN;  // exact frame index
      
      array_image_3.setAutoDraw(true);
    }

    
    // *array_image_4* updates
    if (t >= 0.0 && array_image_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4.tStart = t;  // (not accounting for frame time here)
      array_image_4.frameNStart = frameN;  // exact frame index
      
      array_image_4.setAutoDraw(true);
    }

    
    // *array_image_5* updates
    if (t >= 0.0 && array_image_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5.tStart = t;  // (not accounting for frame time here)
      array_image_5.frameNStart = frameN;  // exact frame index
      
      array_image_5.setAutoDraw(true);
    }

    
    // *array_image_6* updates
    if (t >= 0.0 && array_image_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6.tStart = t;  // (not accounting for frame time here)
      array_image_6.frameNStart = frameN;  // exact frame index
      
      array_image_6.setAutoDraw(true);
    }

    
    // *array_image_7* updates
    if (t >= 0.0 && array_image_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7.tStart = t;  // (not accounting for frame time here)
      array_image_7.frameNStart = frameN;  // exact frame index
      
      array_image_7.setAutoDraw(true);
    }

    
    // *array_image_8* updates
    if (t >= 0.0 && array_image_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8.tStart = t;  // (not accounting for frame time here)
      array_image_8.frameNStart = frameN;  // exact frame index
      
      array_image_8.setAutoDraw(true);
    }

    
    // *polygon_17* updates
    if (t >= 0.0 && polygon_17.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_17.tStart = t;  // (not accounting for frame time here)
      polygon_17.frameNStart = frameN;  // exact frame index
      
      polygon_17.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_2RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_2'-------
    for (const thisComponent of instruction_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_6_allKeys;
var instruction_3Components;
function instruction_3RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_3'-------
    t = 0;
    instruction_3Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    array_image_1b.setImage(pic_1_name, false);
    array_image_1b.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_2b.setImage(pic_2_name, false);
    array_image_2b.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_3b.setImage(pic_3_name, false);
    array_image_3b.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_4b.setImage(pic_4_name, false);
    array_image_4b.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_5b.setImage(pic_5_name, false);
    array_image_5b.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_6b.setImage(pic_6_name, false);
    array_image_6b.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_7b.setImage(pic_7_name, false);
    array_image_7b.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_8b.setImage(pic_8_name, false);
    array_image_8b.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    
    static_key1_6.keys = undefined;
    static_key1_6.rt = undefined;
    _static_key1_6_allKeys = [];
    array_image_1b.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1b.setImage(pic_1_name);
    array_image_2b.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2b.setImage(pic_2_name);
    array_image_3b.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3b.setImage(pic_3_name);
    array_image_4b.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4b.setImage(pic_4_name);
    array_image_5b.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5b.setImage(pic_5_name);
    array_image_6b.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6b.setImage(pic_6_name);
    array_image_7b.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7b.setImage(pic_7_name);
    array_image_8b.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8b.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8b.setImage(pic_8_name);
    polygon_18.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_3Components = [];
    instruction_3Components.push(static_instruction_15);
    instruction_3Components.push(static_key1_6);
    instruction_3Components.push(static_instruction_16);
    instruction_3Components.push(array_image_1b);
    instruction_3Components.push(array_image_2b);
    instruction_3Components.push(array_image_3b);
    instruction_3Components.push(array_image_4b);
    instruction_3Components.push(array_image_5b);
    instruction_3Components.push(array_image_6b);
    instruction_3Components.push(array_image_7b);
    instruction_3Components.push(array_image_8b);
    instruction_3Components.push(polygon_18);
    
    for (const thisComponent of instruction_3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function instruction_3RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_3'-------
    // get current time
    t = instruction_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_15* updates
    if (t >= 1.0 && static_instruction_15.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_15.tStart = t;  // (not accounting for frame time here)
      static_instruction_15.frameNStart = frameN;  // exact frame index
      
      static_instruction_15.setAutoDraw(true);
    }

    
    // *static_key1_6* updates
    if (t >= 1.0 && static_key1_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_6.tStart = t;  // (not accounting for frame time here)
      static_key1_6.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_6.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_6.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_6.clearEvents(); });
    }

    if (static_key1_6.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_6.getKeys({keyList: [], waitRelease: false});
      _static_key1_6_allKeys = _static_key1_6_allKeys.concat(theseKeys);
      if (_static_key1_6_allKeys.length > 0) {
        static_key1_6.keys = _static_key1_6_allKeys[_static_key1_6_allKeys.length - 1].name;  // just the last key pressed
        static_key1_6.rt = _static_key1_6_allKeys[_static_key1_6_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_16* updates
    if (t >= 1.0 && static_instruction_16.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_16.tStart = t;  // (not accounting for frame time here)
      static_instruction_16.frameNStart = frameN;  // exact frame index
      
      static_instruction_16.setAutoDraw(true);
    }

    
    // *array_image_1b* updates
    if (t >= 0.5 && array_image_1b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1b.tStart = t;  // (not accounting for frame time here)
      array_image_1b.frameNStart = frameN;  // exact frame index
      
      array_image_1b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_1b.status === PsychoJS.Status.STARTED || array_image_1b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_1b.setAutoDraw(false);
    }
    
    // *array_image_2b* updates
    if (t >= 0.5 && array_image_2b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2b.tStart = t;  // (not accounting for frame time here)
      array_image_2b.frameNStart = frameN;  // exact frame index
      
      array_image_2b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_2b.status === PsychoJS.Status.STARTED || array_image_2b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_2b.setAutoDraw(false);
    }
    
    // *array_image_3b* updates
    if (t >= 0.5 && array_image_3b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3b.tStart = t;  // (not accounting for frame time here)
      array_image_3b.frameNStart = frameN;  // exact frame index
      
      array_image_3b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_3b.status === PsychoJS.Status.STARTED || array_image_3b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_3b.setAutoDraw(false);
    }
    
    // *array_image_4b* updates
    if (t >= 0.5 && array_image_4b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4b.tStart = t;  // (not accounting for frame time here)
      array_image_4b.frameNStart = frameN;  // exact frame index
      
      array_image_4b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_4b.status === PsychoJS.Status.STARTED || array_image_4b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_4b.setAutoDraw(false);
    }
    
    // *array_image_5b* updates
    if (t >= 0.5 && array_image_5b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5b.tStart = t;  // (not accounting for frame time here)
      array_image_5b.frameNStart = frameN;  // exact frame index
      
      array_image_5b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_5b.status === PsychoJS.Status.STARTED || array_image_5b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_5b.setAutoDraw(false);
    }
    
    // *array_image_6b* updates
    if (t >= 0.5 && array_image_6b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6b.tStart = t;  // (not accounting for frame time here)
      array_image_6b.frameNStart = frameN;  // exact frame index
      
      array_image_6b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_6b.status === PsychoJS.Status.STARTED || array_image_6b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_6b.setAutoDraw(false);
    }
    
    // *array_image_7b* updates
    if (t >= 0.5 && array_image_7b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7b.tStart = t;  // (not accounting for frame time here)
      array_image_7b.frameNStart = frameN;  // exact frame index
      
      array_image_7b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_7b.status === PsychoJS.Status.STARTED || array_image_7b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_7b.setAutoDraw(false);
    }
    
    // *array_image_8b* updates
    if (t >= 0.5 && array_image_8b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8b.tStart = t;  // (not accounting for frame time here)
      array_image_8b.frameNStart = frameN;  // exact frame index
      
      array_image_8b.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_8b.status === PsychoJS.Status.STARTED || array_image_8b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_8b.setAutoDraw(false);
    }
    
    // *polygon_18* updates
    if (t >= 0.0 && polygon_18.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_18.tStart = t;  // (not accounting for frame time here)
      polygon_18.frameNStart = frameN;  // exact frame index
      
      polygon_18.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_3RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_3'-------
    for (const thisComponent of instruction_3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_7_allKeys;
var instruction_4Components;
function instruction_4RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_4'-------
    t = 0;
    instruction_4Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    array_image_1c.setImage(pic_1_name, false);
    array_image_1c.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_2c.setImage(pic_2_name, false);
    array_image_2c.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_3c.setImage(pic_3_name, false);
    array_image_3c.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_4c.setImage(pic_4_name, false);
    array_image_4c.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_5c.setImage(pic_5_name, false);
    array_image_5c.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_6c.setImage(pic_6_name, false);
    array_image_6c.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_7c.setImage(pic_7_name, false);
    array_image_7c.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    array_image_8c.setImage(pic_8_name, false);
    array_image_8c.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    
    static_key1_7.keys = undefined;
    static_key1_7.rt = undefined;
    _static_key1_7_allKeys = [];
    array_image_1c.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1c.setImage(pic_1_name);
    array_image_2c.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2c.setImage(pic_2_name);
    array_image_3c.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3c.setImage(pic_3_name);
    array_image_4c.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4c.setImage(pic_4_name);
    array_image_5c.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5c.setImage(pic_5_name);
    array_image_6c.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6c.setImage(pic_6_name);
    array_image_7c.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7c.setImage(pic_7_name);
    array_image_8c.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8c.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8c.setImage(pic_8_name);
    polygon_19.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_4Components = [];
    instruction_4Components.push(static_instruction_17);
    instruction_4Components.push(static_key1_7);
    instruction_4Components.push(static_instruction_18);
    instruction_4Components.push(array_image_1c);
    instruction_4Components.push(array_image_2c);
    instruction_4Components.push(array_image_3c);
    instruction_4Components.push(array_image_4c);
    instruction_4Components.push(array_image_5c);
    instruction_4Components.push(array_image_6c);
    instruction_4Components.push(array_image_7c);
    instruction_4Components.push(array_image_8c);
    instruction_4Components.push(polygon_19);
    
    for (const thisComponent of instruction_4Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_4RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_4'-------
    // get current time
    t = instruction_4Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_17* updates
    if (t >= 0 && static_instruction_17.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_17.tStart = t;  // (not accounting for frame time here)
      static_instruction_17.frameNStart = frameN;  // exact frame index
      
      static_instruction_17.setAutoDraw(true);
    }

    
    // *static_key1_7* updates
    if (t >= 0 && static_key1_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_7.tStart = t;  // (not accounting for frame time here)
      static_key1_7.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_7.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_7.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_7.clearEvents(); });
    }

    if (static_key1_7.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_7.getKeys({keyList: [], waitRelease: false});
      _static_key1_7_allKeys = _static_key1_7_allKeys.concat(theseKeys);
      if (_static_key1_7_allKeys.length > 0) {
        static_key1_7.keys = _static_key1_7_allKeys[_static_key1_7_allKeys.length - 1].name;  // just the last key pressed
        static_key1_7.rt = _static_key1_7_allKeys[_static_key1_7_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_18* updates
    if (t >= 0 && static_instruction_18.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_18.tStart = t;  // (not accounting for frame time here)
      static_instruction_18.frameNStart = frameN;  // exact frame index
      
      static_instruction_18.setAutoDraw(true);
    }

    
    // *array_image_1c* updates
    if (t >= 0 && array_image_1c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1c.tStart = t;  // (not accounting for frame time here)
      array_image_1c.frameNStart = frameN;  // exact frame index
      
      array_image_1c.setAutoDraw(true);
    }

    
    // *array_image_2c* updates
    if (t >= 0 && array_image_2c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2c.tStart = t;  // (not accounting for frame time here)
      array_image_2c.frameNStart = frameN;  // exact frame index
      
      array_image_2c.setAutoDraw(true);
    }

    
    // *array_image_3c* updates
    if (t >= 0 && array_image_3c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3c.tStart = t;  // (not accounting for frame time here)
      array_image_3c.frameNStart = frameN;  // exact frame index
      
      array_image_3c.setAutoDraw(true);
    }

    
    // *array_image_4c* updates
    if (t >= 0 && array_image_4c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4c.tStart = t;  // (not accounting for frame time here)
      array_image_4c.frameNStart = frameN;  // exact frame index
      
      array_image_4c.setAutoDraw(true);
    }

    
    // *array_image_5c* updates
    if (t >= 0 && array_image_5c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5c.tStart = t;  // (not accounting for frame time here)
      array_image_5c.frameNStart = frameN;  // exact frame index
      
      array_image_5c.setAutoDraw(true);
    }

    
    // *array_image_6c* updates
    if (t >= 0 && array_image_6c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6c.tStart = t;  // (not accounting for frame time here)
      array_image_6c.frameNStart = frameN;  // exact frame index
      
      array_image_6c.setAutoDraw(true);
    }

    
    // *array_image_7c* updates
    if (t >= 0 && array_image_7c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7c.tStart = t;  // (not accounting for frame time here)
      array_image_7c.frameNStart = frameN;  // exact frame index
      
      array_image_7c.setAutoDraw(true);
    }

    
    // *array_image_8c* updates
    if (t >= 0 && array_image_8c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8c.tStart = t;  // (not accounting for frame time here)
      array_image_8c.frameNStart = frameN;  // exact frame index
      
      array_image_8c.setAutoDraw(true);
    }

    
    // *polygon_19* updates
    if (t >= 0.0 && polygon_19.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_19.tStart = t;  // (not accounting for frame time here)
      polygon_19.frameNStart = frameN;  // exact frame index
      
      polygon_19.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_4Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_4RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_4'-------
    for (const thisComponent of instruction_4Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_11_allKeys;
var instructions_4to5Components;
function instructions_4to5RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instructions_4to5'-------
    t = 0;
    instructions_4to5Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1_11.keys = undefined;
    static_key1_11.rt = undefined;
    _static_key1_11_allKeys = [];
    // keep track of which components have finished
    instructions_4to5Components = [];
    instructions_4to5Components.push(static_instruction_25);
    instructions_4to5Components.push(static_key1_11);
    instructions_4to5Components.push(static_instruction_26);
    
    for (const thisComponent of instructions_4to5Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instructions_4to5RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instructions_4to5'-------
    // get current time
    t = instructions_4to5Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_25* updates
    if (t >= 0.0 && static_instruction_25.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_25.tStart = t;  // (not accounting for frame time here)
      static_instruction_25.frameNStart = frameN;  // exact frame index
      
      static_instruction_25.setAutoDraw(true);
    }

    
    // *static_key1_11* updates
    if (t >= 0.0 && static_key1_11.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_11.tStart = t;  // (not accounting for frame time here)
      static_key1_11.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_11.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_11.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_11.clearEvents(); });
    }

    if (static_key1_11.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_11.getKeys({keyList: [], waitRelease: false});
      _static_key1_11_allKeys = _static_key1_11_allKeys.concat(theseKeys);
      if (_static_key1_11_allKeys.length > 0) {
        static_key1_11.keys = _static_key1_11_allKeys[_static_key1_11_allKeys.length - 1].name;  // just the last key pressed
        static_key1_11.rt = _static_key1_11_allKeys[_static_key1_11_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_26* updates
    if (t >= 0.0 && static_instruction_26.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_26.tStart = t;  // (not accounting for frame time here)
      static_instruction_26.frameNStart = frameN;  // exact frame index
      
      static_instruction_26.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instructions_4to5Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instructions_4to5RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instructions_4to5'-------
    for (const thisComponent of instructions_4to5Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instructions_4to5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_8_allKeys;
var instruction_5Components;
function instruction_5RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_5'-------
    t = 0;
    instruction_5Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1_8.keys = undefined;
    static_key1_8.rt = undefined;
    _static_key1_8_allKeys = [];
    polygon_20.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    pointer_dot_0.setPos([(((x_y_gains_per_angle[1][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale), (((x_y_gains_per_angle[1][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale)]);
    pointer_dot_0.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_1.setPos([((((x_y_gains_per_angle[1][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale) * 0.66667), ((((x_y_gains_per_angle[1][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale) * 0.66667)]);
    pointer_dot_1.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_2.setPos([((((x_y_gains_per_angle[1][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale) * 0.33333), ((((x_y_gains_per_angle[1][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale) * 0.33333)]);
    pointer_dot_2.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    // keep track of which components have finished
    instruction_5Components = [];
    instruction_5Components.push(static_instruction_19);
    instruction_5Components.push(static_key1_8);
    instruction_5Components.push(static_instruction_20);
    instruction_5Components.push(polygon_20);
    instruction_5Components.push(pointer_dot_0);
    instruction_5Components.push(pointer_dot_1);
    instruction_5Components.push(pointer_dot_2);
    
    for (const thisComponent of instruction_5Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_5RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_5'-------
    // get current time
    t = instruction_5Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_19* updates
    if (t >= 0.0 && static_instruction_19.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_19.tStart = t;  // (not accounting for frame time here)
      static_instruction_19.frameNStart = frameN;  // exact frame index
      
      static_instruction_19.setAutoDraw(true);
    }

    
    // *static_key1_8* updates
    if (t >= 0.0 && static_key1_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_8.tStart = t;  // (not accounting for frame time here)
      static_key1_8.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_8.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_8.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_8.clearEvents(); });
    }

    if (static_key1_8.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_8.getKeys({keyList: [], waitRelease: false});
      _static_key1_8_allKeys = _static_key1_8_allKeys.concat(theseKeys);
      if (_static_key1_8_allKeys.length > 0) {
        static_key1_8.keys = _static_key1_8_allKeys[_static_key1_8_allKeys.length - 1].name;  // just the last key pressed
        static_key1_8.rt = _static_key1_8_allKeys[_static_key1_8_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_20* updates
    if (t >= 0.0 && static_instruction_20.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_20.tStart = t;  // (not accounting for frame time here)
      static_instruction_20.frameNStart = frameN;  // exact frame index
      
      static_instruction_20.setAutoDraw(true);
    }

    
    // *polygon_20* updates
    if (t >= 0.0 && polygon_20.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_20.tStart = t;  // (not accounting for frame time here)
      polygon_20.frameNStart = frameN;  // exact frame index
      
      polygon_20.setAutoDraw(true);
    }

    
    // *pointer_dot_0* updates
    if (t >= 0.0 && pointer_dot_0.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_0.tStart = t;  // (not accounting for frame time here)
      pointer_dot_0.frameNStart = frameN;  // exact frame index
      
      pointer_dot_0.setAutoDraw(true);
    }

    
    // *pointer_dot_1* updates
    if (t >= 0.0 && pointer_dot_1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_1.tStart = t;  // (not accounting for frame time here)
      pointer_dot_1.frameNStart = frameN;  // exact frame index
      
      pointer_dot_1.setAutoDraw(true);
    }

    
    // *pointer_dot_2* updates
    if (t >= 0.0 && pointer_dot_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_2.tStart = t;  // (not accounting for frame time here)
      pointer_dot_2.frameNStart = frameN;  // exact frame index
      
      pointer_dot_2.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_5Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_5RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_5'-------
    for (const thisComponent of instruction_5Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_12_allKeys;
var instruction_6Components;
function instruction_6RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_6'-------
    t = 0;
    instruction_6Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1_12.keys = undefined;
    static_key1_12.rt = undefined;
    _static_key1_12_allKeys = [];
    polygon_24.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_6Components = [];
    instruction_6Components.push(static_instruction_27);
    instruction_6Components.push(static_key1_12);
    instruction_6Components.push(static_instruction_28);
    instruction_6Components.push(polygon_24);
    
    for (const thisComponent of instruction_6Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_6RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_6'-------
    // get current time
    t = instruction_6Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_27* updates
    if (t >= 0.0 && static_instruction_27.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_27.tStart = t;  // (not accounting for frame time here)
      static_instruction_27.frameNStart = frameN;  // exact frame index
      
      static_instruction_27.setAutoDraw(true);
    }

    
    // *static_key1_12* updates
    if (t >= 0.0 && static_key1_12.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_12.tStart = t;  // (not accounting for frame time here)
      static_key1_12.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_12.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_12.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_12.clearEvents(); });
    }

    if (static_key1_12.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_12.getKeys({keyList: [], waitRelease: false});
      _static_key1_12_allKeys = _static_key1_12_allKeys.concat(theseKeys);
      if (_static_key1_12_allKeys.length > 0) {
        static_key1_12.keys = _static_key1_12_allKeys[_static_key1_12_allKeys.length - 1].name;  // just the last key pressed
        static_key1_12.rt = _static_key1_12_allKeys[_static_key1_12_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_28* updates
    if (t >= 0.0 && static_instruction_28.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_28.tStart = t;  // (not accounting for frame time here)
      static_instruction_28.frameNStart = frameN;  // exact frame index
      
      static_instruction_28.setAutoDraw(true);
    }

    
    // *polygon_24* updates
    if (t >= 0.0 && polygon_24.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_24.tStart = t;  // (not accounting for frame time here)
      polygon_24.frameNStart = frameN;  // exact frame index
      
      polygon_24.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_6Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_6RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_6'-------
    for (const thisComponent of instruction_6Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_17_allKeys;
var instruction_67Components;
function instruction_67RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_67'-------
    t = 0;
    instruction_67Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_instruction_37.setText(key_mapping_instruction_text);
    static_key1_17.keys = undefined;
    static_key1_17.rt = undefined;
    _static_key1_17_allKeys = [];
    SD_image.setImage(key_mapping_instruction_image);
    // keep track of which components have finished
    instruction_67Components = [];
    instruction_67Components.push(static_instruction_37);
    instruction_67Components.push(static_key1_17);
    instruction_67Components.push(static_instruction_38);
    instruction_67Components.push(SD_image);
    
    for (const thisComponent of instruction_67Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_67RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_67'-------
    // get current time
    t = instruction_67Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_37* updates
    if (t >= 0.0 && static_instruction_37.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_37.tStart = t;  // (not accounting for frame time here)
      static_instruction_37.frameNStart = frameN;  // exact frame index
      
      static_instruction_37.setAutoDraw(true);
    }

    
    // *static_key1_17* updates
    if (t >= 0.0 && static_key1_17.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_17.tStart = t;  // (not accounting for frame time here)
      static_key1_17.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_17.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_17.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_17.clearEvents(); });
    }

    if (static_key1_17.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_17.getKeys({keyList: [], waitRelease: false});
      _static_key1_17_allKeys = _static_key1_17_allKeys.concat(theseKeys);
      if (_static_key1_17_allKeys.length > 0) {
        static_key1_17.keys = _static_key1_17_allKeys[_static_key1_17_allKeys.length - 1].name;  // just the last key pressed
        static_key1_17.rt = _static_key1_17_allKeys[_static_key1_17_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_38* updates
    if (t >= 0.0 && static_instruction_38.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_38.tStart = t;  // (not accounting for frame time here)
      static_instruction_38.frameNStart = frameN;  // exact frame index
      
      static_instruction_38.setAutoDraw(true);
    }

    
    // *SD_image* updates
    if (t >= 0.0 && SD_image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      SD_image.tStart = t;  // (not accounting for frame time here)
      SD_image.frameNStart = frameN;  // exact frame index
      
      SD_image.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_67Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_67RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_67'-------
    for (const thisComponent of instruction_67Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_67" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_18_allKeys;
var instruction_67bComponents;
function instruction_67bRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_67b'-------
    t = 0;
    instruction_67bClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1_18.keys = undefined;
    static_key1_18.rt = undefined;
    _static_key1_18_allKeys = [];
    polygon_37.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_67bComponents = [];
    instruction_67bComponents.push(static_instruction_39);
    instruction_67bComponents.push(static_key1_18);
    instruction_67bComponents.push(static_instruction_40);
    instruction_67bComponents.push(polygon_37);
    
    for (const thisComponent of instruction_67bComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_67bRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_67b'-------
    // get current time
    t = instruction_67bClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_39* updates
    if (t >= 0.0 && static_instruction_39.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_39.tStart = t;  // (not accounting for frame time here)
      static_instruction_39.frameNStart = frameN;  // exact frame index
      
      static_instruction_39.setAutoDraw(true);
    }

    
    // *static_key1_18* updates
    if (t >= 0.0 && static_key1_18.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_18.tStart = t;  // (not accounting for frame time here)
      static_key1_18.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_18.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_18.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_18.clearEvents(); });
    }

    if (static_key1_18.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_18.getKeys({keyList: [], waitRelease: false});
      _static_key1_18_allKeys = _static_key1_18_allKeys.concat(theseKeys);
      if (_static_key1_18_allKeys.length > 0) {
        static_key1_18.keys = _static_key1_18_allKeys[_static_key1_18_allKeys.length - 1].name;  // just the last key pressed
        static_key1_18.rt = _static_key1_18_allKeys[_static_key1_18_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_40* updates
    if (t >= 0.0 && static_instruction_40.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_40.tStart = t;  // (not accounting for frame time here)
      static_instruction_40.frameNStart = frameN;  // exact frame index
      
      static_instruction_40.setAutoDraw(true);
    }

    
    // *polygon_37* updates
    if (t >= 0.0 && polygon_37.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_37.tStart = t;  // (not accounting for frame time here)
      polygon_37.frameNStart = frameN;  // exact frame index
      
      polygon_37.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_67bComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_67bRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_67b'-------
    for (const thisComponent of instruction_67bComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_67b" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_14_allKeys;
var instruction_7Components;
function instruction_7RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_7'-------
    t = 0;
    instruction_7Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    static_key1_14.keys = undefined;
    static_key1_14.rt = undefined;
    _static_key1_14_allKeys = [];
    polygon_29.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_7Components = [];
    instruction_7Components.push(static_instruction_31);
    instruction_7Components.push(static_key1_14);
    instruction_7Components.push(static_instruction_32);
    instruction_7Components.push(polygon_29);
    
    for (const thisComponent of instruction_7Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_7RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_7'-------
    // get current time
    t = instruction_7Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_31* updates
    if (t >= 0.0 && static_instruction_31.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_31.tStart = t;  // (not accounting for frame time here)
      static_instruction_31.frameNStart = frameN;  // exact frame index
      
      static_instruction_31.setAutoDraw(true);
    }

    
    // *static_key1_14* updates
    if (t >= 0.0 && static_key1_14.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_14.tStart = t;  // (not accounting for frame time here)
      static_key1_14.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_14.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_14.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_14.clearEvents(); });
    }

    if (static_key1_14.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_14.getKeys({keyList: [], waitRelease: false});
      _static_key1_14_allKeys = _static_key1_14_allKeys.concat(theseKeys);
      if (_static_key1_14_allKeys.length > 0) {
        static_key1_14.keys = _static_key1_14_allKeys[_static_key1_14_allKeys.length - 1].name;  // just the last key pressed
        static_key1_14.rt = _static_key1_14_allKeys[_static_key1_14_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_32* updates
    if (t >= 0.0 && static_instruction_32.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_32.tStart = t;  // (not accounting for frame time here)
      static_instruction_32.frameNStart = frameN;  // exact frame index
      
      static_instruction_32.setAutoDraw(true);
    }

    
    // *polygon_29* updates
    if (t >= 0.0 && polygon_29.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_29.tStart = t;  // (not accounting for frame time here)
      polygon_29.frameNStart = frameN;  // exact frame index
      
      polygon_29.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_7Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_7RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_7'-------
    for (const thisComponent of instruction_7Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var trials_2;
var currentLoop;
function trials_2LoopBegin(trials_2LoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_2 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: permutation_data_file_string_practice,
    seed: undefined, name: 'trials_2'
  });
  psychoJS.experiment.addLoop(trials_2); // add the loop to the experiment
  currentLoop = trials_2;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_2 of trials_2) {
    const snapshot = trials_2.getSnapshot();
    trials_2LoopScheduler.add(importConditions(snapshot));
    trials_2LoopScheduler.add(load_practice_trialsRoutineBegin(snapshot));
    trials_2LoopScheduler.add(load_practice_trialsRoutineEachFrame(snapshot));
    trials_2LoopScheduler.add(load_practice_trialsRoutineEnd(snapshot));
    trials_2LoopScheduler.add(endLoopIteration(trials_2LoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_2LoopEnd() {
  psychoJS.experiment.removeLoop(trials_2);

  return Scheduler.Event.NEXT;
}


var trials;
function trialsLoopBegin(trialsLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials = new TrialHandler({
    psychoJS: psychoJS,
    nReps: num_practice_trials, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'trials'
  });
  psychoJS.experiment.addLoop(trials); // add the loop to the experiment
  currentLoop = trials;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial of trials) {
    const snapshot = trials.getSnapshot();
    trialsLoopScheduler.add(importConditions(snapshot));
    trialsLoopScheduler.add(get_ready_practiceRoutineBegin(snapshot));
    trialsLoopScheduler.add(get_ready_practiceRoutineEachFrame(snapshot));
    trialsLoopScheduler.add(get_ready_practiceRoutineEnd(snapshot));
    trialsLoopScheduler.add(blank_at_trial_startRoutineBegin(snapshot));
    trialsLoopScheduler.add(blank_at_trial_startRoutineEachFrame(snapshot));
    trialsLoopScheduler.add(blank_at_trial_startRoutineEnd(snapshot));
    trialsLoopScheduler.add(practice_mem_displayRoutineBegin(snapshot));
    trialsLoopScheduler.add(practice_mem_displayRoutineEachFrame(snapshot));
    trialsLoopScheduler.add(practice_mem_displayRoutineEnd(snapshot));
    trialsLoopScheduler.add(retention_cue_and_test_intval_practice_2RoutineBegin(snapshot));
    trialsLoopScheduler.add(retention_cue_and_test_intval_practice_2RoutineEachFrame(snapshot));
    trialsLoopScheduler.add(retention_cue_and_test_intval_practice_2RoutineEnd(snapshot));
    trialsLoopScheduler.add(confidence_rating_practiceRoutineBegin(snapshot));
    trialsLoopScheduler.add(confidence_rating_practiceRoutineEachFrame(snapshot));
    trialsLoopScheduler.add(confidence_rating_practiceRoutineEnd(snapshot));
    trialsLoopScheduler.add(endLoopIteration(trialsLoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trialsLoopEnd() {
  psychoJS.experiment.removeLoop(trials);

  return Scheduler.Event.NEXT;
}


var trials_4;
function trials_4LoopBegin(trials_4LoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_4 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: permutation_data_file_string_practice,
    seed: undefined, name: 'trials_4'
  });
  psychoJS.experiment.addLoop(trials_4); // add the loop to the experiment
  currentLoop = trials_4;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_4 of trials_4) {
    const snapshot = trials_4.getSnapshot();
    trials_4LoopScheduler.add(importConditions(snapshot));
    trials_4LoopScheduler.add(load_training_trialsRoutineBegin(snapshot));
    trials_4LoopScheduler.add(load_training_trialsRoutineEachFrame(snapshot));
    trials_4LoopScheduler.add(load_training_trialsRoutineEnd(snapshot));
    trials_4LoopScheduler.add(endLoopIteration(trials_4LoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_4LoopEnd() {
  psychoJS.experiment.removeLoop(trials_4);

  return Scheduler.Event.NEXT;
}


var trials_10;
function trials_10LoopBegin(trials_10LoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_10 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: max_num_training_trials, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'trials_10'
  });
  psychoJS.experiment.addLoop(trials_10); // add the loop to the experiment
  currentLoop = trials_10;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_10 of trials_10) {
    const snapshot = trials_10.getSnapshot();
    trials_10LoopScheduler.add(importConditions(snapshot));
    trials_10LoopScheduler.add(get_ready_training_blockstartRoutineBegin(snapshot));
    trials_10LoopScheduler.add(get_ready_training_blockstartRoutineEachFrame(snapshot));
    trials_10LoopScheduler.add(get_ready_training_blockstartRoutineEnd(snapshot));
    trials_10LoopScheduler.add(blank_at_trial_startRoutineBegin(snapshot));
    trials_10LoopScheduler.add(blank_at_trial_startRoutineEachFrame(snapshot));
    trials_10LoopScheduler.add(blank_at_trial_startRoutineEnd(snapshot));
    trials_10LoopScheduler.add(training_mem_displayRoutineBegin(snapshot));
    trials_10LoopScheduler.add(training_mem_displayRoutineEachFrame(snapshot));
    trials_10LoopScheduler.add(training_mem_displayRoutineEnd(snapshot));
    trials_10LoopScheduler.add(retention_cue_and_test_intval_trainingRoutineBegin(snapshot));
    trials_10LoopScheduler.add(retention_cue_and_test_intval_trainingRoutineEachFrame(snapshot));
    trials_10LoopScheduler.add(retention_cue_and_test_intval_trainingRoutineEnd(snapshot));
    trials_10LoopScheduler.add(confidence_rating_trainingRoutineBegin(snapshot));
    trials_10LoopScheduler.add(confidence_rating_trainingRoutineEachFrame(snapshot));
    trials_10LoopScheduler.add(confidence_rating_trainingRoutineEnd(snapshot));
    trials_10LoopScheduler.add(endLoopIteration(trials_10LoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_10LoopEnd() {
  psychoJS.experiment.removeLoop(trials_10);

  return Scheduler.Event.NEXT;
}


var main_exp_blocks;
function main_exp_blocksLoopBegin(main_exp_blocksLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  main_exp_blocks = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'main_exp_condFiles.xlsx',
    seed: undefined, name: 'main_exp_blocks'
  });
  psychoJS.experiment.addLoop(main_exp_blocks); // add the loop to the experiment
  currentLoop = main_exp_blocks;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisMain_exp_block of main_exp_blocks) {
    const snapshot = main_exp_blocks.getSnapshot();
    main_exp_blocksLoopScheduler.add(importConditions(snapshot));
    const trials_7LoopScheduler = new Scheduler(psychoJS);
    main_exp_blocksLoopScheduler.add(trials_7LoopBegin, trials_7LoopScheduler);
    main_exp_blocksLoopScheduler.add(trials_7LoopScheduler);
    main_exp_blocksLoopScheduler.add(trials_7LoopEnd);
    const trials_8LoopScheduler = new Scheduler(psychoJS);
    main_exp_blocksLoopScheduler.add(trials_8LoopBegin, trials_8LoopScheduler);
    main_exp_blocksLoopScheduler.add(trials_8LoopScheduler);
    main_exp_blocksLoopScheduler.add(trials_8LoopEnd);
    main_exp_blocksLoopScheduler.add(endLoopIteration(main_exp_blocksLoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


var trials_7;
function trials_7LoopBegin(trials_7LoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_7 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: condFile,
    seed: undefined, name: 'trials_7'
  });
  psychoJS.experiment.addLoop(trials_7); // add the loop to the experiment
  currentLoop = trials_7;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_7 of trials_7) {
    const snapshot = trials_7.getSnapshot();
    trials_7LoopScheduler.add(importConditions(snapshot));
    trials_7LoopScheduler.add(load_real_trialsRoutineBegin(snapshot));
    trials_7LoopScheduler.add(load_real_trialsRoutineEachFrame(snapshot));
    trials_7LoopScheduler.add(load_real_trialsRoutineEnd(snapshot));
    trials_7LoopScheduler.add(endLoopIteration(trials_7LoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_7LoopEnd() {
  psychoJS.experiment.removeLoop(trials_7);

  return Scheduler.Event.NEXT;
}


var trials_8;
function trials_8LoopBegin(trials_8LoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_8 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: trials_per_block, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'trials_8'
  });
  psychoJS.experiment.addLoop(trials_8); // add the loop to the experiment
  currentLoop = trials_8;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_8 of trials_8) {
    const snapshot = trials_8.getSnapshot();
    trials_8LoopScheduler.add(importConditions(snapshot));
    trials_8LoopScheduler.add(get_ready_real_blockstartRoutineBegin(snapshot));
    trials_8LoopScheduler.add(get_ready_real_blockstartRoutineEachFrame(snapshot));
    trials_8LoopScheduler.add(get_ready_real_blockstartRoutineEnd(snapshot));
    trials_8LoopScheduler.add(blank_at_trial_startRoutineBegin(snapshot));
    trials_8LoopScheduler.add(blank_at_trial_startRoutineEachFrame(snapshot));
    trials_8LoopScheduler.add(blank_at_trial_startRoutineEnd(snapshot));
    trials_8LoopScheduler.add(real_mem_displayRoutineBegin(snapshot));
    trials_8LoopScheduler.add(real_mem_displayRoutineEachFrame(snapshot));
    trials_8LoopScheduler.add(real_mem_displayRoutineEnd(snapshot));
    trials_8LoopScheduler.add(retention_cue_and_test_intval_realRoutineBegin(snapshot));
    trials_8LoopScheduler.add(retention_cue_and_test_intval_realRoutineEachFrame(snapshot));
    trials_8LoopScheduler.add(retention_cue_and_test_intval_realRoutineEnd(snapshot));
    trials_8LoopScheduler.add(confidence_rating_real_2RoutineBegin(snapshot));
    trials_8LoopScheduler.add(confidence_rating_real_2RoutineEachFrame(snapshot));
    trials_8LoopScheduler.add(confidence_rating_real_2RoutineEnd(snapshot));
    trials_8LoopScheduler.add(endLoopIteration(trials_8LoopScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_8LoopEnd() {
  psychoJS.experiment.removeLoop(trials_8);

  return Scheduler.Event.NEXT;
}


function main_exp_blocksLoopEnd() {
  psychoJS.experiment.removeLoop(main_exp_blocks);

  return Scheduler.Event.NEXT;
}


var load_practice_trialsComponents;
function load_practice_trialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'load_practice_trials'-------
    t = 0;
    load_practice_trialsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    pylink.getEYELINK().setOfflineMode();
    pylink.pumpDelay(100);
    practice_trial_image_list.append([Used_image_1, Used_image_2, Used_image_3, Used_image_4, Used_image_5, Used_image_6, Used_image_7, Used_image_8, Unused_image_1, Unused_image_2]);
    if (((practice_trial_load_number + 1) <= (num_practice_trials / 2))) {
        retro_0_post_1_practice.append(0);
        change_0_1_practice.append(0);
    } else {
        retro_0_post_1_practice.append(1);
        change_0_1_practice.append(1);
    }
    mem_display_to_cue_times_practice.append(mem_display_to_cue_time_s);
    
    // keep track of which components have finished
    load_practice_trialsComponents = [];
    
    for (const thisComponent of load_practice_trialsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function load_practice_trialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'load_practice_trials'-------
    // get current time
    t = load_practice_trialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of load_practice_trialsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var practice_trial_index;
var confidence_onset_time_s;
var confidence_image_onset_time_s;
function load_practice_trialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'load_practice_trials'-------
    for (const thisComponent of load_practice_trialsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    practice_trial_load_number = (practice_trial_load_number + 1);
    if ((practice_trial_load_number >= num_practice_trials)) {
        trials_2.finished = true;
        shuffle(retro_0_post_1_practice);
        shuffle(mem_display_to_cue_times_practice);
        shuffle(change_0_1_practice);
        practice_trial_index = 0;
        confidence_onset_time_s = 0.0;
        confidence_image_onset_time_s = 0.0;
    }
    
    // the Routine "load_practice_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var practice_trial_text;
var _static_key1_13_allKeys;
var get_ready_practiceComponents;
function get_ready_practiceRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'get_ready_practice'-------
    t = 0;
    get_ready_practiceClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if ((practice_trial_index === 0)) {
        practice_trial_text = (("Get ready for practice trial 1 out of " + num_practice_trials.toString()) + ".\n\nPlease don't forget to keep your eyes on the dot in the middle, and to use your right hand to indicate 'Same' or 'Different' using your index and middle finger, respectively.");
    } else {
        if ((practice_trial_index >= (num_practice_trials / 2.0))) {
            practice_trial_text = ((("Get ready for practice trial " + (practice_trial_index + 1).toString()) + " out of ") + num_practice_trials.toString());
        } else {
            practice_trial_text = (((("Get ready for practice trial " + (practice_trial_index + 1).toString()) + " out of ") + num_practice_trials.toString()) + ".\n\nPlease don't forget to keep your eyes on the dot in the middle, and to use your right hand to indicate 'Same' or 'Different' with your index and middle finger, respectively, and then your left hand to indicate your confidence in that choice (ring finger=certain, middle finger=doubt, index finger=guess).");
        }
    }
    static_instruction_29.setText(practice_trial_text, false);
    
    static_instruction_29.setText(practice_trial_text);
    static_key1_13.keys = undefined;
    static_key1_13.rt = undefined;
    _static_key1_13_allKeys = [];
    polygon_25.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    document.body.style.cursor='none';
    // keep track of which components have finished
    get_ready_practiceComponents = [];
    get_ready_practiceComponents.push(static_instruction_29);
    get_ready_practiceComponents.push(static_key1_13);
    get_ready_practiceComponents.push(static_instruction_30);
    get_ready_practiceComponents.push(polygon_25);
    
    for (const thisComponent of get_ready_practiceComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function get_ready_practiceRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'get_ready_practice'-------
    // get current time
    t = get_ready_practiceClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_29* updates
    if (t >= 0.0 && static_instruction_29.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_29.tStart = t;  // (not accounting for frame time here)
      static_instruction_29.frameNStart = frameN;  // exact frame index
      
      static_instruction_29.setAutoDraw(true);
    }

    
    // *static_key1_13* updates
    if (t >= 0.0 && static_key1_13.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_13.tStart = t;  // (not accounting for frame time here)
      static_key1_13.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_13.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_13.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_13.clearEvents(); });
    }

    if (static_key1_13.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_13.getKeys({keyList: [], waitRelease: false});
      _static_key1_13_allKeys = _static_key1_13_allKeys.concat(theseKeys);
      if (_static_key1_13_allKeys.length > 0) {
        static_key1_13.keys = _static_key1_13_allKeys[_static_key1_13_allKeys.length - 1].name;  // just the last key pressed
        static_key1_13.rt = _static_key1_13_allKeys[_static_key1_13_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_30* updates
    if (t >= 0.0 && static_instruction_30.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_30.tStart = t;  // (not accounting for frame time here)
      static_instruction_30.frameNStart = frameN;  // exact frame index
      
      static_instruction_30.setAutoDraw(true);
    }

    
    // *polygon_25* updates
    if (t >= 0.0 && polygon_25.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_25.tStart = t;  // (not accounting for frame time here)
      polygon_25.frameNStart = frameN;  // exact frame index
      
      polygon_25.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of get_ready_practiceComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function get_ready_practiceRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'get_ready_practice'-------
    for (const thisComponent of get_ready_practiceComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "get_ready_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var blank_at_trial_startComponents;
function blank_at_trial_startRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'blank_at_trial_start'-------
    t = 0;
    blank_at_trial_startClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    polygon_39.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    blank_at_trial_startComponents = [];
    blank_at_trial_startComponents.push(polygon_39);
    
    for (const thisComponent of blank_at_trial_startComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function blank_at_trial_startRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'blank_at_trial_start'-------
    // get current time
    t = blank_at_trial_startClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *polygon_39* updates
    if (t >= 0.0 && polygon_39.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_39.tStart = t;  // (not accounting for frame time here)
      polygon_39.frameNStart = frameN;  // exact frame index
      
      polygon_39.setAutoDraw(true);
    }

    frameRemains = 0.0 + blank_trial_start_dur_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((polygon_39.status === PsychoJS.Status.STARTED || polygon_39.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      polygon_39.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of blank_at_trial_startComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function blank_at_trial_startRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'blank_at_trial_start'-------
    for (const thisComponent of blank_at_trial_startComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "blank_at_trial_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var Image_1;
var Image_2;
var Image_3;
var Image_4;
var Image_5;
var Image_6;
var Image_7;
var Image_8;
var practice_mem_displayComponents;
function practice_mem_displayRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'practice_mem_display'-------
    t = 0;
    practice_mem_displayClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    Image_1 = all_pic_names_training[practice_trial_image_list[practice_trial_index][0]];
    array_image_1d.setImage(Image_1, false);
    array_image_1d.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names_training[practice_trial_image_list[practice_trial_index][1]];
    array_image_2d.setImage(Image_2, false);
    array_image_2d.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names_training[practice_trial_image_list[practice_trial_index][2]];
    array_image_3d.setImage(Image_3, false);
    array_image_3d.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names_training[practice_trial_image_list[practice_trial_index][3]];
    array_image_4d.setImage(Image_4, false);
    array_image_4d.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names_training[practice_trial_image_list[practice_trial_index][4]];
    array_image_5d.setImage(Image_5, false);
    array_image_5d.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names_training[practice_trial_image_list[practice_trial_index][5]];
    array_image_6d.setImage(Image_6, false);
    array_image_6d.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names_training[practice_trial_image_list[practice_trial_index][6]];
    array_image_7d.setImage(Image_7, false);
    array_image_7d.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names_training[practice_trial_image_list[practice_trial_index][7]];
    array_image_8d.setImage(Image_8, false);
    array_image_8d.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    thisExp.addData("memDisplayOnTimePractice", timer.getTime());
    for (var image_index = 0, _pj_a = 8; (image_index < _pj_a); image_index += 1) {
        thisExp.addData((("mem_image_" + image_index.toString()) + "_practice"), practice_trial_image_list[practice_trial_index][image_index]);
    }
    
    array_image_1d.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1d.setImage(Image_1);
    array_image_2d.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2d.setImage(Image_2);
    array_image_3d.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3d.setImage(Image_3);
    array_image_4d.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4d.setImage(Image_4);
    array_image_5d.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5d.setImage(Image_5);
    array_image_6d.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6d.setImage(Image_6);
    array_image_7d.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7d.setImage(Image_7);
    array_image_8d.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8d.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8d.setImage(Image_8);
    polygon_26.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    practice_mem_displayComponents = [];
    practice_mem_displayComponents.push(array_image_1d);
    practice_mem_displayComponents.push(array_image_2d);
    practice_mem_displayComponents.push(array_image_3d);
    practice_mem_displayComponents.push(array_image_4d);
    practice_mem_displayComponents.push(array_image_5d);
    practice_mem_displayComponents.push(array_image_6d);
    practice_mem_displayComponents.push(array_image_7d);
    practice_mem_displayComponents.push(array_image_8d);
    practice_mem_displayComponents.push(polygon_26);
    
    for (const thisComponent of practice_mem_displayComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function practice_mem_displayRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'practice_mem_display'-------
    // get current time
    t = practice_mem_displayClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *array_image_1d* updates
    if (t >= 0.0 && array_image_1d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1d.tStart = t;  // (not accounting for frame time here)
      array_image_1d.frameNStart = frameN;  // exact frame index
      
      array_image_1d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_1d.status === PsychoJS.Status.STARTED || array_image_1d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_1d.setAutoDraw(false);
    }
    
    // *array_image_2d* updates
    if (t >= 0.0 && array_image_2d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2d.tStart = t;  // (not accounting for frame time here)
      array_image_2d.frameNStart = frameN;  // exact frame index
      
      array_image_2d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_2d.status === PsychoJS.Status.STARTED || array_image_2d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_2d.setAutoDraw(false);
    }
    
    // *array_image_3d* updates
    if (t >= 0.0 && array_image_3d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3d.tStart = t;  // (not accounting for frame time here)
      array_image_3d.frameNStart = frameN;  // exact frame index
      
      array_image_3d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_3d.status === PsychoJS.Status.STARTED || array_image_3d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_3d.setAutoDraw(false);
    }
    
    // *array_image_4d* updates
    if (t >= 0.0 && array_image_4d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4d.tStart = t;  // (not accounting for frame time here)
      array_image_4d.frameNStart = frameN;  // exact frame index
      
      array_image_4d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_4d.status === PsychoJS.Status.STARTED || array_image_4d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_4d.setAutoDraw(false);
    }
    
    // *array_image_5d* updates
    if (t >= 0.0 && array_image_5d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5d.tStart = t;  // (not accounting for frame time here)
      array_image_5d.frameNStart = frameN;  // exact frame index
      
      array_image_5d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_5d.status === PsychoJS.Status.STARTED || array_image_5d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_5d.setAutoDraw(false);
    }
    
    // *array_image_6d* updates
    if (t >= 0.0 && array_image_6d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6d.tStart = t;  // (not accounting for frame time here)
      array_image_6d.frameNStart = frameN;  // exact frame index
      
      array_image_6d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_6d.status === PsychoJS.Status.STARTED || array_image_6d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_6d.setAutoDraw(false);
    }
    
    // *array_image_7d* updates
    if (t >= 0.0 && array_image_7d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7d.tStart = t;  // (not accounting for frame time here)
      array_image_7d.frameNStart = frameN;  // exact frame index
      
      array_image_7d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_7d.status === PsychoJS.Status.STARTED || array_image_7d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_7d.setAutoDraw(false);
    }
    
    // *array_image_8d* updates
    if (t >= 0.0 && array_image_8d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8d.tStart = t;  // (not accounting for frame time here)
      array_image_8d.frameNStart = frameN;  // exact frame index
      
      array_image_8d.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_8d.status === PsychoJS.Status.STARTED || array_image_8d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_8d.setAutoDraw(false);
    }
    
    // *polygon_26* updates
    if (t >= 0.0 && polygon_26.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_26.tStart = t;  // (not accounting for frame time here)
      polygon_26.frameNStart = frameN;  // exact frame index
      
      polygon_26.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((polygon_26.status === PsychoJS.Status.STARTED || polygon_26.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      polygon_26.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of practice_mem_displayComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var mem_display_to_cue_duration_s;
var mem_display_to_test_array_duration_s;
var routine_duration_s;
var response_time_s;
var change_from_index_practice;
var image_index_list_in_order;
var change_or_no_practice;
var change_to_index_practice;
function practice_mem_displayRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'practice_mem_display'-------
    for (const thisComponent of practice_mem_displayComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    
            // add-on: list(s: string): string[]
            function list(s) {
                // if s is a string, we return a list of its characters
                if (typeof s === 'string')
                    return s.split('');
                else
                    // otherwise we return s:
                    return s;
            }
    
            mem_display_to_cue_duration_s = mem_display_to_cue_times_practice[practice_trial_index];
    if (retro_0_post_1_practice[practice_trial_index]) {
        mem_display_to_test_array_duration_s = (mem_display_to_cue_duration_s - 0.1);
        routine_duration_s = (mem_display_to_cue_duration_s + cue_duration_s);
        response_time_s = (routine_duration_s - cue_duration_s);
    } else {
        mem_display_to_test_array_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        routine_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        response_time_s = routine_duration_s;
    }
    shuffle(image_index_list_for_shuffling);
    change_from_index_practice = image_index_list_for_shuffling[0];
    image_index_list_in_order = list(range(0, num_images_per_display));
    change_or_no_practice = change_0_1_practice[practice_trial_index];
    if (change_or_no_practice) {
        change_to_index_practice = num_images_per_display;
        image_index_list_in_order[change_from_index_practice] = change_to_index_practice;
    }
    thisExp.addData("retention_interval_practice", mem_display_to_cue_duration_s);
    thisExp.addData("retro_0_post_1_practice", retro_0_post_1_practice[practice_trial_index]);
    thisExp.addData("change_or_no_practice", change_or_no_practice);
    thisExp.addData("cue_location_practice", change_from_index_practice);
    thisExp.addData("mem_display_to_test_array_duration_s", mem_display_to_test_array_duration_s);
    thisExp.addData("mem_display_to_cue_duration_s", mem_display_to_cue_duration_s);
    thisExp.addData("routine_duration_s", routine_duration_s);
    
    // the Routine "practice_mem_display" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var x_pointer_0;
var y_pointer_0;
var x_pointer_1;
var y_pointer_1;
var x_pointer_2;
var y_pointer_2;
var _sd_practice_resp_allKeys;
var retention_cue_and_test_intval_practice_2Components;
function retention_cue_and_test_intval_practice_2RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'retention_cue_and_test_intval_practice_2'-------
    t = 0;
    retention_cue_and_test_intval_practice_2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    Image_1 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[0]]];
    array_image_1f.setImage(Image_1, false);
    array_image_1f.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[1]]];
    array_image_2f.setImage(Image_2, false);
    array_image_2f.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[2]]];
    array_image_3f.setImage(Image_3, false);
    array_image_3f.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[3]]];
    array_image_4f.setImage(Image_4, false);
    array_image_4f.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[4]]];
    array_image_5f.setImage(Image_5, false);
    array_image_5f.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[5]]];
    array_image_6f.setImage(Image_6, false);
    array_image_6f.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[6]]];
    array_image_7f.setImage(Image_7, false);
    array_image_7f.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names_training[practice_trial_image_list[practice_trial_index][image_index_list_in_order[7]]];
    array_image_8f.setImage(Image_8, false);
    array_image_8f.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    x_pointer_0 = (((x_y_gains_per_angle[change_from_index_practice][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale);
    y_pointer_0 = (((x_y_gains_per_angle[change_from_index_practice][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale);
    pointer_dot_0b.setPos([x_pointer_0, y_pointer_0], false);
    x_pointer_1 = (x_pointer_0 * 0.66667);
    y_pointer_1 = (y_pointer_0 * 0.66667);
    pointer_dot_1b.setPos([x_pointer_1, y_pointer_1], false);
    x_pointer_2 = (x_pointer_0 * 0.33333);
    y_pointer_2 = (y_pointer_0 * 0.33333);
    pointer_dot_2b.setPos([x_pointer_2, y_pointer_2], false);
    
    polygon_28.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    array_image_1f.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1f.setImage(Image_1);
    array_image_2f.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2f.setImage(Image_2);
    array_image_3f.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3f.setImage(Image_3);
    array_image_4f.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4f.setImage(Image_4);
    array_image_5f.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5f.setImage(Image_5);
    array_image_6f.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6f.setImage(Image_6);
    array_image_7f.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7f.setImage(Image_7);
    array_image_8f.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8f.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8f.setImage(Image_8);
    pointer_dot_0b.setPos([x_pointer_0, y_pointer_0]);
    pointer_dot_0b.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_1b.setPos([x_pointer_1, y_pointer_1]);
    pointer_dot_1b.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_2b.setPos([x_pointer_2, y_pointer_2]);
    pointer_dot_2b.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    sd_practice_resp.keys = undefined;
    sd_practice_resp.rt = undefined;
    _sd_practice_resp_allKeys = [];
    // keep track of which components have finished
    retention_cue_and_test_intval_practice_2Components = [];
    retention_cue_and_test_intval_practice_2Components.push(polygon_28);
    retention_cue_and_test_intval_practice_2Components.push(array_image_1f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_2f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_3f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_4f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_5f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_6f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_7f);
    retention_cue_and_test_intval_practice_2Components.push(array_image_8f);
    retention_cue_and_test_intval_practice_2Components.push(pointer_dot_0b);
    retention_cue_and_test_intval_practice_2Components.push(pointer_dot_1b);
    retention_cue_and_test_intval_practice_2Components.push(pointer_dot_2b);
    retention_cue_and_test_intval_practice_2Components.push(sd_practice_resp);
    
    for (const thisComponent of retention_cue_and_test_intval_practice_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function retention_cue_and_test_intval_practice_2RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'retention_cue_and_test_intval_practice_2'-------
    // get current time
    t = retention_cue_and_test_intval_practice_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *polygon_28* updates
    if (t >= 0.0 && polygon_28.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_28.tStart = t;  // (not accounting for frame time here)
      polygon_28.frameNStart = frameN;  // exact frame index
      
      polygon_28.setAutoDraw(true);
    }

    
    // *array_image_1f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_1f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1f.tStart = t;  // (not accounting for frame time here)
      array_image_1f.frameNStart = frameN;  // exact frame index
      
      array_image_1f.setAutoDraw(true);
    }

    
    // *array_image_2f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_2f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2f.tStart = t;  // (not accounting for frame time here)
      array_image_2f.frameNStart = frameN;  // exact frame index
      
      array_image_2f.setAutoDraw(true);
    }

    
    // *array_image_3f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_3f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3f.tStart = t;  // (not accounting for frame time here)
      array_image_3f.frameNStart = frameN;  // exact frame index
      
      array_image_3f.setAutoDraw(true);
    }

    
    // *array_image_4f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_4f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4f.tStart = t;  // (not accounting for frame time here)
      array_image_4f.frameNStart = frameN;  // exact frame index
      
      array_image_4f.setAutoDraw(true);
    }

    
    // *array_image_5f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_5f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5f.tStart = t;  // (not accounting for frame time here)
      array_image_5f.frameNStart = frameN;  // exact frame index
      
      array_image_5f.setAutoDraw(true);
    }

    
    // *array_image_6f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_6f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6f.tStart = t;  // (not accounting for frame time here)
      array_image_6f.frameNStart = frameN;  // exact frame index
      
      array_image_6f.setAutoDraw(true);
    }

    
    // *array_image_7f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_7f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7f.tStart = t;  // (not accounting for frame time here)
      array_image_7f.frameNStart = frameN;  // exact frame index
      
      array_image_7f.setAutoDraw(true);
    }

    
    // *array_image_8f* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_8f.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8f.tStart = t;  // (not accounting for frame time here)
      array_image_8f.frameNStart = frameN;  // exact frame index
      
      array_image_8f.setAutoDraw(true);
    }

    
    // *pointer_dot_0b* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_0b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_0b.tStart = t;  // (not accounting for frame time here)
      pointer_dot_0b.frameNStart = frameN;  // exact frame index
      
      pointer_dot_0b.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_0b.status === PsychoJS.Status.STARTED || pointer_dot_0b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_0b.setAutoDraw(false);
    }
    
    // *pointer_dot_1b* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_1b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_1b.tStart = t;  // (not accounting for frame time here)
      pointer_dot_1b.frameNStart = frameN;  // exact frame index
      
      pointer_dot_1b.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_1b.status === PsychoJS.Status.STARTED || pointer_dot_1b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_1b.setAutoDraw(false);
    }
    
    // *pointer_dot_2b* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_2b.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_2b.tStart = t;  // (not accounting for frame time here)
      pointer_dot_2b.frameNStart = frameN;  // exact frame index
      
      pointer_dot_2b.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_2b.status === PsychoJS.Status.STARTED || pointer_dot_2b.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_2b.setAutoDraw(false);
    }
    
    // *sd_practice_resp* updates
    if (t >= response_time_s && sd_practice_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sd_practice_resp.tStart = t;  // (not accounting for frame time here)
      sd_practice_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { sd_practice_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { sd_practice_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { sd_practice_resp.clearEvents(); });
    }

    if (sd_practice_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = sd_practice_resp.getKeys({keyList: ['6', '7'], waitRelease: false});
      _sd_practice_resp_allKeys = _sd_practice_resp_allKeys.concat(theseKeys);
      if (_sd_practice_resp_allKeys.length > 0) {
        sd_practice_resp.keys = _sd_practice_resp_allKeys[_sd_practice_resp_allKeys.length - 1].name;  // just the last key pressed
        sd_practice_resp.rt = _sd_practice_resp_allKeys[_sd_practice_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of retention_cue_and_test_intval_practice_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function retention_cue_and_test_intval_practice_2RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'retention_cue_and_test_intval_practice_2'-------
    for (const thisComponent of retention_cue_and_test_intval_practice_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('sd_practice_resp.keys', sd_practice_resp.keys);
    if (typeof sd_practice_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('sd_practice_resp.rt', sd_practice_resp.rt);
        routineTimer.reset();
        }
    
    sd_practice_resp.stop();
    // the Routine "retention_cue_and_test_intval_practice_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var confidence_reminder_text;
var trialLetters;
var trialLetter;
var _conf_practice_resp_allKeys;
var confidence_rating_practiceComponents;
function confidence_rating_practiceRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'confidence_rating_practice'-------
    t = 0;
    confidence_rating_practiceClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if ((practice_trial_index === 0)) {
        confidence_reminder_text = "Excellent. At this point, right after each Same/Different choice, you\u2019re expected to indicate how confident you are that the choice you just made was correct.\n\nPlease use your right hand to press either O, K, or M: three keys that are located above each other on the right part of your keyboard. From top to bottom, O indicates high confidence, K indicates some doubt, and M indicates that your choice was just a guess.";
    } else {
        confidence_reminder_text = "Don't forget: after each Same/Different choice, you should indicate how confident you are in the choice you just made.\n\nPlease use your right hand to press either O (I'm certain), K (I have some doubt), or M (it was just a guess.)";
    }
    text_10.setText(confidence_reminder_text, false);
    
    trialLetters = ["o", "k", "m"];
    event.clearEvents()
    trialLetter='nothing'
    text_10.setText(confidence_reminder_text);
    conf_practice_resp.keys = undefined;
    conf_practice_resp.rt = undefined;
    _conf_practice_resp_allKeys = [];
    // keep track of which components have finished
    confidence_rating_practiceComponents = [];
    confidence_rating_practiceComponents.push(text_10);
    confidence_rating_practiceComponents.push(conf_practice_resp);
    confidence_rating_practiceComponents.push(image);
    
    for (const thisComponent of confidence_rating_practiceComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var _pj;
var keys;
function confidence_rating_practiceRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'confidence_rating_practice'-------
    // get current time
    t = confidence_rating_practiceClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    
    keys = event.getKeys();
    if (keys.length) {
        for (var letterIndex = 0, _pj_a = 3; (letterIndex < _pj_a); letterIndex += 1) {
            trialLetter = trialLetters[letterIndex];
            if (_pj.in_es6(trialLetter, keys)) {
                thisExp.addData("practiceConfidenceKeyPressed", trialLetter);
                thisExp.addData("confidenceKeyPressedTimePractice", timer.getTime());
                
                thisExp.nextEntry();
                continueRoutine = false;
            }
        }
        if (continueRoutine) {
            thisExp.addData("practiceIllegalKeyPressed", 1); //only reached if none of the letters in trialLetters was a hit
        }
    }
    
    // *text_10* updates
    if (t >= confidence_onset_time_s && text_10.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_10.tStart = t;  // (not accounting for frame time here)
      text_10.frameNStart = frameN;  // exact frame index
      
      text_10.setAutoDraw(true);
    }

    
    // *conf_practice_resp* updates
    if (t >= 0.0 && conf_practice_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      conf_practice_resp.tStart = t;  // (not accounting for frame time here)
      conf_practice_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { conf_practice_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { conf_practice_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { conf_practice_resp.clearEvents(); });
    }

    if (conf_practice_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = conf_practice_resp.getKeys({keyList: ['1', '2', '3'], waitRelease: false});
      _conf_practice_resp_allKeys = _conf_practice_resp_allKeys.concat(theseKeys);
      if (_conf_practice_resp_allKeys.length > 0) {
        conf_practice_resp.keys = _conf_practice_resp_allKeys[_conf_practice_resp_allKeys.length - 1].name;  // just the last key pressed
        conf_practice_resp.rt = _conf_practice_resp_allKeys[_conf_practice_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *image* updates
    if (t >= 0.0 && image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image.tStart = t;  // (not accounting for frame time here)
      image.frameNStart = frameN;  // exact frame index
      
      image.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of confidence_rating_practiceComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function confidence_rating_practiceRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'confidence_rating_practice'-------
    for (const thisComponent of confidence_rating_practiceComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    practice_trial_index = (practice_trial_index + 1);
    confidence_onset_time_s=5;
    confidence_image_onset_time_s=9999999;
    psychoJS.experiment.addData('conf_practice_resp.keys', conf_practice_resp.keys);
    if (typeof conf_practice_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('conf_practice_resp.rt', conf_practice_resp.rt);
        routineTimer.reset();
        }
    
    conf_practice_resp.stop();
    // the Routine "confidence_rating_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _static_key1_20_allKeys;
var instructions_specific_trainingComponents;
function instructions_specific_trainingRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instructions_specific_training'-------
    t = 0;
    instructions_specific_trainingClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    polygon_36.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    static_key1_20.keys = undefined;
    static_key1_20.rt = undefined;
    _static_key1_20_allKeys = [];
    // keep track of which components have finished
    instructions_specific_trainingComponents = [];
    instructions_specific_trainingComponents.push(static_instruction_42);
    instructions_specific_trainingComponents.push(polygon_36);
    instructions_specific_trainingComponents.push(static_instruction_43);
    instructions_specific_trainingComponents.push(static_key1_20);
    
    for (const thisComponent of instructions_specific_trainingComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instructions_specific_trainingRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instructions_specific_training'-------
    // get current time
    t = instructions_specific_trainingClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_42* updates
    if (t >= 0.0 && static_instruction_42.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_42.tStart = t;  // (not accounting for frame time here)
      static_instruction_42.frameNStart = frameN;  // exact frame index
      
      static_instruction_42.setAutoDraw(true);
    }

    
    // *polygon_36* updates
    if (t >= 0.0 && polygon_36.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_36.tStart = t;  // (not accounting for frame time here)
      polygon_36.frameNStart = frameN;  // exact frame index
      
      polygon_36.setAutoDraw(true);
    }

    
    // *static_instruction_43* updates
    if (t >= 0.0 && static_instruction_43.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_43.tStart = t;  // (not accounting for frame time here)
      static_instruction_43.frameNStart = frameN;  // exact frame index
      
      static_instruction_43.setAutoDraw(true);
    }

    
    // *static_key1_20* updates
    if (t >= 0.0 && static_key1_20.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_20.tStart = t;  // (not accounting for frame time here)
      static_key1_20.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_20.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_20.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_20.clearEvents(); });
    }

    if (static_key1_20.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_20.getKeys({keyList: [], waitRelease: false});
      _static_key1_20_allKeys = _static_key1_20_allKeys.concat(theseKeys);
      if (_static_key1_20_allKeys.length > 0) {
        static_key1_20.keys = _static_key1_20_allKeys[_static_key1_20_allKeys.length - 1].name;  // just the last key pressed
        static_key1_20.rt = _static_key1_20_allKeys[_static_key1_20_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instructions_specific_trainingComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instructions_specific_trainingRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instructions_specific_training'-------
    for (const thisComponent of instructions_specific_trainingComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instructions_specific_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var tracker_software_ver;
var eyelink_ver;
var tvstr;
var vindex;
var err;
var eyetracker_calibComponents;
function eyetracker_calibRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'eyetracker_calib'-------
    t = 0;
    eyetracker_calibClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if (expInfo["EyeTrack"]) {
        pylink.flushGetkeyQueue();
        pylink.getEYELINK().setOfflineMode();
        pylink.getEYELINK().sendCommand(((("screen_pixel_coords =  0 0 " + (scnWidth - 1).toString()) + " ") + (scnHeight - 1).toString()));
        pylink.getEYELINK().sendMessage(((("DISPLAY_COORDS  0 0 " + (scnWidth - 1).toString()) + " ") + (scnHeight - 1).toString()));
        tracker_software_ver = 0;
        eyelink_ver = pylink.getEYELINK().getTrackerVersion();
        if ((eyelink_ver === 3)) {
            tvstr = pylink.getEYELINK().getTrackerVersionString();
            vindex = tvstr.find("EYELINK CL");
            tracker_software_ver = Number.parseInt(Number.parseFloat(tvstr.slice((vindex + "EYELINK CL".length)).strip()));
        }
        if ((eyelink_ver >= 2)) {
            pylink.getEYELINK().sendCommand("select_parser_configuration 0");
            if ((eyelink_ver === 2)) {
                pylink.getEYELINK().sendCommand("scene_camera_gazemap = NO");
            }
        } else {
            pylink.getEYELINK().sendCommand("saccade_velocity_threshold = 35");
            pylink.getEYELINK().sendCommand("saccade_acceleration_threshold = 9500");
        }
        pylink.getEYELINK().sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT");
        if ((tracker_software_ver >= 4)) {
            pylink.getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT");
        } else {
            pylink.getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT");
        }
        pylink.getEYELINK().sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT");
        if ((tracker_software_ver >= 4)) {
            pylink.getEYELINK().sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT");
        } else {
            pylink.getEYELINK().sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT");
        }
        pylink.getEYELINK().setCalibrationType("HV9");
        pylink.getEYELINK().sendCommand("binocular_enabled = NO");
        pylink.getEYELINK().sendCommand("sample_rate = 1000");
        pylink.getEYELINK().sendCommand("elcl_tt_power = 1");
        pylink.openGraphics([scnWidth, scnHeight], 32);
        pylink.setCalibrationColors([0, 0, 0], [255, 255, 255]);
        pylink.setTargetSize(Number.parseInt((scnWidth / 70.0)), Number.parseInt((scnHeight / 300.0)));
        pylink.setCalibrationSounds("", "", "");
        pylink.setDriftCorrectSounds("", "off", "off");
        while ((! (pylink.getEYELINK().isConnected() && (! pylink.getEYELINK().breakPressed())))) {
        }
        pylink.getEYELINK().doTrackerSetup();
        pylink.closeGraphics();
        err = pylink.getEYELINK().startRecording(1, 1, 1, 1);
        pylink.pumpDelay(100);
    } else {
        continueRoutine = false;
    }
    
    // keep track of which components have finished
    eyetracker_calibComponents = [];
    
    for (const thisComponent of eyetracker_calibComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function eyetracker_calibRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'eyetracker_calib'-------
    // get current time
    t = eyetracker_calibClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of eyetracker_calibComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function eyetracker_calibRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'eyetracker_calib'-------
    for (const thisComponent of eyetracker_calibComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "eyetracker_calib" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var load_training_trialsComponents;
function load_training_trialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'load_training_trials'-------
    t = 0;
    load_training_trialsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    training_trial_image_list.append([Used_image_1, Used_image_2, Used_image_3, Used_image_4, Used_image_5, Used_image_6, Used_image_7, Used_image_8, Unused_image_1, Unused_image_2]);
    if (((training_trial_load_number + 1) <= (max_num_training_trials / 2))) {
        retro_0_post_1_training.append(0);
        change_0_1_training.append(0);
    } else {
        retro_0_post_1_training.append(1);
        change_0_1_training.append(1);
    }
    
    // keep track of which components have finished
    load_training_trialsComponents = [];
    
    for (const thisComponent of load_training_trialsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function load_training_trialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'load_training_trials'-------
    // get current time
    t = load_training_trialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of load_training_trialsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var training_trial_index;
var training_trial_index_long;
var mem_display_to_cue_time_training_s;
function load_training_trialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'load_training_trials'-------
    for (const thisComponent of load_training_trialsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    training_trial_load_number = (training_trial_load_number + 1);
    if ((training_trial_load_number >= max_num_training_trials)) {
        trials_4.finished = true;
        training_trial_index = 0;
        training_trial_index_long = 0;
        shuffle(retro_0_post_1_training);
        shuffle(change_0_1_training);
    }
    confidence_onset_time_s = 0.0;
    confidence_image_onset_time_s = 0.0;
    mem_display_to_cue_time_training_s = mem_display_to_cue_times_training_s[0];
    
    // the Routine "load_training_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var block_break_text;
var _static_key1_21_allKeys;
var get_ready_training_blockstartComponents;
function get_ready_training_blockstartRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'get_ready_training_blockstart'-------
    t = 0;
    get_ready_training_blockstartClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if (((training_trial_index > 0) && (mod(training_trial_index, repeat_intructions_every) === 0))) {
        block_break_text = "You are doing quite well! Don't forget this tip from our best performers:\n\nAs soon as the red dotted line points to a spot, immediately try to recall what was in that spot in the first group of pictures. So if the dotted line appears before the second group of pictures, don't wait for the second group of pictures to appear before recalling what it was.\n\nPlease press any key to continue your training.";
        continueRoutine = true;
    } else {
        block_break_text = "";
        continueRoutine = false;
    }
    static_instruction_44.setText(block_break_text, false);
    
    static_instruction_44.setText(block_break_text);
    static_key1_21.keys = undefined;
    static_key1_21.rt = undefined;
    _static_key1_21_allKeys = [];
    document.body.style.cursor='none';
    // keep track of which components have finished
    get_ready_training_blockstartComponents = [];
    get_ready_training_blockstartComponents.push(static_instruction_44);
    get_ready_training_blockstartComponents.push(static_key1_21);
    
    for (const thisComponent of get_ready_training_blockstartComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function get_ready_training_blockstartRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'get_ready_training_blockstart'-------
    // get current time
    t = get_ready_training_blockstartClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_44* updates
    if (t >= 0.0 && static_instruction_44.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_44.tStart = t;  // (not accounting for frame time here)
      static_instruction_44.frameNStart = frameN;  // exact frame index
      
      static_instruction_44.setAutoDraw(true);
    }

    
    // *static_key1_21* updates
    if (t >= 0.0 && static_key1_21.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_21.tStart = t;  // (not accounting for frame time here)
      static_key1_21.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_21.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_21.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_21.clearEvents(); });
    }

    if (static_key1_21.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_21.getKeys({keyList: [], waitRelease: false});
      _static_key1_21_allKeys = _static_key1_21_allKeys.concat(theseKeys);
      if (_static_key1_21_allKeys.length > 0) {
        static_key1_21.keys = _static_key1_21_allKeys[_static_key1_21_allKeys.length - 1].name;  // just the last key pressed
        static_key1_21.rt = _static_key1_21_allKeys[_static_key1_21_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of get_ready_training_blockstartComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function get_ready_training_blockstartRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'get_ready_training_blockstart'-------
    for (const thisComponent of get_ready_training_blockstartComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "get_ready_training_blockstart" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var training_mem_displayComponents;
function training_mem_displayRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'training_mem_display'-------
    t = 0;
    training_mem_displayClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    Image_1 = all_pic_names_training[training_trial_image_list[training_trial_index][0]];
    array_image_1l.setImage(Image_1, false);
    array_image_1l.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names_training[training_trial_image_list[training_trial_index][1]];
    array_image_2l.setImage(Image_2, false);
    array_image_2l.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names_training[training_trial_image_list[training_trial_index][2]];
    array_image_3l.setImage(Image_3, false);
    array_image_3l.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names_training[training_trial_image_list[training_trial_index][3]];
    array_image_4l.setImage(Image_4, false);
    array_image_4l.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names_training[training_trial_image_list[training_trial_index][4]];
    array_image_5l.setImage(Image_5, false);
    array_image_5l.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names_training[training_trial_image_list[training_trial_index][5]];
    array_image_6l.setImage(Image_6, false);
    array_image_6l.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names_training[training_trial_image_list[training_trial_index][6]];
    array_image_7l.setImage(Image_7, false);
    array_image_7l.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names_training[training_trial_image_list[training_trial_index][7]];
    array_image_8l.setImage(Image_8, false);
    array_image_8l.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    thisExp.addData("memDisplayOnTimeTraining", timer.getTime());
    for (var image_index = 0, _pj_a = num_images_per_display; (image_index < _pj_a); image_index += 1) {
        thisExp.addData((("mem_image_" + image_index.toString()) + "_training"), training_trial_image_list[training_trial_index][image_index]);
    }
    
    array_image_1l.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1l.setImage(Image_1);
    array_image_2l.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2l.setImage(Image_2);
    array_image_3l.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3l.setImage(Image_3);
    array_image_4l.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4l.setImage(Image_4);
    array_image_5l.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5l.setImage(Image_5);
    array_image_6l.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6l.setImage(Image_6);
    array_image_7l.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7l.setImage(Image_7);
    array_image_8l.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8l.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8l.setImage(Image_8);
    polygon_44.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    training_mem_displayComponents = [];
    training_mem_displayComponents.push(array_image_1l);
    training_mem_displayComponents.push(array_image_2l);
    training_mem_displayComponents.push(array_image_3l);
    training_mem_displayComponents.push(array_image_4l);
    training_mem_displayComponents.push(array_image_5l);
    training_mem_displayComponents.push(array_image_6l);
    training_mem_displayComponents.push(array_image_7l);
    training_mem_displayComponents.push(array_image_8l);
    training_mem_displayComponents.push(polygon_44);
    
    for (const thisComponent of training_mem_displayComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function training_mem_displayRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'training_mem_display'-------
    // get current time
    t = training_mem_displayClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *array_image_1l* updates
    if (t >= 0.0 && array_image_1l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1l.tStart = t;  // (not accounting for frame time here)
      array_image_1l.frameNStart = frameN;  // exact frame index
      
      array_image_1l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_1l.status === PsychoJS.Status.STARTED || array_image_1l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_1l.setAutoDraw(false);
    }
    
    // *array_image_2l* updates
    if (t >= 0.0 && array_image_2l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2l.tStart = t;  // (not accounting for frame time here)
      array_image_2l.frameNStart = frameN;  // exact frame index
      
      array_image_2l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_2l.status === PsychoJS.Status.STARTED || array_image_2l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_2l.setAutoDraw(false);
    }
    
    // *array_image_3l* updates
    if (t >= 0.0 && array_image_3l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3l.tStart = t;  // (not accounting for frame time here)
      array_image_3l.frameNStart = frameN;  // exact frame index
      
      array_image_3l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_3l.status === PsychoJS.Status.STARTED || array_image_3l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_3l.setAutoDraw(false);
    }
    
    // *array_image_4l* updates
    if (t >= 0.0 && array_image_4l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4l.tStart = t;  // (not accounting for frame time here)
      array_image_4l.frameNStart = frameN;  // exact frame index
      
      array_image_4l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_4l.status === PsychoJS.Status.STARTED || array_image_4l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_4l.setAutoDraw(false);
    }
    
    // *array_image_5l* updates
    if (t >= 0.0 && array_image_5l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5l.tStart = t;  // (not accounting for frame time here)
      array_image_5l.frameNStart = frameN;  // exact frame index
      
      array_image_5l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_5l.status === PsychoJS.Status.STARTED || array_image_5l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_5l.setAutoDraw(false);
    }
    
    // *array_image_6l* updates
    if (t >= 0.0 && array_image_6l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6l.tStart = t;  // (not accounting for frame time here)
      array_image_6l.frameNStart = frameN;  // exact frame index
      
      array_image_6l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_6l.status === PsychoJS.Status.STARTED || array_image_6l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_6l.setAutoDraw(false);
    }
    
    // *array_image_7l* updates
    if (t >= 0.0 && array_image_7l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7l.tStart = t;  // (not accounting for frame time here)
      array_image_7l.frameNStart = frameN;  // exact frame index
      
      array_image_7l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_7l.status === PsychoJS.Status.STARTED || array_image_7l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_7l.setAutoDraw(false);
    }
    
    // *array_image_8l* updates
    if (t >= 0.0 && array_image_8l.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8l.tStart = t;  // (not accounting for frame time here)
      array_image_8l.frameNStart = frameN;  // exact frame index
      
      array_image_8l.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_8l.status === PsychoJS.Status.STARTED || array_image_8l.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_8l.setAutoDraw(false);
    }
    
    // *polygon_44* updates
    if (t >= 0.0 && polygon_44.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_44.tStart = t;  // (not accounting for frame time here)
      polygon_44.frameNStart = frameN;  // exact frame index
      
      polygon_44.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((polygon_44.status === PsychoJS.Status.STARTED || polygon_44.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      polygon_44.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of training_mem_displayComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var change_from_index_training;
var change_or_no_training;
var change_to_index_training;
function training_mem_displayRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'training_mem_display'-------
    for (const thisComponent of training_mem_displayComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    
            // add-on: list(s: string): string[]
            function list(s) {
                // if s is a string, we return a list of its characters
                if (typeof s === 'string')
                    return s.split('');
                else
                    // otherwise we return s:
                    return s;
            }
    
            mem_display_to_cue_duration_s = mem_display_to_cue_time_training_s;
    if (retro_0_post_1_training[training_trial_index]) {
        mem_display_to_test_array_duration_s = (mem_display_to_cue_duration_s - 0.1);
        routine_duration_s = (mem_display_to_cue_duration_s + cue_duration_s);
        response_time_s = (routine_duration_s - cue_duration_s);
    } else {
        mem_display_to_test_array_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        routine_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        response_time_s = routine_duration_s;
    }
    shuffle(image_index_list_for_shuffling);
    change_from_index_training = image_index_list_for_shuffling[0];
    image_index_list_in_order = list(range(0, num_images_per_display));
    change_or_no_training = change_0_1_training[training_trial_index];
    if (change_or_no_training) {
        change_to_index_training = num_images_per_display;
        image_index_list_in_order[change_from_index_training] = change_to_index_training;
    }
    thisExp.addData("retention_interval_training", mem_display_to_cue_duration_s);
    thisExp.addData("retro_0_post_1_training", retro_0_post_1_training[training_trial_index]);
    thisExp.addData("change_or_no_training", change_or_no_training);
    thisExp.addData("cue_location_training", change_from_index_training);
    thisExp.addData("mem_display_to_test_array_duration_s", mem_display_to_test_array_duration_s);
    thisExp.addData("mem_display_to_cue_duration_s", mem_display_to_cue_duration_s);
    thisExp.addData("routine_duration_s", routine_duration_s);
    
    // the Routine "training_mem_display" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _sd_training_resp_allKeys;
var retention_cue_and_test_intval_trainingComponents;
function retention_cue_and_test_intval_trainingRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'retention_cue_and_test_intval_training'-------
    t = 0;
    retention_cue_and_test_intval_trainingClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    Image_1 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[0]]];
    array_image_1m.setImage(Image_1, false);
    array_image_1m.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[1]]];
    array_image_2m.setImage(Image_2, false);
    array_image_2m.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[2]]];
    array_image_3m.setImage(Image_3, false);
    array_image_3m.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[3]]];
    array_image_4m.setImage(Image_4, false);
    array_image_4m.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[4]]];
    array_image_5m.setImage(Image_5, false);
    array_image_5m.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[5]]];
    array_image_6m.setImage(Image_6, false);
    array_image_6m.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[6]]];
    array_image_7m.setImage(Image_7, false);
    array_image_7m.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names_training[training_trial_image_list[training_trial_index][image_index_list_in_order[7]]];
    array_image_8m.setImage(Image_8, false);
    array_image_8m.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    x_pointer_0 = (((x_y_gains_per_angle[change_from_index_training][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale);
    y_pointer_0 = (((x_y_gains_per_angle[change_from_index_training][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale);
    pointer_dot_0d.setPos([x_pointer_0, y_pointer_0], false);
    x_pointer_1 = (x_pointer_0 * 0.66667);
    y_pointer_1 = (y_pointer_0 * 0.66667);
    pointer_dot_1d.setPos([x_pointer_1, y_pointer_1], false);
    x_pointer_2 = (x_pointer_0 * 0.33333);
    y_pointer_2 = (y_pointer_0 * 0.33333);
    pointer_dot_2d.setPos([x_pointer_2, y_pointer_2], false);
    for (var image_index = 0, _pj_a = 8; (image_index < _pj_a); image_index += 1) {
        thisExp.addData((("test_image_" + image_index.toString()) + "_training"), training_trial_image_list[training_trial_index][image_index_list_in_order[image_index]]);
    }
    
    polygon_46.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    array_image_1m.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1m.setImage(Image_1);
    array_image_2m.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2m.setImage(Image_2);
    array_image_3m.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3m.setImage(Image_3);
    array_image_4m.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4m.setImage(Image_4);
    array_image_5m.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5m.setImage(Image_5);
    array_image_6m.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6m.setImage(Image_6);
    array_image_7m.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7m.setImage(Image_7);
    array_image_8m.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8m.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8m.setImage(Image_8);
    pointer_dot_0d.setPos([x_pointer_0, y_pointer_0]);
    pointer_dot_0d.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_1d.setPos([x_pointer_1, y_pointer_1]);
    pointer_dot_1d.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_2d.setPos([x_pointer_2, y_pointer_2]);
    pointer_dot_2d.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    sd_training_resp.keys = undefined;
    sd_training_resp.rt = undefined;
    _sd_training_resp_allKeys = [];
    // keep track of which components have finished
    retention_cue_and_test_intval_trainingComponents = [];
    retention_cue_and_test_intval_trainingComponents.push(polygon_46);
    retention_cue_and_test_intval_trainingComponents.push(array_image_1m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_2m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_3m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_4m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_5m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_6m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_7m);
    retention_cue_and_test_intval_trainingComponents.push(array_image_8m);
    retention_cue_and_test_intval_trainingComponents.push(pointer_dot_0d);
    retention_cue_and_test_intval_trainingComponents.push(pointer_dot_1d);
    retention_cue_and_test_intval_trainingComponents.push(pointer_dot_2d);
    retention_cue_and_test_intval_trainingComponents.push(reminder_training);
    retention_cue_and_test_intval_trainingComponents.push(sd_training_resp);
    
    for (const thisComponent of retention_cue_and_test_intval_trainingComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function retention_cue_and_test_intval_trainingRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'retention_cue_and_test_intval_training'-------
    // get current time
    t = retention_cue_and_test_intval_trainingClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *polygon_46* updates
    if (t >= 0.0 && polygon_46.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_46.tStart = t;  // (not accounting for frame time here)
      polygon_46.frameNStart = frameN;  // exact frame index
      
      polygon_46.setAutoDraw(true);
    }

    
    // *array_image_1m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_1m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1m.tStart = t;  // (not accounting for frame time here)
      array_image_1m.frameNStart = frameN;  // exact frame index
      
      array_image_1m.setAutoDraw(true);
    }

    
    // *array_image_2m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_2m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2m.tStart = t;  // (not accounting for frame time here)
      array_image_2m.frameNStart = frameN;  // exact frame index
      
      array_image_2m.setAutoDraw(true);
    }

    
    // *array_image_3m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_3m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3m.tStart = t;  // (not accounting for frame time here)
      array_image_3m.frameNStart = frameN;  // exact frame index
      
      array_image_3m.setAutoDraw(true);
    }

    
    // *array_image_4m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_4m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4m.tStart = t;  // (not accounting for frame time here)
      array_image_4m.frameNStart = frameN;  // exact frame index
      
      array_image_4m.setAutoDraw(true);
    }

    
    // *array_image_5m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_5m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5m.tStart = t;  // (not accounting for frame time here)
      array_image_5m.frameNStart = frameN;  // exact frame index
      
      array_image_5m.setAutoDraw(true);
    }

    
    // *array_image_6m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_6m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6m.tStart = t;  // (not accounting for frame time here)
      array_image_6m.frameNStart = frameN;  // exact frame index
      
      array_image_6m.setAutoDraw(true);
    }

    
    // *array_image_7m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_7m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7m.tStart = t;  // (not accounting for frame time here)
      array_image_7m.frameNStart = frameN;  // exact frame index
      
      array_image_7m.setAutoDraw(true);
    }

    
    // *array_image_8m* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_8m.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8m.tStart = t;  // (not accounting for frame time here)
      array_image_8m.frameNStart = frameN;  // exact frame index
      
      array_image_8m.setAutoDraw(true);
    }

    
    // *pointer_dot_0d* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_0d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_0d.tStart = t;  // (not accounting for frame time here)
      pointer_dot_0d.frameNStart = frameN;  // exact frame index
      
      pointer_dot_0d.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_0d.status === PsychoJS.Status.STARTED || pointer_dot_0d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_0d.setAutoDraw(false);
    }
    
    // *pointer_dot_1d* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_1d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_1d.tStart = t;  // (not accounting for frame time here)
      pointer_dot_1d.frameNStart = frameN;  // exact frame index
      
      pointer_dot_1d.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_1d.status === PsychoJS.Status.STARTED || pointer_dot_1d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_1d.setAutoDraw(false);
    }
    
    // *pointer_dot_2d* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_2d.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_2d.tStart = t;  // (not accounting for frame time here)
      pointer_dot_2d.frameNStart = frameN;  // exact frame index
      
      pointer_dot_2d.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_2d.status === PsychoJS.Status.STARTED || pointer_dot_2d.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_2d.setAutoDraw(false);
    }
    
    // *reminder_training* updates
    if (t >= 10 && reminder_training.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      reminder_training.tStart = t;  // (not accounting for frame time here)
      reminder_training.frameNStart = frameN;  // exact frame index
      
      reminder_training.setAutoDraw(true);
    }

    
    // *sd_training_resp* updates
    if (t >= response_time_s && sd_training_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sd_training_resp.tStart = t;  // (not accounting for frame time here)
      sd_training_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { sd_training_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { sd_training_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { sd_training_resp.clearEvents(); });
    }

    if (sd_training_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = sd_training_resp.getKeys({keyList: ['6', '7'], waitRelease: false});
      _sd_training_resp_allKeys = _sd_training_resp_allKeys.concat(theseKeys);
      if (_sd_training_resp_allKeys.length > 0) {
        sd_training_resp.keys = _sd_training_resp_allKeys[_sd_training_resp_allKeys.length - 1].name;  // just the last key pressed
        sd_training_resp.rt = _sd_training_resp_allKeys[_sd_training_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    if (sd_training_resp.keys.length) {
        if ((short_block_done === 0)) {
            if ((retro_0_post_1_training[training_trial_index] === 0)) {
                if (((change_or_no_training === 0) && (sd_training_resp.keys === response_string_pair[0]))) {
                    consecutive_correct_retro = (consecutive_correct_retro + 1);
                } else {
                    if (((change_or_no_training === 1) && (sd_training_resp.keys === response_string_pair[1]))) {
                        consecutive_correct_retro = (consecutive_correct_retro + 1);
                    } else {
                        consecutive_correct_retro = 0;
                    }
                }
            }
        } else {
            if ((short_block_done === 1)) {
                if ((retro_0_post_1_training[training_trial_index] === 0)) {
                    retro_trials_long_perblock = (retro_trials_long_perblock + 1);
                    if (((change_or_no_training === 0) && (sd_training_resp.keys === response_string_pair[0]))) {
                        correct_retro_long_perblock = (correct_retro_long_perblock + 1);
                    } else {
                        if (((change_or_no_training === 1) && (sd_training_resp.keys === response_string_pair[1]))) {
                            correct_retro_long_perblock = (correct_retro_long_perblock + 1);
                        }
                    }
                }
            }
        }
    }
    thisExp.addData("consecutive_correct_retro_short", consecutive_correct_retro);
    thisExp.addData("correct_retro_long_perblock", correct_retro_long_perblock);
    thisExp.addData("retro_trials_long_perblock", retro_trials_long_perblock);
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of retention_cue_and_test_intval_trainingComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function retention_cue_and_test_intval_trainingRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'retention_cue_and_test_intval_training'-------
    for (const thisComponent of retention_cue_and_test_intval_trainingComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('sd_training_resp.keys', sd_training_resp.keys);
    if (typeof sd_training_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('sd_training_resp.rt', sd_training_resp.rt);
        routineTimer.reset();
        }
    
    sd_training_resp.stop();
    // the Routine "retention_cue_and_test_intval_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _conf_training_resp_allKeys;
var confidence_rating_trainingComponents;
function confidence_rating_trainingRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'confidence_rating_training'-------
    t = 0;
    confidence_rating_trainingClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    text_15.setText("Don't forget: after each Same/Different choice, you should indicate how confident you are in the choice you just made.\n\nPlease use your left hand, using your ring finger if you're certain, your middle finger if you have doubt, or your index finger if it was just a guess.");
    conf_training_resp.keys = undefined;
    conf_training_resp.rt = undefined;
    _conf_training_resp_allKeys = [];
    // keep track of which components have finished
    confidence_rating_trainingComponents = [];
    confidence_rating_trainingComponents.push(text_15);
    confidence_rating_trainingComponents.push(conf_training_resp);
    
    for (const thisComponent of confidence_rating_trainingComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function confidence_rating_trainingRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'confidence_rating_training'-------
    // get current time
    t = confidence_rating_trainingClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_15* updates
    if (t >= 10 && text_15.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_15.tStart = t;  // (not accounting for frame time here)
      text_15.frameNStart = frameN;  // exact frame index
      
      text_15.setAutoDraw(true);
    }

    
    // *conf_training_resp* updates
    if (t >= 0.0 && conf_training_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      conf_training_resp.tStart = t;  // (not accounting for frame time here)
      conf_training_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { conf_training_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { conf_training_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { conf_training_resp.clearEvents(); });
    }

    if (conf_training_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = conf_training_resp.getKeys({keyList: ['1', '2', '3'], waitRelease: false});
      _conf_training_resp_allKeys = _conf_training_resp_allKeys.concat(theseKeys);
      if (_conf_training_resp_allKeys.length > 0) {
        conf_training_resp.keys = _conf_training_resp_allKeys[_conf_training_resp_allKeys.length - 1].name;  // just the last key pressed
        conf_training_resp.rt = _conf_training_resp_allKeys[_conf_training_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of confidence_rating_trainingComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var long_block_retro_acc;
function confidence_rating_trainingRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'confidence_rating_training'-------
    for (const thisComponent of confidence_rating_trainingComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    training_trial_index = (training_trial_index + 1);
    if (((short_block_done === 0) && (consecutive_correct_retro === consecutive_retro_correct_required))) {
        thisExp.addData("finish_training_early", 0);
        mem_display_to_cue_time_training_s = mem_display_to_cue_times_training_s[1];
        consecutive_correct_retro = 0;
        short_block_done = 1;
    }
    if (short_block_done) {
        training_trial_index_long = (training_trial_index_long + 1);
        if (((training_trial_index_long > 0) && (mod(training_trial_index_long, trials_long_perblock) === 0))) {
            thisExp.addData("finish_training_early", 0);
            long_block_retro_acc = (correct_retro_long_perblock / retro_trials_long_perblock);
            thisExp.addData("long_block_retro_acc", long_block_retro_acc);
            if ((long_block_retro_acc >= 0.8)) {
                trials_10.finished = true;
            } else {
                training_trial_index_long = 0;
                correct_retro_long_perblock = 0;
                retro_trials_long_perblock = 0;
            }
        } else {
            if (((training_trial_index >= max_num_training_trials) && (long_block_retro_acc < 0.8))) {
                thisExp.addData("finish_training_early", 1);
                win.close();
                core.quit();
            }
        }
    }
    thisExp.addData("training_trial_index", training_trial_index);
    thisExp.addData("training_trial_index_long", training_trial_index_long);
    
    psychoJS.experiment.addData('conf_training_resp.keys', conf_training_resp.keys);
    if (typeof conf_training_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('conf_training_resp.rt', conf_training_resp.rt);
        routineTimer.reset();
        }
    
    conf_training_resp.stop();
    // the Routine "confidence_rating_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var real_exp_text;
var real_exp_text_2;
var _static_key1_15_allKeys;
var instruction_8Components;
function instruction_8RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instruction_8'-------
    t = 0;
    instruction_8Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    real_exp_text = (("Great. It is now time to move on to the real experiment. Your task will be exactly the same as during the practice trials you just completed, except that there will be no text on the screen anymore. Instead, you’ll only see the pictures and the red dotted line without announcements or reminders, and the experiment will automatically proceed to the next trial as soon as you’ve pressed S or D for same/different, followed by O, K, or M for your confidence.\n\nYou will, however, be given a break every " + (trials_per_block).toString() + " trials, to give your eyes a rest.\n\nThree final reminders: please keep the experiment in fullscreen mode and, outside of the breaks, please remain at the same natural position relative to the screen and keep your eyes directed at the black dot"));
    static_instruction_33.setText(real_exp_text,false);
    
    real_exp_text_2=(("There are " + (num_real_trials).toString() + " trials total.\n\nPlease press spacebar to begin."));
    static_instruction_34.setText(real_exp_text_2,false);
    static_instruction_33.setText(real_exp_text);
    static_key1_15.keys = undefined;
    static_key1_15.rt = undefined;
    _static_key1_15_allKeys = [];
    static_instruction_34.setText(real_exp_text_2);
    polygon_31.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    instruction_8Components = [];
    instruction_8Components.push(static_instruction_33);
    instruction_8Components.push(static_key1_15);
    instruction_8Components.push(static_instruction_34);
    instruction_8Components.push(polygon_31);
    
    for (const thisComponent of instruction_8Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction_8RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instruction_8'-------
    // get current time
    t = instruction_8Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_33* updates
    if (t >= 0.0 && static_instruction_33.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_33.tStart = t;  // (not accounting for frame time here)
      static_instruction_33.frameNStart = frameN;  // exact frame index
      
      static_instruction_33.setAutoDraw(true);
    }

    
    // *static_key1_15* updates
    if (t >= 0.0 && static_key1_15.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_15.tStart = t;  // (not accounting for frame time here)
      static_key1_15.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_15.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_15.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_15.clearEvents(); });
    }

    if (static_key1_15.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_15.getKeys({keyList: [], waitRelease: false});
      _static_key1_15_allKeys = _static_key1_15_allKeys.concat(theseKeys);
      if (_static_key1_15_allKeys.length > 0) {
        static_key1_15.keys = _static_key1_15_allKeys[_static_key1_15_allKeys.length - 1].name;  // just the last key pressed
        static_key1_15.rt = _static_key1_15_allKeys[_static_key1_15_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *static_instruction_34* updates
    if (t >= 0.0 && static_instruction_34.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_34.tStart = t;  // (not accounting for frame time here)
      static_instruction_34.frameNStart = frameN;  // exact frame index
      
      static_instruction_34.setAutoDraw(true);
    }

    
    // *polygon_31* updates
    if (t >= 0.0 && polygon_31.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_31.tStart = t;  // (not accounting for frame time here)
      polygon_31.frameNStart = frameN;  // exact frame index
      
      polygon_31.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction_8Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_8RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instruction_8'-------
    for (const thisComponent of instruction_8Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "instruction_8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var load_real_trialsComponents;
function load_real_trialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'load_real_trials'-------
    t = 0;
    load_real_trialsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    real_trial_image_list.append([Used_image_1, Used_image_2, Used_image_3, Used_image_4, Used_image_5, Used_image_6, Used_image_7, Used_image_8, Unused_image_1, Unused_image_2]);
    retro_0_post_1_real.append(Retro_post_0_1);
    change_0_1_real.append(Change_yes_no);
    retint_short_0_long_1.append(Short_long_0_1);
    thisExp.addData("DataFile", condFile);
    
    // keep track of which components have finished
    load_real_trialsComponents = [];
    
    for (const thisComponent of load_real_trialsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function load_real_trialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'load_real_trials'-------
    // get current time
    t = load_real_trialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of load_real_trialsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function load_real_trialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'load_real_trials'-------
    for (const thisComponent of load_real_trialsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    trial_load_number = (trial_load_number + 1);
    if ((trial_load_number >= num_real_trials)) {
        trials_7.finished = true;
        real_trial_index = 0;
    }
    
    // the Routine "load_real_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var first_part_text_text;
var _static_key1_19_allKeys;
var get_ready_real_blockstartComponents;
function get_ready_real_blockstartRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'get_ready_real_blockstart'-------
    t = 0;
    get_ready_real_blockstartClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if (((real_trial_index > 0) && (mod(real_trial_index, trials_per_block) === 0))) {
        if ((real_trial_index === trials_per_block)) {
            first_part_text_text = ((((("You just finished your first block of " + trials_per_block.toString()) + " trials. In total you have now completed ") + real_trial_index.toString()) + " trials out of ") + num_real_trials.toString());
        } else {
            first_part_text_text = ((((("You finished another block of " + trials_per_block.toString()) + " trials. In total you have now completed ") + real_trial_index.toString()) + " trials out of ") + num_real_trials.toString());
        }
        block_break_text = (first_part_text_text + ". Please feel free to take a breather. If you want to move now, be sure to get back in your original position relative to the screen before resuming.\n\nWhen you do resume, please don't forget to keep your eyes on the black dot in the center again. Please press any button when you are ready to continue.");
        continueRoutine = true;
    } else {
        block_break_text = "";
        continueRoutine = false;
    }
    static_instruction_41.setText(block_break_text, false);
    
    static_instruction_41.setText(block_break_text);
    static_key1_19.keys = undefined;
    static_key1_19.rt = undefined;
    _static_key1_19_allKeys = [];
    document.body.style.cursor='none';
    // keep track of which components have finished
    get_ready_real_blockstartComponents = [];
    get_ready_real_blockstartComponents.push(static_instruction_41);
    get_ready_real_blockstartComponents.push(static_key1_19);
    
    for (const thisComponent of get_ready_real_blockstartComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function get_ready_real_blockstartRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'get_ready_real_blockstart'-------
    // get current time
    t = get_ready_real_blockstartClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *static_instruction_41* updates
    if (t >= 0.0 && static_instruction_41.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_instruction_41.tStart = t;  // (not accounting for frame time here)
      static_instruction_41.frameNStart = frameN;  // exact frame index
      
      static_instruction_41.setAutoDraw(true);
    }

    
    // *static_key1_19* updates
    if (t >= 0.0 && static_key1_19.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      static_key1_19.tStart = t;  // (not accounting for frame time here)
      static_key1_19.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { static_key1_19.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { static_key1_19.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { static_key1_19.clearEvents(); });
    }

    if (static_key1_19.status === PsychoJS.Status.STARTED) {
      let theseKeys = static_key1_19.getKeys({keyList: [], waitRelease: false});
      _static_key1_19_allKeys = _static_key1_19_allKeys.concat(theseKeys);
      if (_static_key1_19_allKeys.length > 0) {
        static_key1_19.keys = _static_key1_19_allKeys[_static_key1_19_allKeys.length - 1].name;  // just the last key pressed
        static_key1_19.rt = _static_key1_19_allKeys[_static_key1_19_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of get_ready_real_blockstartComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function get_ready_real_blockstartRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'get_ready_real_blockstart'-------
    for (const thisComponent of get_ready_real_blockstartComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "get_ready_real_blockstart" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var real_mem_displayComponents;
function real_mem_displayRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'real_mem_display'-------
    t = 0;
    real_mem_displayClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    Image_1 = all_pic_names[real_trial_image_list[real_trial_index][0]];
    array_image_1h.setImage(Image_1, false);
    array_image_1h.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names[real_trial_image_list[real_trial_index][1]];
    array_image_2h.setImage(Image_2, false);
    array_image_2h.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names[real_trial_image_list[real_trial_index][2]];
    array_image_3h.setImage(Image_3, false);
    array_image_3h.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names[real_trial_image_list[real_trial_index][3]];
    array_image_4h.setImage(Image_4, false);
    array_image_4h.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names[real_trial_image_list[real_trial_index][4]];
    array_image_5h.setImage(Image_5, false);
    array_image_5h.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names[real_trial_image_list[real_trial_index][5]];
    array_image_6h.setImage(Image_6, false);
    array_image_6h.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names[real_trial_image_list[real_trial_index][6]];
    array_image_7h.setImage(Image_7, false);
    array_image_7h.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names[real_trial_image_list[real_trial_index][7]];
    array_image_8h.setImage(Image_8, false);
    array_image_8h.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    thisExp.addData("memDisplayOnTime", timer.getTime());
    for (var image_index = 0, _pj_a = 8; (image_index < _pj_a); image_index += 1) {
        thisExp.addData((("mem_image_" + image_index.toString()) + "_real"), real_trial_image_list[real_trial_index][image_index]);
    }
    pylink.getEYELINK().sendMessage("mem_display_onset");
    
    array_image_1h.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1h.setImage(Image_1);
    array_image_2h.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2h.setImage(Image_2);
    array_image_3h.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3h.setImage(Image_3);
    array_image_4h.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4h.setImage(Image_4);
    array_image_5h.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5h.setImage(Image_5);
    array_image_6h.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6h.setImage(Image_6);
    array_image_7h.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7h.setImage(Image_7);
    array_image_8h.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8h.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8h.setImage(Image_8);
    polygon_33.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    // keep track of which components have finished
    real_mem_displayComponents = [];
    real_mem_displayComponents.push(array_image_1h);
    real_mem_displayComponents.push(array_image_2h);
    real_mem_displayComponents.push(array_image_3h);
    real_mem_displayComponents.push(array_image_4h);
    real_mem_displayComponents.push(array_image_5h);
    real_mem_displayComponents.push(array_image_6h);
    real_mem_displayComponents.push(array_image_7h);
    real_mem_displayComponents.push(array_image_8h);
    real_mem_displayComponents.push(polygon_33);
    
    for (const thisComponent of real_mem_displayComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function real_mem_displayRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'real_mem_display'-------
    // get current time
    t = real_mem_displayClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *array_image_1h* updates
    if (t >= 0.0 && array_image_1h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1h.tStart = t;  // (not accounting for frame time here)
      array_image_1h.frameNStart = frameN;  // exact frame index
      
      array_image_1h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_1h.status === PsychoJS.Status.STARTED || array_image_1h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_1h.setAutoDraw(false);
    }
    
    // *array_image_2h* updates
    if (t >= 0.0 && array_image_2h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2h.tStart = t;  // (not accounting for frame time here)
      array_image_2h.frameNStart = frameN;  // exact frame index
      
      array_image_2h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_2h.status === PsychoJS.Status.STARTED || array_image_2h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_2h.setAutoDraw(false);
    }
    
    // *array_image_3h* updates
    if (t >= 0.0 && array_image_3h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3h.tStart = t;  // (not accounting for frame time here)
      array_image_3h.frameNStart = frameN;  // exact frame index
      
      array_image_3h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_3h.status === PsychoJS.Status.STARTED || array_image_3h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_3h.setAutoDraw(false);
    }
    
    // *array_image_4h* updates
    if (t >= 0.0 && array_image_4h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4h.tStart = t;  // (not accounting for frame time here)
      array_image_4h.frameNStart = frameN;  // exact frame index
      
      array_image_4h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_4h.status === PsychoJS.Status.STARTED || array_image_4h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_4h.setAutoDraw(false);
    }
    
    // *array_image_5h* updates
    if (t >= 0.0 && array_image_5h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5h.tStart = t;  // (not accounting for frame time here)
      array_image_5h.frameNStart = frameN;  // exact frame index
      
      array_image_5h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_5h.status === PsychoJS.Status.STARTED || array_image_5h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_5h.setAutoDraw(false);
    }
    
    // *array_image_6h* updates
    if (t >= 0.0 && array_image_6h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6h.tStart = t;  // (not accounting for frame time here)
      array_image_6h.frameNStart = frameN;  // exact frame index
      
      array_image_6h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_6h.status === PsychoJS.Status.STARTED || array_image_6h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_6h.setAutoDraw(false);
    }
    
    // *array_image_7h* updates
    if (t >= 0.0 && array_image_7h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7h.tStart = t;  // (not accounting for frame time here)
      array_image_7h.frameNStart = frameN;  // exact frame index
      
      array_image_7h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_7h.status === PsychoJS.Status.STARTED || array_image_7h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_7h.setAutoDraw(false);
    }
    
    // *array_image_8h* updates
    if (t >= 0.0 && array_image_8h.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8h.tStart = t;  // (not accounting for frame time here)
      array_image_8h.frameNStart = frameN;  // exact frame index
      
      array_image_8h.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((array_image_8h.status === PsychoJS.Status.STARTED || array_image_8h.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      array_image_8h.setAutoDraw(false);
    }
    
    // *polygon_33* updates
    if (t >= 0.0 && polygon_33.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_33.tStart = t;  // (not accounting for frame time here)
      polygon_33.frameNStart = frameN;  // exact frame index
      
      polygon_33.setAutoDraw(true);
    }

    frameRemains = 0.0 + mem_display_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((polygon_33.status === PsychoJS.Status.STARTED || polygon_33.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      polygon_33.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of real_mem_displayComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var ret_int;
var change_from_index_real;
var change_or_no_real;
var change_to_index_real;
function real_mem_displayRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'real_mem_display'-------
    for (const thisComponent of real_mem_displayComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    
            // add-on: list(s: string): string[]
            function list(s) {
                // if s is a string, we return a list of its characters
                if (typeof s === 'string')
                    return s.split('');
                else
                    // otherwise we return s:
                    return s;
            }
    
            ret_int = mem_display_to_cue_times_training_s[retint_short_0_long_1[real_trial_index]];
    mem_display_to_cue_duration_s = ret_int;
    if (retro_0_post_1_real[real_trial_index]) {
        mem_display_to_test_array_duration_s = (mem_display_to_cue_duration_s - 0.1);
        routine_duration_s = (mem_display_to_cue_duration_s + cue_duration_s);
        response_time_s = (routine_duration_s - cue_duration_s);
    } else {
        mem_display_to_test_array_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        routine_duration_s = ((mem_display_to_cue_duration_s + cue_duration_s) + retention_duration_s);
        response_time_s = routine_duration_s;
    }
    shuffle(image_index_list_for_shuffling);
    change_from_index_real = image_index_list_for_shuffling[0];
    image_index_list_in_order = list(range(0, num_images_per_display));
    change_or_no_real = change_0_1_real[real_trial_index];
    if (change_or_no_real) {
        change_to_index_real = num_images_per_display;
        image_index_list_in_order[change_from_index_real] = change_to_index_real;
    }
    thisExp.addData("retention_interval_real", mem_display_to_cue_duration_s);
    thisExp.addData("retro_0_post_1_real", retro_0_post_1_real[real_trial_index]);
    thisExp.addData("change_or_no_real", change_or_no_real);
    thisExp.addData("cue_location_real", change_from_index_real);
    thisExp.addData("ret_int_real", ret_int);
    pylink.getEYELINK().sendMessage("mem_display_offset");
    
    // the Routine "real_mem_display" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _sd_resp_real_allKeys;
var retention_cue_and_test_intval_realComponents;
function retention_cue_and_test_intval_realRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'retention_cue_and_test_intval_real'-------
    t = 0;
    retention_cue_and_test_intval_realClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    polygon_34.setSize([(dot_diam_dva * x_scale), (dot_diam_dva * y_scale)]);
    Image_1 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[0]]];
    array_image_1j.setImage(Image_1, false);
    array_image_1j.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)], false);
    array_image_1j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_2 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[1]]];
    array_image_2j.setImage(Image_2, false);
    array_image_2j.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)], false);
    array_image_2j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_3 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[2]]];
    array_image_3j.setImage(Image_3, false);
    array_image_3j.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)], false);
    array_image_3j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_4 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[3]]];
    array_image_4j.setImage(Image_4, false);
    array_image_4j.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)], false);
    array_image_4j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_5 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[4]]];
    array_image_5j.setImage(Image_5, false);
    array_image_5j.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)], false);
    array_image_5j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_6 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[5]]];
    array_image_6j.setImage(Image_6, false);
    array_image_6j.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)], false);
    array_image_6j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_7 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[6]]];
    array_image_7j.setImage(Image_7, false);
    array_image_7j.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)], false);
    array_image_7j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    Image_8 = all_pic_names[real_trial_image_list[real_trial_index][image_index_list_in_order[7]]];
    array_image_8j.setImage(Image_8, false);
    array_image_8j.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)], false);
    array_image_8j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)], false);
    x_pointer_0 = (((x_y_gains_per_angle[change_from_index_real][0] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * x_scale);
    y_pointer_0 = (((x_y_gains_per_angle[change_from_index_real][1] * pointer_length_relative_to_image_eccentricity) * image_eccentr_cm) * y_scale);
    pointer_dot_0c.setPos([x_pointer_0, y_pointer_0], false);
    x_pointer_1 = (x_pointer_0 * 0.66667);
    y_pointer_1 = (y_pointer_0 * 0.66667);
    pointer_dot_1c.setPos([x_pointer_1, y_pointer_1], false);
    x_pointer_2 = (x_pointer_0 * 0.33333);
    y_pointer_2 = (y_pointer_0 * 0.33333);
    pointer_dot_2c.setPos([x_pointer_2, y_pointer_2], false);
    
    array_image_1j.setPos([((x_y_gains_per_angle[0][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[0][1] * image_eccentr_cm) * y_scale)]);
    array_image_1j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_1j.setImage(Image_1);
    array_image_2j.setPos([((x_y_gains_per_angle[1][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[1][1] * image_eccentr_cm) * y_scale)]);
    array_image_2j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_2j.setImage(Image_2);
    array_image_3j.setPos([((x_y_gains_per_angle[2][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[2][1] * image_eccentr_cm) * y_scale)]);
    array_image_3j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_3j.setImage(Image_3);
    array_image_4j.setPos([((x_y_gains_per_angle[3][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[3][1] * image_eccentr_cm) * y_scale)]);
    array_image_4j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_4j.setImage(Image_4);
    array_image_5j.setPos([((x_y_gains_per_angle[4][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[4][1] * image_eccentr_cm) * y_scale)]);
    array_image_5j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_5j.setImage(Image_5);
    array_image_6j.setPos([((x_y_gains_per_angle[5][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[5][1] * image_eccentr_cm) * y_scale)]);
    array_image_6j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_6j.setImage(Image_6);
    array_image_7j.setPos([((x_y_gains_per_angle[6][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[6][1] * image_eccentr_cm) * y_scale)]);
    array_image_7j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_7j.setImage(Image_7);
    array_image_8j.setPos([((x_y_gains_per_angle[7][0] * image_eccentr_cm) * x_scale), ((x_y_gains_per_angle[7][1] * image_eccentr_cm) * y_scale)]);
    array_image_8j.setSize([(image_side_cm * x_scale), (image_side_cm * y_scale)]);
    array_image_8j.setImage(Image_8);
    pointer_dot_0c.setPos([x_pointer_0, y_pointer_0]);
    pointer_dot_0c.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_1c.setPos([x_pointer_1, y_pointer_1]);
    pointer_dot_1c.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    pointer_dot_2c.setPos([x_pointer_2, y_pointer_2]);
    pointer_dot_2c.setSize([((dot_diam_dva * x_scale) / 2.0), ((dot_diam_dva * y_scale) / 2.0)]);
    sd_resp_real.keys = undefined;
    sd_resp_real.rt = undefined;
    _sd_resp_real_allKeys = [];
    // keep track of which components have finished
    retention_cue_and_test_intval_realComponents = [];
    retention_cue_and_test_intval_realComponents.push(polygon_34);
    retention_cue_and_test_intval_realComponents.push(array_image_1j);
    retention_cue_and_test_intval_realComponents.push(array_image_2j);
    retention_cue_and_test_intval_realComponents.push(array_image_3j);
    retention_cue_and_test_intval_realComponents.push(array_image_4j);
    retention_cue_and_test_intval_realComponents.push(array_image_5j);
    retention_cue_and_test_intval_realComponents.push(array_image_6j);
    retention_cue_and_test_intval_realComponents.push(array_image_7j);
    retention_cue_and_test_intval_realComponents.push(array_image_8j);
    retention_cue_and_test_intval_realComponents.push(pointer_dot_0c);
    retention_cue_and_test_intval_realComponents.push(pointer_dot_1c);
    retention_cue_and_test_intval_realComponents.push(pointer_dot_2c);
    retention_cue_and_test_intval_realComponents.push(sd_resp_real);
    retention_cue_and_test_intval_realComponents.push(reminder_real);
    
    for (const thisComponent of retention_cue_and_test_intval_realComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function retention_cue_and_test_intval_realRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'retention_cue_and_test_intval_real'-------
    // get current time
    t = retention_cue_and_test_intval_realClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *polygon_34* updates
    if (t >= 0.0 && polygon_34.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_34.tStart = t;  // (not accounting for frame time here)
      polygon_34.frameNStart = frameN;  // exact frame index
      
      polygon_34.setAutoDraw(true);
    }

    
    // *array_image_1j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_1j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_1j.tStart = t;  // (not accounting for frame time here)
      array_image_1j.frameNStart = frameN;  // exact frame index
      
      array_image_1j.setAutoDraw(true);
    }

    
    // *array_image_2j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_2j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_2j.tStart = t;  // (not accounting for frame time here)
      array_image_2j.frameNStart = frameN;  // exact frame index
      
      array_image_2j.setAutoDraw(true);
    }

    
    // *array_image_3j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_3j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_3j.tStart = t;  // (not accounting for frame time here)
      array_image_3j.frameNStart = frameN;  // exact frame index
      
      array_image_3j.setAutoDraw(true);
    }

    
    // *array_image_4j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_4j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_4j.tStart = t;  // (not accounting for frame time here)
      array_image_4j.frameNStart = frameN;  // exact frame index
      
      array_image_4j.setAutoDraw(true);
    }

    
    // *array_image_5j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_5j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_5j.tStart = t;  // (not accounting for frame time here)
      array_image_5j.frameNStart = frameN;  // exact frame index
      
      array_image_5j.setAutoDraw(true);
    }

    
    // *array_image_6j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_6j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_6j.tStart = t;  // (not accounting for frame time here)
      array_image_6j.frameNStart = frameN;  // exact frame index
      
      array_image_6j.setAutoDraw(true);
    }

    
    // *array_image_7j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_7j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_7j.tStart = t;  // (not accounting for frame time here)
      array_image_7j.frameNStart = frameN;  // exact frame index
      
      array_image_7j.setAutoDraw(true);
    }

    
    // *array_image_8j* updates
    if (t >= mem_display_to_test_array_duration_s && array_image_8j.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      array_image_8j.tStart = t;  // (not accounting for frame time here)
      array_image_8j.frameNStart = frameN;  // exact frame index
      
      array_image_8j.setAutoDraw(true);
    }

    
    // *pointer_dot_0c* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_0c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_0c.tStart = t;  // (not accounting for frame time here)
      pointer_dot_0c.frameNStart = frameN;  // exact frame index
      
      pointer_dot_0c.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_0c.status === PsychoJS.Status.STARTED || pointer_dot_0c.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_0c.setAutoDraw(false);
    }
    
    // *pointer_dot_1c* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_1c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_1c.tStart = t;  // (not accounting for frame time here)
      pointer_dot_1c.frameNStart = frameN;  // exact frame index
      
      pointer_dot_1c.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_1c.status === PsychoJS.Status.STARTED || pointer_dot_1c.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_1c.setAutoDraw(false);
    }
    
    // *pointer_dot_2c* updates
    if (t >= mem_display_to_cue_duration_s && pointer_dot_2c.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pointer_dot_2c.tStart = t;  // (not accounting for frame time here)
      pointer_dot_2c.frameNStart = frameN;  // exact frame index
      
      pointer_dot_2c.setAutoDraw(true);
    }

    frameRemains = mem_display_to_cue_duration_s + cue_duration_s - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pointer_dot_2c.status === PsychoJS.Status.STARTED || pointer_dot_2c.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pointer_dot_2c.setAutoDraw(false);
    }
    
    // *sd_resp_real* updates
    if (t >= response_time_s && sd_resp_real.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sd_resp_real.tStart = t;  // (not accounting for frame time here)
      sd_resp_real.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { sd_resp_real.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { sd_resp_real.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { sd_resp_real.clearEvents(); });
    }

    if (sd_resp_real.status === PsychoJS.Status.STARTED) {
      let theseKeys = sd_resp_real.getKeys({keyList: ['6', '7'], waitRelease: false});
      _sd_resp_real_allKeys = _sd_resp_real_allKeys.concat(theseKeys);
      if (_sd_resp_real_allKeys.length > 0) {
        sd_resp_real.keys = _sd_resp_real_allKeys[_sd_resp_real_allKeys.length - 1].name;  // just the last key pressed
        sd_resp_real.rt = _sd_resp_real_allKeys[_sd_resp_real_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *reminder_real* updates
    if (t >= 10 && reminder_real.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      reminder_real.tStart = t;  // (not accounting for frame time here)
      reminder_real.frameNStart = frameN;  // exact frame index
      
      reminder_real.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of retention_cue_and_test_intval_realComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function retention_cue_and_test_intval_realRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'retention_cue_and_test_intval_real'-------
    for (const thisComponent of retention_cue_and_test_intval_realComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('sd_resp_real.keys', sd_resp_real.keys);
    if (typeof sd_resp_real.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('sd_resp_real.rt', sd_resp_real.rt);
        routineTimer.reset();
        }
    
    sd_resp_real.stop();
    // the Routine "retention_cue_and_test_intval_real" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _conf_real_resp_allKeys;
var confidence_rating_real_2Components;
function confidence_rating_real_2RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'confidence_rating_real_2'-------
    t = 0;
    confidence_rating_real_2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    trialLetters = ["o", "k", "m"];
    event.clearEvents()
    trialLetter='nothing'
    text_14.setText("Don't forget: after each Same/Different choice, you should indicate how confident you are in the choice you just made.\n\nPlease use your left hand, using your ring finger if you're certain, your middle finger if you have doubt, or your index finger if it was just a guess.");
    conf_real_resp.keys = undefined;
    conf_real_resp.rt = undefined;
    _conf_real_resp_allKeys = [];
    // keep track of which components have finished
    confidence_rating_real_2Components = [];
    confidence_rating_real_2Components.push(text_14);
    confidence_rating_real_2Components.push(conf_real_resp);
    
    for (const thisComponent of confidence_rating_real_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function confidence_rating_real_2RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'confidence_rating_real_2'-------
    // get current time
    t = confidence_rating_real_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    
    keys = event.getKeys();
    if (keys.length) {
        for (var letterIndex = 0, _pj_a = 3; (letterIndex < _pj_a); letterIndex += 1) {
            trialLetter = trialLetters[letterIndex];
            if (_pj.in_es6(trialLetter, keys)) {
                thisExp.addData("confidenceKeyPressed", trialLetter);
                thisExp.addData("confidenceKeyPressedTime", timer.getTime());
                thisExp.nextEntry();
                continueRoutine = false;
            }
        }
        if (continueRoutine) {
            thisExp.addData("illegalKeyPressed", 1); //only reached if none of the letters in trialLetters was a hit
        }
    }
    
    // *text_14* updates
    if (t >= 10 && text_14.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_14.tStart = t;  // (not accounting for frame time here)
      text_14.frameNStart = frameN;  // exact frame index
      
      text_14.setAutoDraw(true);
    }

    
    // *conf_real_resp* updates
    if (t >= 0.0 && conf_real_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      conf_real_resp.tStart = t;  // (not accounting for frame time here)
      conf_real_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { conf_real_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { conf_real_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { conf_real_resp.clearEvents(); });
    }

    if (conf_real_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = conf_real_resp.getKeys({keyList: ['1', '2', '3'], waitRelease: false});
      _conf_real_resp_allKeys = _conf_real_resp_allKeys.concat(theseKeys);
      if (_conf_real_resp_allKeys.length > 0) {
        conf_real_resp.keys = _conf_real_resp_allKeys[_conf_real_resp_allKeys.length - 1].name;  // just the last key pressed
        conf_real_resp.rt = _conf_real_resp_allKeys[_conf_real_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of confidence_rating_real_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function confidence_rating_real_2RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'confidence_rating_real_2'-------
    for (const thisComponent of confidence_rating_real_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    real_trial_index = (real_trial_index + 1);
    psychoJS.experiment.addData('conf_real_resp.keys', conf_real_resp.keys);
    if (typeof conf_real_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('conf_real_resp.rt', conf_real_resp.rt);
        routineTimer.reset();
        }
    
    conf_real_resp.stop();
    // the Routine "confidence_rating_real_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var dataFolder;
var EndComponents;
function EndRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'End'-------
    t = 0;
    EndClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(8.000000);
    // update component parameters for each repeat
    thisExp.addData("reachedFinish", 1);
    if (expInfo["EyeTrack"]) {
        pylink.getEYELINK().setOfflineMode();
        pylink.pumpDelay(100);
        pylink.getEYELINK().closeDataFile();
        pylink.pumpDelay(500);
        dataFolder = (os.getcwd() + "/edfData/");
        if ((! os.path.exists(dataFolder))) {
            os.makedirs(dataFolder);
        }
        pylink.getEYELINK().receiveDataFile(dataFileName, (("edfData" + os.sep) + dataFileName));
        pylink.getEYELINK().close();
    }
    core.quit();
    
    // keep track of which components have finished
    EndComponents = [];
    EndComponents.push(finish);
    
    for (const thisComponent of EndComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function EndRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'End'-------
    // get current time
    t = EndClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *finish* updates
    if (t >= 0.0 && finish.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      finish.tStart = t;  // (not accounting for frame time here)
      finish.frameNStart = frameN;  // exact frame index
      
      finish.setAutoDraw(true);
    }

    frameRemains = 0.0 + 8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((finish.status === PsychoJS.Status.STARTED || finish.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      finish.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of EndComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function EndRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'End'-------
    for (const thisComponent of EndComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        const thisTrial = snapshot.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(snapshot);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(currentLoop) {
  return function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
