#----- File description -----
"""
This script analyzes the results of the behavioural pilot data obtained online. It generates output figures (in 'results' folder) and statistical report (in prompt)
"""

#----- Imports -----
import pandas, os, numpy, copy, scipy
from scipy import stats
import matplotlib.pyplot as pl

#----- Define methods -----
def calculateAccuracy(retro_post_subconditions_in = None, subconditions_column_in = None, subconditions_labels_in = None, dv_column = None):
	"""
	Calculates the training performance per cue type and per level of intput factor (subconditions).

	Arguments:
		retro_post_subconditions_in : list of two lists of ints
			List of levels of subconditions; it includes two sublists corresponding to conditions within retro-cue and withing post-cue trials
		subconditions_column_in : string 
			The name of the column in training dataset pandas.DataFrame
		subconditions_labels_in : lists of two lists of strings 
			Lists of labels for levels of subconditions; must be the same dimensions as retro_post_subconditions_in
		dv_column : string
			THe name of the column of the dependent variable in the same pandas.DataFrame
   
	Returns:
		analysis_dictionaries : list of dict
			Dictionaries of the results per cue type and subcondition
	"""
	if retro_post_subconditions_in:
		retro_post_subconditions = copy.deepcopy(retro_post_subconditions_in)
	else:
		retro_post_subconditions = [[0.75, 1.6],[0.75, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]]
  
	if subconditions_column_in:
		sub_conditions_column = subconditions_column_in
	else:
		sub_conditions_column = 'mem_display_to_cue_duration_s'

	if subconditions_labels_in:
		subconditions_labels_in = subconditions_labels_in
	else:
		subconditions_labels_in = copy.deepcopy(retro_post_subconditions)

	analysis_dictionaries = [{},{}]
	for retro_post_index in [0,1]:	#0 for retro, 1 for post
		output_dict = {}
		for isi_index, mem_display_to_cue_duration_s in enumerate(retro_post_subconditions[retro_post_index]):
			this_cue_and_interval_indices=training_data.index[(training_data['retro_0_post_1_training']==retro_post_index) & (round(training_data[sub_conditions_column], 2) == mem_display_to_cue_duration_s)].tolist()
			this_cue_and_interval_actual = []
			for trial in this_cue_and_interval_indices:
				this_cue_and_interval_actual.append(training_data.at[trial, 'change_or_no_training'])

			this_cue_and_interval_response = []
			for trial in this_cue_and_interval_indices:
				this_cue_and_interval_response.append(training_data.at[trial, 'sd_training_resp.keys'])
			
			for idx,resp in enumerate(this_cue_and_interval_response):
				this_cue_and_interval_response[idx] = change_or_no_key_dict[resp]

			this_cue_and_interval_accuracy = []
			for idx in range(len(this_cue_and_interval_response)):
				this_cue_and_interval_accuracy.append(this_cue_and_interval_response[idx] == this_cue_and_interval_actual[idx])

			confidence_mapping = {'z':2, 'x': 1, 'c': 0}
			this_cue_and_interval_confidence = []
			for trial in this_cue_and_interval_indices:
				this_cue_and_interval_confidence.append(confidence_mapping[training_data.at[trial, 'conf_training_resp.keys']])

			if dv_column == 'acc':
				output_dict[subconditions_labels_in[retro_post_index][isi_index]] = numpy.sum(this_cue_and_interval_accuracy)/len(this_cue_and_interval_accuracy)
			elif dv_column == 'conf':
				output_dict[subconditions_labels_in[retro_post_index][isi_index]] = numpy.sum(this_cue_and_interval_confidence)/len(this_cue_and_interval_confidence)
		analysis_dictionaries[retro_post_index] = output_dict
	return analysis_dictionaries

def ci_within(df, indexvar, withinvars, measvar, confint=0.95,
                      copy=True):
    """ Compute CI / SEM correction factor
    Morey 2008, Cousinaueu 2005, Loftus & Masson, 1994
    Also see R-cookbook http://goo.gl/QdwJl
    Note. This functions helps to generate appropriate confidence
    intervals for repeated measure designs.
    Standard confidence intervals are are computed on normalized data
    and a correction factor is applied that prevents insanely small values.
    
    Arguments:
	    df : instance of pandas.DataFrame
    	    The data frame objetct.
	    indexvar : str
    	    The column name of of the identifier variable that representing subjects or repeated measures
	    withinvars : str | list of str
    	    The column names of the categorial data identifying random effects
	    measvar : str
    	    The column name of the response measure
	    confint : float
    	    The confidence interval
	    copy : bool
    	    Whether to copy the data frame or not.
        
    Return:
		pandas.DataFrame of results
    """
    if copy:
        df = df.copy(deep = True)

    # Apply Cousinaueu's method:
    # compute grand mean
    mean_ = df[measvar].mean()

    # compute subject means
    subj_means = df.groupby(indexvar)[measvar].mean().values
    subj_names = df.groupby(indexvar)[measvar].mean().index.tolist()
    for subj, smean_ in zip(subj_names, subj_means):
        # center
        #df[measvar][df[indexvar] == subj] -= smean_
        df.loc[df[indexvar] == subj, measvar] -= smean_
        # add grand average
        df.loc[df[indexvar] == subj, measvar] += mean_
        #df[measvar][df[indexvar] == subj] += mean_

    def sem(x):
        return x.std() / numpy.sqrt(len(x))

    def ci(x):
        se = sem(x)
        return se * scipy.stats.t.interval(confint, len(x) - 1)[1]

    aggfuncs = [numpy.mean, numpy.std, sem, ci]
    
    out = df.groupby(withinvars)[measvar].agg(aggfuncs)
    # compute & apply correction factor
    n_within = numpy.prod([len(df[k].unique()) for k in withinvars],
                       dtype= df[measvar].dtype)
    cf = numpy.sqrt(n_within / (n_within - 1))
    for k in ['sem', 'std', 'ci']:
        out[k] *= cf

    return out

#----- Set paths to data -----
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
inputs_path = 'data'
full_inputs_path = os.path.join(current, inputs_path)
outputs_path = 'results'
full_outputs_path = os.path.join(current, outputs_path)
os.makedirs(full_outputs_path, exist_ok=True)

#----- Data quality checks -----
enough_correct_retro_cues_check = False #80% correct retro-cue trials in Phase II

min_max_pix_per_cm_accepted=[20,80]
deviation_from_square_pix_accepted_prop=0.1
blue_square_bounces_till_end=8
pause_for_reading_before_blue_square_s=8.
time_per_blue_square_crossing_s=8.
total_time_blind_spot_part=blue_square_bounces_till_end*time_per_blue_square_crossing_s+pause_for_reading_before_blue_square_s

min_nospace_duration_for_consideration_s=.75
min_nospace_duration_for_consideration_last_value_s=.4	#this is the duration, not between two consecutive moments where a space is registered, but between the last time a space was registered and the end of the sweep. Allow slightly shorter interval here and still count it as a key release. Is only used if there is not other blank elsewhere

device_options=['Laptop','Desktop']

data_files=[element for element in os.listdir(os.path.join(full_inputs_path)) if '.csv' in element]

complete_ID_paths = [] #This list will contain paths to data that pass the checks
num_excluded_scaling=0
for data_file in data_files:
	
	one_obs_output_dict={}
	
	prolific_ID = data_file.split('_')[0]
	data_file_path = os.path.join(full_inputs_path, data_file)

	try:
		data = pandas.read_csv(data_file_path, low_memory=False)
		finished_dataset = False
		complete_dataset = False
		pixels_are_square = False

		if enough_correct_retro_cues_check:
			enough_correct_retro_cues = False
		else:
			enough_correct_retro_cues = True
    
		if 'reachedFinish' in data.columns:		#reachedFinish is a column name that will exist if the participant reached the end of the experiment, because it is created at the end. We will probably make sure all experiments do that, so this line may remain the same for other experiments
			finished_dataset=True
		else: 
			print('Excluding participant '+ prolific_ID + ' because training was not finished')

		if 'short_interval_training_finished' in data.columns:
			complete_dataset=True

			one_obs_output_dict['prolific_ID_from_filename']=prolific_ID

			date=data['date'].dropna().tolist()[0]
			one_obs_output_dict['date']=date
			one_obs_output_dict['mapping'] = data['mapping'].dropna().tolist()[0]
    
			#Bank card check
			x_pix_per_cm=data['X Scale'].dropna().tolist()[0]
			y_pix_per_cm=data['Y Scale'].dropna().tolist()[0]
		
			if abs(x_pix_per_cm-y_pix_per_cm)>(numpy.average([x_pix_per_cm,y_pix_per_cm])*deviation_from_square_pix_accepted_prop):
				print('Excluding participant '+ prolific_ID +' because credit card scaling procedure suggests pixels that aren\'t square: ' + ' '.join([str(x_pix_per_cm), str(y_pix_per_cm)]))
				num_excluded_scaling=num_excluded_scaling+1
			elif x_pix_per_cm>min_max_pix_per_cm_accepted[1] or y_pix_per_cm>min_max_pix_per_cm_accepted[1] or x_pix_per_cm<min_max_pix_per_cm_accepted[0] or y_pix_per_cm<min_max_pix_per_cm_accepted[0]:
				print('Excluding participant '+ prolific_ID +' because credit card scaling procedure suggests implausible resolution.')
				num_excluded_scaling=num_excluded_scaling+1
			else:
				pixels_are_square = True

			if finished_dataset:
				if data['reached_criterion'].dropna().tolist()[-1] == 1:
					enough_correct_retro_cues = True

		if finished_dataset and pixels_are_square and complete_dataset and enough_correct_retro_cues:			
			complete_ID_paths.append(data_file_path)

	except ValueError:
		print("Excluding participant " + prolific_ID + " because of empty data file")


#----- Analyses -----
all_cueTypes = ['retro','post']

#We'll loop through analyses each with their individual values for important variables
analyses = {'per_isi': {'colnames': ['sub', 'cueType', 'isi', 'acc'],
                        'withinvars': ['cueType','isi'],
                        'retro_post_subconditions_in': None, 
                        'subconditions_column_in': None, 
                        'subconditions_labels_in': None,
                        'vars': {'dv': {'label': 'Accuracy', 'column': 'acc'},
                                    'iv_x': {'label': 'Inter-stimulus interval', 'column': 'isi'},
                                    'iv_colour': {'label': 'Cue type', 'column': 'cueType'}}},
            'retroConfidence': {'colnames': ['sub', 'cueType', 'correct_or_no_training', 'conf'],
                                'withinvars': ['cueType', 'correct_or_no_training'],
                                'retro_post_subconditions_in': [[0,1],[]], 
                                'subconditions_column_in': 'correct_or_no_training', 
                                'subconditions_labels_in': [['retro-cue\nincorrect', 'retro-cue\ncorrect'],[]],
                                'vars': {'dv': {'label': 'Confidence rating', 'column': 'conf'},
                                         'iv_x': {'label': '', 'column': 'correct_or_no_training'},
                                         'iv_colour': {'label': 'Cue type', 'column': 'cueType'}}}}

for analysis in analyses:
	colnames = copy.deepcopy(analyses[analysis]['colnames'])
	allData = pandas.DataFrame(columns = colnames)
	for complete_ID_path in complete_ID_paths:
	    #Read input data
		training_data = pandas.read_csv(complete_ID_path, low_memory = False)
		subject = training_data['participant'][0]
		#Keeping track of button mapping for given subject
		participant_response_mapping=training_data['mapping'].dropna().tolist()[0]
		if participant_response_mapping==23:
			change_or_no_key_dict={'down':1.0,'left':0.0}
		elif participant_response_mapping==32:
			change_or_no_key_dict={'left':1.0,'down':0.0}
			
		#marking correct and wrong responses
		training_data.loc[training_data['training_trial_index'] >= 0, 'change_or_no_actual_training'] = [change_or_no_key_dict[one_response] for one_response in training_data.loc[training_data['training_trial_index'] >= 0, 'sd_training_resp.keys']]
		training_data['correct_or_no_training'] = [int(row) for row in training_data['change_or_no_actual_training'] == training_data['change_or_no_training']]

		#Use an existing (copied) function to calculate accuracy
		for cueType_index in range(len(calculateAccuracy(analyses[analysis]['retro_post_subconditions_in'], analyses[analysis]['subconditions_column_in'], analyses[analysis]['subconditions_labels_in'], analyses[analysis]['vars']['dv']['column']))):
			this_cueType_accuracies = calculateAccuracy(analyses[analysis]['retro_post_subconditions_in'], analyses[analysis]['subconditions_column_in'], analyses[analysis]['subconditions_labels_in'], analyses[analysis]['vars']['dv']['column'])[cueType_index]
			for this_isi in this_cueType_accuracies:
				allData.loc[len(allData.index)] = [subject, all_cueTypes[cueType_index], this_isi, this_cueType_accuracies[this_isi]]

	#to save out appropriate values for analyses
	if analysis == 'per_isi':
		allData_per_isi = allData.copy(deep=True)
	elif analysis == 'retroConfidence':
		allData_retroConfidence = allData.copy(deep=True)
	#Calculate confidence intervals using existing (copied) function
	CIs = ci_within(df = allData, indexvar = 'sub', withinvars = analyses[analysis]['withinvars'], measvar = analyses[analysis]['vars']['dv']['column'])
	#Setting up variables to control plotting
	cueType_names = {'retro': 'retro', 'post': 'post'}
	colours = {'retro': 'red', 'post': 'blue'}

	cueTypes = [this_pdIndex for this_pdIndex in CIs.index.levels if this_pdIndex.name == 'cueType'][0].tolist()
	jitter = {'retro': -0.03, 'post': 0.03} #to set put different cue accuracies apart when they have the same ISIs
	if len(cueTypes) == 1:
		jitter = {'retro': 0.0, 'post': 0.0}

	#Begin plotting
	figsize_x_min = 7.
	figsize_x = max(max([len(CIs.loc[cueType]) for cueType in cueTypes]) * 1., figsize_x_min)
	ci_0figure, the_plot = pl.subplots(figsize = (figsize_x, 10))

	big_text = 35
	small_text = 26
	if analyses[analysis]['vars']['dv']['column'] == 'acc':
		pl.axhline(0.5, color = 'grey', linewidth = 3, linestyle='dashed') #random guessing line
		pl.text((max([len(CIs.loc[cueType]) for cueType in cueTypes]) - 1) / 2., 0.46, 'Chance level', color = 'grey', size = small_text, horizontalalignment='center', verticalalignment='center')
  
	cue_duration_dict_per_cueType = []
	for this_cueType in cueTypes:
		this_cueType_ci = CIs.loc[this_cueType] #Taking all subconditions for the current cue type
		cue_duration_dict = {dur: dur_index for dur_index, dur in enumerate(this_cueType_ci.index)}
		cue_duration_dict_per_cueType.append(copy.deepcopy(cue_duration_dict))
		if len(cue_duration_dict) >= max([len(durs) for durs in cue_duration_dict_per_cueType]):
			max_cue_duration_dict = copy.deepcopy(cue_duration_dict)
		pl.scatter([-0.0015 + cue_duration_dict[this_duration] + jitter[this_cueType] for this_duration in this_cueType_ci.index], this_cueType_ci['mean'], color = colours[this_cueType], label = cueType_names[this_cueType], s = 80) #plot dots for accuracies
		pl.plot([cue_duration_dict[this_duration] + jitter[this_cueType] for this_duration in this_cueType_ci.index], this_cueType_ci['mean'], color = colours[this_cueType], linewidth = 3)
		for this_duration in this_cueType_ci.index:
			pl.errorbar(cue_duration_dict[this_duration] + jitter[this_cueType], this_cueType_ci.loc[this_duration]['mean'], yerr = this_cueType_ci.loc[this_duration]['ci'], color = colours[this_cueType], linewidth = 3, capsize = 7, capthick=2)

	#This is to produce prettier axes
	xs = [this_pdIndex for this_pdIndex in CIs.index.levels if this_pdIndex.name == analyses[analysis]['vars']['iv_x']['column']][0].tolist()
	pl.xticks(ticks = [max_cue_duration_dict[dur] for dur in xs], labels = xs, fontsize = small_text)
	if analyses[analysis]['vars']['dv']['column'] == 'acc':
		pl.yticks(numpy.round(numpy.arange(0,1.1,0.1),1),fontsize = small_text)
	the_plot.tick_params(which='major', labelsize=small_text, length = 7, width = 3)

	if figsize_x_min == figsize_x:
		pl.xlim(-0.5, max([len(CIs.loc[cueType]) for cueType in cueTypes]) - 1. + 0.5)
	if analyses[analysis]['vars']['dv']['column'] == 'acc':
		pl.ylim(0.3,1.05)
	elif analyses[analysis]['vars']['dv']['column'] == 'conf':
		pl.ylim(.9,2.)
	for axis in ['top','bottom','left','right']:
		the_plot.spines[axis].set_linewidth(3) #change the thickness of plot borders

	# Get rid of some reference lines
	pl.grid(axis = 'x')

	if analysis != 'retroConfidence':
		pl.legend(title = analyses[analysis]['vars']['iv_colour']['label'], title_fontsize = big_text, prop={'size': small_text}, frameon = False)
	pl.xlabel(analyses[analysis]['vars']['iv_x']['label'],labelpad=20, fontsize=big_text)
	pl.ylabel(analyses[analysis]['vars']['dv']['label'],labelpad=20, fontsize=big_text)

	pl.savefig(os.path.join(full_outputs_path, analysis), bbox_inches='tight')
	pl.close()


#----- Statistics -----
#Now, to output statistics
from statsmodels.stats.anova import AnovaRM
#post vs retro averaged across ISIs
print('Cue type effect on accuracy:')
aov = AnovaRM(
    allData_per_isi,
    depvar='acc',
    subject='sub',
    within=['cueType'],
    aggregate_func='mean'
).fit()
print(aov)

#Across all post-cues
print('Effect of inter-stimulus interval on accuracy in post-cue trials:')
aov = AnovaRM(
    allData_per_isi[allData_per_isi['cueType'] == 'post'],
    depvar='acc',
    subject='sub',
    within=['isi'],
    aggregate_func='mean'
).fit()
print(aov)

#Confidence rating in incorrect and correct retro-trials
print('The effect of trial reponse on confidence rating:')
ttest1 = allData_retroConfidence[(allData_retroConfidence['cueType'] == 'retro') & (allData_retroConfidence['correct_or_no_training'] == analyses['retroConfidence']['subconditions_labels_in'][0][0])]['conf']
ttest2 = allData_retroConfidence[(allData_retroConfidence['cueType'] == 'retro') & (allData_retroConfidence['correct_or_no_training'] == analyses['retroConfidence']['subconditions_labels_in'][0][1])]['conf']
ttest = stats.ttest_rel(ttest1, ttest2)
print(ttest)