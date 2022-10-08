#modified version of dataset.py
from data_utils import nhlbi_data

from sklearn.model_selection import train_test_split

import numpy as np

from copy import deepcopy

class TimetoEventStudy:

	def __init__(self, outcomes, features,
							 cat_feats, num_feats,
							 study_name, study_description="",
							 validation_size=0.5,
							 random_seed=0):

		self.study_name = study_name
		self._outcomes = outcomes
		self._covariates = features
		self.cat_features = cat_feats
		self.num_features = num_feats
		self.study_description = study_description
		self.validation_size = validation_size
		self._split = 'all'

		self._valid_endpoints = list(outcomes.keys())
		self._endpoint = self._valid_endpoints[0] 

		self.random_seed = random_seed
		self.n = len(self._outcomes)
		self._split = 'all'

	def __doc__(self):
		return self.study_description 

	@property
	def valid_endpoints(self):
		return self._valid_endpoints

	@property
	def covariates(self):
	
		if self._split == 'all': idx = self.all_ix
		if self._split == 'train': idx = self.tr_ix
		elif self._split == 'test': idx = self.te_ix

		return self._covariates.loc[idx]

	@property
	def outcomes(self):
		
		if self._split == 'all': idx = self.all_ix
		if self._split == 'train': idx = self.tr_ix
		elif self._split == 'test': idx = self.te_ix

		return self._outcomes[self.endpoint].loc[idx]

	@property
	def endpoint(self):
		return self._endpoint

	@endpoint.setter
	def endpoint(self, ep):
		assert ep in self.valid_endpoints
		self._endpoint = ep

	@property
	def split(self):
		return self._split
	
	def train(self):
		self._split = 'train'

	def test(self):
		self._split = 'test'

	def all(self):
		self._split = 'all'

class ObservationalTimetoEventStudy(TimetoEventStudy):

	def __init__(self):
		raise NotImplementedError()


class InterventionalTimeToEventStudy(TimetoEventStudy):

	def __call__(self):
		return self.study_description

	def __init__(self, study_name, 
							 outcomes, features, interventions, 
							 cat_feats, num_feats, 
							 study_description="", random_seed=0):

		super().__init__(study_name=study_name, 
										 outcomes=outcomes,
 										 features=features, 
										 cat_feats=cat_feats,
										 num_feats=num_feats,
										 study_description=study_description,
										 random_seed=random_seed)
	
		self._interventions = interventions

		self.valid_interventions = list(interventions.keys())

		self._valid_arms = None

		if len(self.valid_interventions)==1: 
			self.intervention = self.valid_interventions[0]
		else:
			self._study_intervention = None

		self._control_arms = None
		self._treatment_arms = None



		self.set_split_indices()



	def set_split_indices(self):

		strata = []

		for endpoint in self._outcomes:
			strata.append(self._outcomes[endpoint].event.to_list())
		for intervention in self._interventions:
			strata.append(self._interventions[intervention].tolist())

		strata = np.array(strata)
		strata = strata.T

		strata = [tuple(strat) for strat in strata]

		self.all_ix = self._covariates.index

		self.tr_ix, self.te_ix = train_test_split(self._covariates.index,
																							test_size=self.validation_size,
																							stratify=strata,
																							random_state=self.random_seed)
	

	@property
	def treatment_assignment(self):

		assert self._treatment_arms is not None, "No Treatment Arm is Specified !!!"
		assert self._control_arms is not None, "No Control Arm is Specified !!!"

		if self._split == 'all': idx = self.all_ix
		if self._split == 'train': idx = self.tr_ix
		elif self._split == 'test': idx = self.te_ix


		interventions = deepcopy(self._interventions[self._study_intervention].loc[idx])

		for treatment_arm in self._treatment_arms:
			interventions.loc[interventions==treatment_arm] = self.treatment_arm

		for control_arm in self._control_arms:
			interventions.loc[interventions==control_arm] = self.control_arm	

		return interventions

	def set_arms(self, treatment_arms, control_arms):

		if type(treatment_arms) not in (tuple, list, set):
			treatment_arms = [treatment_arms]

		if type(control_arms) not in (tuple, list, set):
			control_arms = [control_arms]

		treatment_arms = sorted(treatment_arms)	
		control_arms = sorted(control_arms)

		assert len(set(treatment_arms) & set(control_arms)) == 0, "Treatment and Control Assignments Overlap !!!"
		assert set(treatment_arms).issubset(set(self._valid_arms)), "Assigned Treatment Arms is invalid" 	
		assert set(control_arms).issubset(set(self._valid_arms)), "Assigned Control Arms is invalid"

		self._treatment_arms = treatment_arms
		self._control_arms = control_arms

	@property
	def treatment_arm(self):

		assert self._treatment_arms is not None, "Please First Assign Treatment and Control Arms"
		return "/".join(self._treatment_arms)

	@property
	def control_arm(self):

		assert self._control_arms is not None, "Please First Assign Treatment and Control Arms"
		return "/".join(self._control_arms)

	@property
	def intervention(self):
		assert self._study_intervention is not None, "Please Set the Study Intervention !!!" 		
		return self._study_intervention 

	@intervention.setter
	def intervention(self, std):
		assert std in self.valid_interventions, "Intervention: "+ str(std)+" This is not a valid for this dataset !!!"
		self._study_intervention = std
		
		self._valid_arms = list(set(self._interventions[self._study_intervention]))

		self._treatment_arms = None
		self._control_arms = None

	@property
	def valid_arms(self):
		return self._valid_arms
	

	@staticmethod
	def load_study(study_name='', data_dir='./'):

		print("RELOADING THE DATASET!!!!!")
		
		intervention = 'Main Study'
		kwargs = {}		
		if (study_name == 'BARI2D-Cardiac'):
			load_dataset = nhlbi_data.load_bari_2d_dataset
			valid_endpoints = ('dthmistr', )
			kwargs['intervention'] = 'cardtrt'
		elif (study_name == 'BARI2D-Diabetic'):
			load_dataset = nhlbi_data.load_bari_2d_dataset
			kwargs['intervention'] = 'diabtrt'
			valid_endpoints = ('dthmistr', )
		elif (study_name == 'ACCORD-Hypertension'):
			load_dataset = nhlbi_data.load_accord_dataset
			valid_endpoints = ('po', )
		elif (study_name == 'ACCORD-Glycemia'):
			load_dataset = nhlbi_data.load_accord_dataset
			valid_endpoints = ('po', )
			intervention = 'Hypertension'
		elif (study_name == 'ACCORD-Lipid'):
			load_dataset = nhlbi_data.load_accord_dataset
			valid_endpoints = ('po', )
		elif (study_name == 'PEACE'):
			load_dataset = nhlbi_data.load_peace_dataset
			valid_endpoints = ('PRIMARY',)
		elif (study_name == 'OAT'):
			load_dataset = nhlbi_data.load_oat_dataset
		elif (study_name == 'AIMHIGH'):
			load_dataset = nhlbi_data.load_aimhigh_dataset
		elif (study_name == 'SPRINT'):
			load_dataset = nhlbi_data.load_sprint_dataset
			valid_endpoints = ('PRIMARY', 'DEATH', 'CVDDEATH')
		elif (study_name == 'ALLHAT'):
			load_dataset = nhlbi_data.load_allhat_antihypertensive_dataset
			valid_endpoints = ('CCVD', 'DEATH')
			intervention = 'Hypertension'
		else:
			raise NotImplementedError()

		outcomes = {}
		interventions = {}
		for endpoint in valid_endpoints: 
			outcomes_endpoint, features, interventions_, cat_feats, num_feats = load_dataset(location=data_dir,
																																											outcome=endpoint,
																																											**kwargs)
			outcomes[endpoint] = outcomes_endpoint
			interventions[intervention] = interventions_ 


		return InterventionalTimeToEventStudy(study_name=study_name, 
								  				  							outcomes=outcomes,
																			  	features=features,
																					interventions=interventions,
																					cat_feats=cat_feats,
																					num_feats=num_feats,
																					study_description=load_dataset.__doc__)


		#return("The "+self.study_name+" Study \n Size: "+ str(self.n))

	