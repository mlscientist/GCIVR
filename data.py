import torch
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn import model_selection
import warnings
from sklearn.cluster import KMeans
import random
import os




class Communities(Dataset):

	@property
	def train_labels(self):
		warnings.warn("train_labels has been renamed targets")
		return self.targets

	@property
	def test_labels(self):
		warnings.warn("test_labels has been renamed targets")
		return self.targets

	@property
	def train_data(self):
		warnings.warn("train_data has been renamed data")
		return self.data

	@property
	def test_data(self):
		warnings.warn("test_data has been renamed data")
		return self.data

	def __init__(self,data, split='train'):
		self.split=split
		self.data, self.target, self.group, self.group_memberships_list, self.group_info = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.target[idx], self.group[idx], self.group_memberships_list[:,idx].T

	def group_membership(self, group_thresholds):
		return (self.group[:, 0] > group_thresholds[0]) & (
					self.group[:, 1] > group_thresholds[1]) & (
							self.group[:, 2] > group_thresholds[2])

	def to(self, device):
		if self.data.device.type != device.type:
			self.data = self.data.to(device)
			self.target = self.target.to(device)
			self.group = self.group.to(device)
			self.group_memberships_list = self.group_memberships_list.to(device)

def group_membership_thresholds(
    group_feature_train, group_feature_vali, group_feature_test, thresholds):
  """Returns the group membership vectors on train, test and vali sets."""
  group_memberships_list_train_ = []
  group_memberships_list_vali_ = []
  group_memberships_list_test_ = []
  group_thresholds_list = []

  for t1 in thresholds[0]:
    for t2 in thresholds[1]:
      for t3 in thresholds[2]:
        group_membership_train = (group_feature_train[:, 0] > t1) & (
            group_feature_train[:, 1] > t2) & (group_feature_train[:, 2] > t3)
        group_membership_vali = (group_feature_vali[:, 0] > t1) & (
            group_feature_vali[:, 1] > t2) & (group_feature_vali[:, 2] > t3)
        group_membership_test = (group_feature_test[:, 0] > t1) & (
            group_feature_test[:, 1] > t2) & (group_feature_test[:, 2] > t3)
        if (np.mean(group_membership_train) <= 0.01) or (
            np.mean(group_membership_vali) <= 0.01) or (
                np.mean(group_membership_test) <= 0.01):
          # Only consider groups that are at least 1% in size.
          continue
        group_memberships_list_train_.append(group_membership_train)
        group_memberships_list_vali_.append(group_membership_vali)
        group_memberships_list_test_.append(group_membership_test)
        group_thresholds_list.append([t1, t2, t3])

  group_memberships_list_train_ = np.array(group_memberships_list_train_)
  group_memberships_list_vali_ = np.array(group_memberships_list_vali_)
  group_memberships_list_test_ = np.array(group_memberships_list_test_)
  group_thresholds_list = np.array(group_thresholds_list)

  return (group_memberships_list_train_, group_memberships_list_vali_,
          group_memberships_list_test_, group_thresholds_list)


def load_data():
  """Loads and returns data."""
  # List of column names in the dataset.
  column_names = ["state", "county", "community", "communityname", "fold",
                  "population", "householdsize", "racepctblack", "racePctWhite",
                  "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
                  "agePct16t24", "agePct65up", "numbUrban", "pctUrban",
                  "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc",
                  "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
                  "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap",
                  "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
                  "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
                  "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
                  "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
                  "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
                  "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
                  "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
                  "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig",
                  "PctImmigRecent", "PctImmigRec5", "PctImmigRec8",
                  "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                  "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
                  "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
                  "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
                  "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR",
                  "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc",
                  "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt",
                  "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
                  "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
                  "RentHighQ", "MedRent", "MedRentPctHousInc",
                  "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
                  "NumStreet", "PctForeignBorn", "PctBornSameState",
                  "PctSameHouse85", "PctSameCity85", "PctSameState85",
                  "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
                  "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
                  "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol",
                  "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
                  "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                  "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea",
                  "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
                  "LemasPctPolicOnPatr", "LemasGangUnitDeploy",
                  "LemasPctOfficDrugUn", "PolicBudgPerPop",
                  "ViolentCrimesPerPop"]

  dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
  # Read dataset from the UCI web repository and assign column names.
  data_df = pd.read_csv(dataset_url, sep=",", names=column_names,
                        na_values="?")

  # Make sure there are no missing values in the "ViolentCrimesPerPop" column.
  assert not data_df["ViolentCrimesPerPop"].isna().any()

  # Binarize the "ViolentCrimesPerPop" column and obtain labels.
  crime_rate_70_percentile = data_df["ViolentCrimesPerPop"].quantile(q=0.7)
  labels_df = (data_df["ViolentCrimesPerPop"] >= crime_rate_70_percentile)

  # Now that we have assigned binary labels,
  # we drop the "ViolentCrimesPerPop" column from the data frame.
  data_df.drop(columns="ViolentCrimesPerPop", inplace=True)

  # Group features.
  groups_df = pd.concat(
      [data_df["racepctblack"], data_df["racePctAsian"],
       data_df["racePctHisp"]], axis=1)

  # Drop categorical features.
  data_df.drop(
      columns=["state", "county", "community", "communityname", "fold"],
      inplace=True)

  # Handle missing features.
  feature_names = data_df.columns
  for feature_name in feature_names:
    missing_rows = data_df[feature_name].isna()
    if missing_rows.any():
      data_df[feature_name].fillna(0.0, inplace=True)  # Fill NaN with 0.
      missing_rows.rename(feature_name + "_is_missing", inplace=True)
      # Append boolean "is_missing" feature.
      data_df = data_df.join(missing_rows)

  labels = labels_df.values.astype(np.float32)
  groups = groups_df.values.astype(np.float32)
  features = data_df.values.astype(np.float32)

  # Set random seed so that the results are reproducible.
  np.random.seed(121212)

  # Train, vali and test indices.
  train_indices, test_indices = model_selection.train_test_split(
      range(features.shape[0]), test_size=0.25)
  train_indices, vali_indices = model_selection.train_test_split(
      train_indices, test_size=1./3.)

  # Train features, labels and protected groups.
  x_train = features[train_indices, :]
  y_train = labels[train_indices]
  z_train = groups[train_indices]
  
  

  # Vali features, labels and protected groups.
  x_val = features[vali_indices, :]
  y_val = labels[vali_indices]
  z_val = groups[vali_indices]

  

  # Test features, labels and protected groups.
  x_test = features[test_indices, :]
  y_test = labels[test_indices]
  z_test = groups[test_indices]

  # Group Thresholds for 3 Groups
  group_threshold_range = []
  for jj in range(3):
    group_threshold_range.append([np.quantile(
        z_train[:, jj], kk) for kk in np.arange(0.05, 1.0,0.1)])

  # Group memberships based on group thresholds.
  group_info = group_membership_thresholds(
      z_train, z_val, z_test, group_threshold_range)

  data_train = (torch.tensor(x_train), torch.tensor(y_train), torch.tensor(z_train), torch.tensor(group_info[0]), group_info[3])
  data_val = (torch.tensor(x_val),torch.tensor(y_val), torch.tensor(z_val), torch.tensor(group_info[1]), group_info[3])
  data_test = (torch.tensor(x_test), torch.tensor(y_test), torch.tensor(z_test), torch.tensor(group_info[2]), group_info[3])

  return Communities(data_train,'train'), Communities(data_val, 'val'), Communities(data_test,'test')



class Adult(Dataset):

	@property
	def train_labels(self):
		warnings.warn("train_labels has been renamed targets")
		return self.targets

	@property
	def test_labels(self):
		warnings.warn("test_labels has been renamed targets")
		return self.targets

	@property
	def train_data(self):
		warnings.warn("train_data has been renamed data")
		return self.data

	@property
	def test_data(self):
		warnings.warn("test_data has been renamed data")
		return self.data

	def __init__(self, data, split='train', uniform_groups=False, min_group_frac=0.05, use_noise_array=True,
								group_features_type='full_group_vec', num_group_clusters=100):
		self.split=split
		self.uniform_groups = uniform_groups
		self.min_group_frac = min_group_frac
		self.use_noise_array = use_noise_array
		self.group_features_type = group_features_type
		self.num_group_clusters = num_group_clusters
		self.data_df, self.feature_names, self.label_name, self.protected_columns, self.proxy_columns = data
		self.data, self.targets, self.proxy_groups_tensor , self.true_groups_tensor = self.extract_features()
		self.num_groups = self.proxy_groups_tensor.shape[1]

		self.num_examples = len(self.data_df)
		self.num_features = len(self.feature_names)
		self.noise_array = None
		if self.use_noise_array and not self.uniform_groups:
			self.noise_array = self.get_noise_array()

		
		# Get number of group features.
		if self.split == 'train':
			self.kmeans_model = None
			self.num_group_features = None
			if self.group_features_type == 'full_group_vec':
				self.num_group_features = self.num_examples
			elif self.group_features_type == 'size_alone':
				self.num_group_features = 1
			elif self.group_features_type == 'size_and_pr':
				self.num_group_features = 2
			elif self.group_features_type == 'avg_features':
				self.num_group_features = self.num_features + 1
			elif self.group_features_type == 'kmeans':
				self.kmeans_model = KMeans(
						n_clusters=self.num_group_clusters, random_state=0).fit(self.data)
				self.num_group_features = self.num_group_clusters
		

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.targets[idx]

	def to(self, device):
		if self.data.device.type != device.type:
			self.data = self.data.to(device)
			self.targets = self.targets.to(device)
			self.proxy_groups_tensor = self.proxy_groups_tensor.to(device)
			self.true_groups_tensor = self.true_groups_tensor.to(device)
		return
    
	def extract_features(self):
    
		"""Extracts features from dataframe."""
		features = []
		for feature_name in self.feature_names:
			features.append(self.data_df[feature_name].values.astype(float))
		labels = self.data_df[self.label_name].values.astype(float)
		proxy_groups_tensor = None
		true_groups_tensor = None
		if self.uniform_groups:
			proxy_groups_tensor = torch.tensor(generate_proxy_groups_uniform(
					len(self.data_df), min_group_frac=self.min_group_frac)).long()
			true_groups_tensor =  torch.tensor(generate_proxy_groups_uniform(
					len(self.data_df), min_group_frac=self.min_group_frac)).long()
		else:
			proxy_groups = []
			true_groups = []
			for group_name in self.proxy_columns:
				proxy_groups.append(self.data_df[group_name].values.astype(float))
			for group_name in self.protected_columns:
				true_groups.append(self.data_df[group_name].values.astype(float))
			proxy_groups_tensor = torch.tensor(proxy_groups).T.long()
			true_groups_tensor = torch.tensor(true_groups).long()
		return torch.tensor(features).T.float(), torch.tensor(labels).reshape(
				(-1, 1)).squeeze(), proxy_groups_tensor, true_groups_tensor

	def extract_group_features(self):
		"""Extracts features from groups."""
		input_groups_t = self.proxy_groups_tensor.T
		all_group_features = []
		for group_indices in input_groups_t:
			group_fraction = group_indices.float().mean()
			if self.group_features_type == 'size_alone':
				all_group_features.append(group_fraction.unsqueeze(0))
			elif self.group_features_type == 'size_and_pr':
				mean_labels = torch.mean(self.targets[group_indices == 1], dim=0)
				mean_features = torch.cat((mean_labels, group_fraction.unsqueeze()))
				all_group_features.append(mean_features)
			elif self.group_features_type == 'avg_features':
				mean_features = torch.mean(self.data[group_indices == 1], axis=0)
				mean_features = torch.cat((mean_features, group_fraction.unsqueeze()))
				all_group_features.append(mean_features)
			elif self.group_features_type == 'full_group_vec':
				# print('group_indices shape', group_indices.shape)
				all_group_features.append(group_indices)
			elif self.group_features_type == 'kmeans':
				group_xs = self.data[group_indices == 1]
				clusters = self.kmeans_model.predict(group_xs)
				# Counter doesn't include clusters with count 0.
				# Need to manually add 0 counts for clusters that aren't seen.
				count_dict = dict.fromkeys(range(self.num_group_clusters), 0)
				count_dict.update(Counter(clusters))
				compressed_clusters = np.fromiter(count_dict.values(), dtype='float32')
				all_group_features.append(torch.tensor(compressed_clusters))
		return torch.stack(all_group_features)
	
	def get_noise_array(self, print_noises=False):
		"""Returns an array where noise_params[k][j] = P(G=j | hatG=k)."""
		noise_array = torch.zeros((self.num_groups, self.num_groups))
		for k in range(self.num_groups):
			for j in range(self.num_groups):
				frac = np.sum(
						self.data_df[self.protected_columns[j]] * self.data_df[self.proxy_columns[k]]) / np.sum(
								self.data_df[self.proxy_columns[k]])
				noise_array[k][j] = frac
				if print_noises:
					print('P(G=%d | hatG=%d) = %f' % (j, k, frac))
		return noise_array


########### Load Adult Data ##############
def load_data_adult(noise_level, 
					uniform_groups=False, 
					min_group_frac=0.05, 
					use_noise_array=True,
					group_features_type='full_group_vec', 
					num_group_clusters=100,
					data_seed=12345):
	"""Loads Adult dataset."""
	df = preprocess_data_adult()
	df = add_proxy_columns_adult(df)
	label_name = 'label'
	feature_names = list(df.keys())
	feature_names.remove(label_name)
	protected_columns = ['race_White', 'race_Black', 'race_Other_combined']
	for column in protected_columns:
		feature_names.remove(column)
	proxy_columns = get_proxy_column_names(protected_columns, noise_level)
	feature_names = remove_saved_noise_levels(
			protected_columns, feature_names, keep_noise_level=noise_level)
	# return df, feature_names, label_name, protected_columns, proxy_columns
	train_df, test_df = train_test_split(
      df, 0.2, seed=data_seed)
	train_data = (train_df, feature_names, label_name, protected_columns, proxy_columns)
	test_data  = (test_df,  feature_names, label_name, protected_columns, proxy_columns)
	train_dataset = Adult(train_data, split='train', uniform_groups=uniform_groups, min_group_frac=min_group_frac,
												use_noise_array=use_noise_array, group_features_type=group_features_type, 
												num_group_clusters=num_group_clusters)
	test_dataset  = Adult(test_data, split='test', uniform_groups=uniform_groups, min_group_frac=min_group_frac,
												use_noise_array=use_noise_array, group_features_type=group_features_type, 
												num_group_clusters=num_group_clusters)
	return train_dataset, test_dataset
	

def preprocess_data_adult():
  """Preprocess Adult dataset."""
  categorical_columns = [
      'workclass', 'education', 'marital_status', 'occupation', 'relationship',
      'race', 'gender', 'native_country'
  ]
  continuous_columns = [
      'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
  ]
  columns = [
      'age', 'workclass', 'fnlwgt', 'education', 'education_num',
      'marital_status', 'occupation', 'relationship', 'race', 'gender',
      'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
      'income_bracket'
  ]
  label_column = 'label'

  train_df_raw = pd.read_csv(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
      names=columns,
      skipinitialspace=True)
  test_df_raw = pd.read_csv(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
      names=columns,
      skipinitialspace=True,
      skiprows=1)

  train_df_raw[label_column] = (
      train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  test_df_raw[label_column] = (
      test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  # Preprocessing Features
  pd.options.mode.chained_assignment = None  # default='warn'

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_train_df,
                                   input_test_df,
                                   categorical_columns=None):

    def fix_columns(input_train_df, input_test_df):
      test_df_missing_cols = set(input_train_df.columns) - set(
          input_test_df.columns)
      for c in test_df_missing_cols:
        input_test_df[c] = 0
        train_df_missing_cols = set(input_test_df.columns) - set(
            input_train_df.columns)
      for c in train_df_missing_cols:
        input_train_df[c] = 0
        input_train_df = input_train_df[input_test_df.columns]
      return input_train_df, input_test_df

    # Binarize categorical columns.
    binarized_train_df = pd.get_dummies(
        input_train_df, columns=categorical_columns)
    binarized_test_df = pd.get_dummies(
        input_test_df, columns=categorical_columns)
    # Make sure the train and test dataframes have the same binarized columns.
    fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                binarized_test_df)
    return fixed_train_df, fixed_test_df

  def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
    assert (num_quantiles is None or bins is None)
    if num_quantiles is not None:
      _, bins_quantized = pd.qcut(
          input_train_df[continuous_column_name],
          num_quantiles,
          retbins=True,
          labels=False)
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins_quantized, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins_quantized, labels=False)
    elif bins is not None:
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins, labels=False)

  # Filter out all columns except the ones specified.
  train_df = train_df_raw[categorical_columns + continuous_columns +
                          [label_column]]
  test_df = test_df_raw[categorical_columns + continuous_columns +
                        [label_column]]

  # Bucketize continuous columns.
  bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
  bucketize_continuous_column(
      train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
  bucketize_continuous_column(
      train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
  bucketize_continuous_column(
      train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
  bucketize_continuous_column(
      train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
  train_df, test_df = binarize_categorical_columns(
      train_df,
      test_df,
      categorical_columns=categorical_columns + continuous_columns)
  full_df = train_df.append(test_df,ignore_index=True)
  full_df['race_Other_combined'] = full_df['race_Amer-Indian-Eskimo'] + full_df[
      'race_Asian-Pac-Islander'] + full_df['race_Other']
  return full_df

def add_proxy_columns_adult(df):
  """Adds noisy proxy columns to adult dataset."""
  proxy_noises = [0.1, 0.2, 0.3, 0.4, 0.5]
  protected_columns = ['race_White', 'race_Black', 'race_Other_combined']
  # Generate proxy groups.
  for noise in proxy_noises:
    df = generate_proxy_columns(df, protected_columns, noise_param=noise)
  return df

def generate_proxy_columns(df, protected_columns, noise_param=1):
  """Generates noisy proxy columns from binarized protected columns."""
  proxy_columns = get_proxy_column_names(protected_columns, noise_param)
  num_datapoints = len(df)
  num_groups = len(protected_columns)
  noise_idx = random.sample(
      range(num_datapoints), int(noise_param * num_datapoints))
  df_proxy = df.copy()
  for i in range(num_groups):
    df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
  for j in noise_idx:
    group_index = -1
    for i in range(num_groups):
      if df_proxy[proxy_columns[i]][j] == 1:
        df_proxy.at[j, proxy_columns[i]] = 0
        group_index = i
        allowed_new_groups = list(range(num_groups))
        allowed_new_groups.remove(group_index)
        new_group_index = random.choice(allowed_new_groups)
        df_proxy.at[j, proxy_columns[new_group_index]] = 1
        break
    if group_index == -1:
      print('missing group information for datapoint ', j)
  return df_proxy

# Split into train/test
def train_test_split(df, train_fraction, seed=None):
  """Split the whole dataset into train/test."""
  if seed is not None:
    np.random.seed(seed=seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_fraction * m)
  train = df.iloc[perm[:train_end]]
  test = df.iloc[perm[train_end:]]
  return train, test

# Get proxy columns.
def get_proxy_column_names(protected_columns, noise_param, noise_index=''):
  """Gets proxy column names."""
  return [
      'PROXY' + noise_index + '_' + '%0.2f_' % noise_param + column_name
      for column_name in protected_columns
  ]

def remove_saved_noise_levels(protected_columns, feature_names,
                              keep_noise_level):
  """Removes saved noise level columns from feature columns."""
  saved_noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
  saved_noise_levels.remove(keep_noise_level)
  for noise_level in saved_noise_levels:
    proxy_columns = get_proxy_column_names(protected_columns, noise_level)
    for column in proxy_columns:
      feature_names.remove(column)
  return feature_names

def generate_proxy_groups_uniform(num_examples, min_group_frac=0.05):
  """Generate proxy groups within noise noise_param."""

  # Generate a random array of the same shape as input groups. Each column
  # in the array is a a random binary vector where the number of 1's is at least
  # min_group_size.
  group_frac = np.random.uniform(min_group_frac, 1)
  num_in_group = int(num_examples * group_frac)
  group_assignment = np.array([0] * (num_examples - num_in_group) +
                              [1] * num_in_group)
  np.random.shuffle(group_assignment)
  return group_assignment.reshape((-1, 1))











################ Learning to rank dataset ##################
class MSLTR(Dataset):

	@property
	def train_labels(self):
		warnings.warn("train_labels has been renamed targets")
		return self.group_pairs

	@property
	def test_labels(self):
		warnings.warn("test_labels has been renamed targets")
		return self.group_pairs

	@property
	def train_data(self):
		warnings.warn("train_data has been renamed data")
		return self.features_pairs

	@property
	def test_data(self):
		warnings.warn("test_data has been renamed data")
		return self.features_pairs

	def __init__(self, file_path, split='train', data_size=None, seed=42):
		self.split=split # 'train', 'vali', or 'test'
		self.path = os.path.join(file_path, self.split + '.txt')
		self.seed = seed
		self.data_size=data_size
		np.random.seed(self.seed)

		self.features, self.queries, self.groups, self.labels, \
			self.data_dimension, self.multiplier_dimension, self.num_queries = self.create_dataset()

		self.features_pairs, self.group_pairs, self.queries_pairs = \
					self.convert_labeled_to_paired_data(max_num_pairs=1000000, max_query_bandwidth=10000)
		self.features, self.queries, self.groups, self.labels \
			= torch.tensor(self.features, dtype=torch.float32), \
			  torch.tensor(self.queries), torch.tensor(self.groups), torch.tensor(self.labels)
		self.features_pairs, self.group_pairs, self.queries_pairs = \
			torch.tensor(self.features_pairs,  dtype=torch.float32), \
			torch.tensor(self.group_pairs), torch.tensor(self.queries_pairs)

	def __len__(self):
		return len(self.features_pairs)

	def __getitem__(self, idx):
		return self.features_pairs[idx], self.group_pairs[idx], self.queries_pairs[idx]
	
	def query(self,batch_index):
		return self.features_pairs[self.queries_pairs == batch_index], \
			   self.group_pairs[self.queries_pairs == batch_index]

	def to(self, device):
		if self.data.device.type != device.type:
			self.data = self.data.to(device)
			self.targets = self.targets.to(device)
			self.proxy_groups_tensor = self.proxy_groups_tensor.to(device)
			self.true_groups_tensor = self.true_groups_tensor.to(device)
		return

	def _read_data(self, file_path):
		query_ids = []
		labels = []
		features = []
		n_examples = 0
		with open(file_path) as input_file:
			# The input file can be large. We manually process each line.
			for line in input_file:
				raw_features = line.strip().split(' ')
				labels.append(int(int(raw_features[0]) > 1))
				query_ids.append(int(raw_features[1].split(':')[1]))
				features.append([float(v.split(':')[1]) for v in raw_features[2:]])
				n_examples += 1
				if n_examples % 1000 == 0:
					print('\rFinished {} lines.'.format(str(n_examples)), end='')

			print('\rFinished {} lines.'.format(str(n_examples)), end='\n')
			return (query_ids, features, labels, n_examples)

	def create_dataset(self):
		selected_feature_indices_ = np.array(range(125))
		features_raw = self._read_data(self.path)
		queries_, features_, labels_, _ = features_raw

		# Manipulate the training dataset
		# Grouping decided by the No.133 feature
		# We sample the docs in the query by a set of query level rules
		# to create heterogeneous queries and query level features.
		features_ = np.array(features_)

		# Group score is decided by the 40 percentile of QualityScore2
		group_threshold = np.percentile(features_[:, 132], 40)
		groups_ = np.array(features_[:, 132] > group_threshold, dtype=int)
		labels_ = np.array(labels_)
		queries_ = np.array(queries_)
		
		features = []
		queries = []
		groups = []
		labels = []
		feature_indices = selected_feature_indices_
		dimension = feature_indices.shape[0]

		num_queries = 0
		for query_id in np.unique(queries_):
			query_example_indices = np.where(queries_ == query_id)[0]
			# Same as in the paper, we exclude queries with less than 20 docs
			if query_example_indices.shape[0] < 20:
				continue
			# Sort by PageRank
			query_example_indices = query_example_indices[np.argsort(
					features_[query_example_indices, 129])]
			# For each, we generate two features and decide how to discard
			# negative docs accordingly
			query_features_ = np.random.uniform(low=-1, high=1, size=2)
			discard_probs = np.linspace(
					start=0.5 + 0.5 * query_features_[0],
					stop=0.5 - 0.5 * query_features_[0],
					num=query_example_indices.shape[0]) * (0.7 + 0.3 * query_features_[1])
			query_example_indices = query_example_indices[(
					labels_[query_example_indices] == 1) | (np.random.uniform(
							size=query_example_indices.shape[0]) > discard_probs)]

			# If there is less than 10 posdocs/neg we discard the query
			if (np.sum(labels_[query_example_indices]) <
					10) or (query_example_indices.shape[0] -
									np.sum(labels_[query_example_indices]) < 10):
				continue
			if np.random.uniform() > 0.1 and self.split == 'vali':
				continue
			# Reconstruct the order
			query_example_indices.sort()

			# Only retain queries with minimum number of pos/neg candidates.
			query_groups = groups_[query_example_indices]
			query_labels = labels_[query_example_indices]
			if (np.sum(np.multiply(query_groups == 1, query_labels == 1)) <= 4) or (
					np.sum(np.multiply(query_groups == 0, query_labels == 1)) <= 4) or (
							np.sum(np.multiply(query_groups == 1, query_labels == 0)) <= 4) or (
									np.sum(np.multiply(query_groups == 0, query_labels == 0)) <= 4):
				continue

			groups.extend(groups_[query_example_indices])
			labels.extend(labels_[query_example_indices])

			features.extend(
					add_query_mean(
							features_[query_example_indices][:, feature_indices],
							z=np.reshape(query_groups, (-1, 1))).tolist())

			queries.extend([num_queries] * query_example_indices.shape[0])
			num_queries += 1
			if (self.data_size is not None) and (num_queries >= self.data_size):
				break
		
		features = np.array(features)
		multiplier_dimension = features.shape[1] - dimension - 1
		return features, queries, groups, labels, features.shape[1], multiplier_dimension, num_queries

	def convert_labeled_to_paired_data(self,
																		 index=None,
                                   	 max_num_pairs=200,
                                   	 max_query_bandwidth=200):
		"""Convert data arrays to pandas DataFrame with required column names."""
		if index is not None:
			data_df = pd.DataFrame(self.features[self.queries == index, :])
			data_df = data_df.assign(label=pd.DataFrame(self.labels[self.queries == index]))
			data_df = data_df.assign(group=pd.DataFrame(self.groups[self.queries == index]))
			data_df = data_df.assign(query_id=pd.DataFrame(self.queries[self.queries == index]))
		else:
			data_df = pd.DataFrame(self.features)
			data_df = data_df.assign(label=pd.DataFrame(self.labels))
			data_df = data_df.assign(group=pd.DataFrame(self.groups))
			data_df = data_df.assign(query_id=pd.DataFrame(self.queries))

		def pair_pos_neg_docs_helper(x):
			return pair_pos_neg_docs(
					x, max_num_pairs=max_num_pairs, max_query_bandwidth=max_query_bandwidth)

		# Forms pairs of positive-negative docs for each query in given DataFrame
		# if the DataFrame has a query_id column. Otherise forms pairs from all rows
		# of the DataFrame.
		data_pairs = data_df.groupby('query_id').apply(pair_pos_neg_docs_helper)

		# Create groups ndarray.
		pos_groups = data_pairs['group_pos'].values.reshape(-1, 1)
		neg_groups = data_pairs['group_neg'].values.reshape(-1, 1)
		group_pairs = np.concatenate((pos_groups, neg_groups), axis=1)

		# Create queries ndarray.
		queries = data_pairs['query_id_pos'].values.reshape(-1,)

		# Create features ndarray.
		feature_names = data_df.columns
		feature_names = feature_names.drop(['query_id', 'label'])
		feature_names = feature_names.drop(['group'])

		pos_features = data_pairs[[str(s) + '_pos' for s in feature_names]].values
		pos_features = pos_features.reshape(-1, 1, len(feature_names))

		neg_features = data_pairs[[str(s) + '_neg' for s in feature_names]].values
		neg_features = neg_features.reshape(-1, 1, len(feature_names))

		features_pairs = np.concatenate((pos_features, neg_features), axis=1)

		return features_pairs, group_pairs, queries


def add_query_mean(x, z=None, columns=None, queries=None):
  """Create query level features as the averages of its document features."""
  x = np.array(x)
  nrow = x.shape[0]
  if columns is not None:
    x_ = np.copy(x[:, columns])
  else:
    x_ = np.copy(x)
  if z is not None:
    # for concatenating grouping
    x = np.concatenate((x, z), axis=1)
  if queries is None:
    return np.concatenate((
        x,
        np.tile(np.mean(x_, axis=0), (nrow, 1)),
    ), axis=1)
  else:
    y_ = np.zeros(x_.shape)
    for query in np.unique(queries):
      query_mask = (queries == query)
      y_[query_mask, :] = np.mean(x_[query_mask, :], axis=0)
    return np.concatenate((x, y_), axis=1)

def pair_pos_neg_docs(data, max_num_pairs=10000, max_query_bandwidth=20):
  """Returns pairs of positive-negative docs from given DataFrame."""

  # Include a row number
  data.insert(0, 'tmp_row_id1', list(range(data.shape[0])))

  # Separate pos and neg docs.
  pos_docs = data[data.label == 1]
  if pos_docs.empty:
    return
  neg_docs = data[data.label == 0]
  if neg_docs.empty:
    return

  # Include a merge key.
  pos_docs.insert(0, 'merge_key', 0)
  neg_docs.insert(0, 'merge_key', 0)

  # Merge docs and drop merge key column.
  pairs = pos_docs.merge(
      neg_docs, on='merge_key', how='outer', suffixes=('_pos', '_neg'))
  pairs = pairs[np.abs(pairs['tmp_row_id1_pos'] -
                       pairs['tmp_row_id1_neg']) <= max_query_bandwidth]
  pairs.drop(
      columns=['merge_key', 'tmp_row_id1_pos', 'tmp_row_id1_neg'], inplace=True)
  if pairs.shape[0] > max_num_pairs:
    pairs = pairs.sample(n=max_num_pairs, axis=0, random_state=543210)
  return pairs


if __name__ == "__main__":
	dataset = MSLTR('./data/MSLR-WEB10K/Fold1/', data_size=1000)