import pandas
from sklearn import preprocessing

# replaces labels with 0,1,2...
def replaceStringsWithNumbers(data):
	# for colName in data.df:
	# 	# print(colName)
	# 	le = preprocessing.LabelEncoder()
	# 	le.fit(data.df[colName])
	# 	data.df[colName] = le.transform(data.df[colName])
	data.df = data.df.apply(preprocessing.LabelEncoder().fit_transform)
	return data

def apply_binning_on_continuous_values(data, columns):
	for column in columns:
		print(column)
		data.df[column] = pandas.cut(data.df[column], 5, labels=['one', 'two', 'three', 'four', 'five'])

def remove_useless_columns(data, columns):
	for column in columns:
		data.df.drop([column], axis = 1, inplace = True, errors = 'ignore')
