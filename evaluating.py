from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

import preprocessing as prep

from sklearn.preprocessing import label_binarize

#ROC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.model_selection import cross_val_predict

import numpy as np

from sklearn.model_selection import StratifiedKFold


def evaluate_all_metrics_on_all_models(data):
	print("dataset: ", data.data_set_name)
	print("threshold: ", data.rare_value_threshold)
	# firstly print accuracy with crossvalidation
	crossvalidation_accuracy_with_model('knn', data)
	crossvalidation_accuracy_with_model('bayes', data)
	crossvalidation_accuracy_with_model('tree', data)

	# secondly print AUC for all models
	auc_for_model('knn', data)
	auc_for_model('bayes', data)
	auc_for_model('tree', data)

	print("======================================")

#model = 'knn' or 'bayes' or 'decision_tree'
def crossvalidation_accuracy_with_model(model, data):

	if model == 'knn':
		#knn in sklearn requires numbers -> gotta do preprocessing
		dataPrep = prep.replaceStringsWithNumbers(data)
		knn = KNeighborsClassifier(n_neighbors = 5)

		score = accuracy_for_model(knn, data)
	elif model == 'bayes':
		gnb = GaussianNB()

		score = accuracy_for_model(gnb, data)
	else: #decision_tree
		clf = DecisionTreeClassifier(random_state=0)

		score = accuracy_for_model(clf, data)

	print (model, ' accuracy: ', score)

	return 0

# data should be of type DataSet
def accuracy_for_model(model, data):
	y = data.y().values
	X = data.X().values

	scores = cross_val_score(model, X, y, cv=10)

	return scores.mean(), scores.std()


def auc_for_model(model, data):
	if model == 'knn':
		#knn in sklearn requires numbers -> gotta do preprocessing
		dataPrep = prep.replaceStringsWithNumbers(data)
		knn = KNeighborsClassifier(n_neighbors = 5)

		score = auc_value_for_model(knn, data)
	elif model == 'bayes':
		dataPrep = prep.replaceStringsWithNumbers(data)
		gnb = GaussianNB()

		score = auc_value_for_model(gnb, dataPrep)
	else: #decision_tree
		dataPrep = prep.replaceStringsWithNumbers(data)
		clf = DecisionTreeClassifier(random_state=0)

		score = auc_value_for_model(clf, dataPrep)

	print (model, ' auc: ', score)

	return 0

def auc_value_for_model(model, data):
	cv = StratifiedKFold(n_splits=10)
	X = data.X().values
	y = data.y().values

	classifier = model

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'red', 'red', 'red'])
	lw = 2


	i = 0
	for (train, test), color in zip(cv.split(X, y), colors):
		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)

		i += 1

	mean_tpr /= cv.get_n_splits(X, y)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)


	return mean_auc

def multiclass_auc_value_for_model(model, data):
	X = data.X().values
	y = data.y().values

	# Binarize the output
	y = label_binarize(y, classes=['B', 'R', 'L'])
	n_classes = y.shape[1]

	# shuffle and split training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
	                                                    random_state=0)

	y_pred = model.fit(X_train, y_train).predict(X_test)

	roc_auc_score(y_pred, y_test)

	print(roc_auc)

def draw_roc_for_model(model, data):
	# Run classifier with cross-validation and plot ROC curves
	cv = StratifiedKFold(n_splits=6)
	# classifier = svm.SVC(kernel='linear', probability=True,
	#                      random_state=random_state)

	X = data.X().values
	y = data.y().values
#	print(X.shape)


	classifier = model

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
	lw = 2

	i = 0
	for (train, test), color in zip(cv.split(X, y), colors):
		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

		i += 1

	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
	         label='Luck')

	mean_tpr /= cv.get_n_splits(X, y)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	print("auc: ", mean_auc)
	plt.plot(mean_fpr, mean_tpr, color='g', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
