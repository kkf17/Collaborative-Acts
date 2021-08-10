import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import lrcn, lstm, lstm_steps

class Model(object):
	def __init__(self):
		pass
		
	def init(self,algo = 'lstm'):
		if algo == 'lstm':
			self.model = lstm.LSTm() 
		
		if algo == 'lstmsteps':
			self.model = lstm_steps.LSTMSteps() 
		
		if algo == 'lrcn':
			self.model = lrcn.LRCN()
			
	def build(self, features):
		(self.model).build(features)
		
	def train(self, X_train, y_train, param=None):
		#param = self.weights(y_train)
		(self.model).train(X_train, y_train, param)
	
	def test(self, X_test, y_test):
		y_pred = (self.model).test(X_test, y_test)
		return y_pred
		
		
	def metrics(self, y_test, y_pred):
		accuracy = accuracy_score(y_test,y_pred)
		n_true = sum(np.where(y_test==y_pred, 1, 0))

		kappa_score = cohen_kappa_score(y_test, y_pred)
		cmtx = confusion_matrix(y_test, y_pred)
		#class_report_1 = classification_report(y_test, y_pred)#, target_names=target)
		
		return accuracy, kappa_score

	
	def weights(self, y):
		return compute_class_weight('balanced', np.unique(y), y)
		
	

	########### SKLEARN ######################
	"""
			model = GaussianNB() 

			if algo == 'nb':
				model = GaussianNB()

			if algo == 'knn':
				#n_neighbors = sqrt(abs(X_train.shape[0])/3)
				model = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='auto', metric='minkowski', metric_params=None)
				#weights{‘uniform’, ‘distance’} ; algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} ; metric{'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'mahalanobis'}	

			if algo == 'svm':
				#model= make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', degree=param, gamma='scale', class_weight=None))
				model = SVC(C=1.0, kernel='rbf', degree=5, gamma='scale', class_weight='balanced')
				#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} ; class_weight: None, dict or ‘balanced’

			if algo == 'nn':
				model = MLPClassifier(hidden_layer_sizes=param, activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', max_iter=200, shuffle=True, tol=0.0001, verbose=False)
				#hidden_layer_sizes=(60, 60) ; activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’} ; solver{‘lbfgs’, ‘sgd’, ‘adam’} ;
			y_pred = model.fit(X_train, y_train).predict(X_test)		

	"""
