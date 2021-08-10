import random

import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split

import model


def defaultclass(vectors, labels, algo='naivebayes', param=None):
	folds = []
	kappas =[]
	accs = []
	for n_fold in range(len(vectors)):
		v, l = concatenate(vectors, labels, range(len(vectors)))
		X_train, X_test, y_train, y_test = train_test_split(v, l, test_size=0.25, random_state=42)
	
		mod = model.Model()
		mod.init(algo)
		mod.build(vectors[0][0].shape[0])
		mod.train(X_train, y_train)
		y_pred = mod.test(X_test, y_test)
		acc, kappa = mod.metrics(y_test, y_pred)
		
		print(acc, kappa)
		
		accs.append(acc)
		kappas.append(kappa)

	return accs, kappas, folds


def foldclass(vectors, labels, n_test=1, algo='lstm', param=None):
	folds = []
	kappas =[]
	accs = []
	SIZE = len(vectors)
	for n_fold in range(10):#range(SIZE-n_test):

		#fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
		fold_test = np.asarray(random.sample(range(0,SIZE), n_test),dtype=np.int32)	# random index
		
		fold_train = np.setdiff1d(np.array(range(SIZE)), fold_test)
		print(fold_test, fold_train)

		mod = model.Model()
		mod.init(algo)
		mod.build(vectors[0][0].shape[0])
		
		for fold in fold_train:
		
			X_train=vectors[fold]
			y_train=labels[fold]
			
			mod.train(X_train, y_train, param)

		for fold in fold_test:
		
			X_test=vectors[fold]
			y_test=labels[fold]
			
			y_pred = mod.test(X_test, y_test)
			acc, kappa = mod.metrics(y_test, y_pred)
			
			print(fold, ': ', acc, kappa)
			
			folds.append(fold)
			accs.append(acc)
			kappas.append(kappa)

	if 1 < n_test :
		folds = np.asarray(folds)
		acc = np.array(accs)
		kappa = np.array(kappas)
		accs = []
		kappas = []
		for i in range(len(vectors)):
			if i in folds:
				index= np.where(folds==i)[0]
				accs.append(np.mean(acc[index]))
				kappas.append(np.mean(kappa[index]))		
	return accs, kappas, folds

"""
def concatenate(X, y, index):
	vectors=[]
	labels=[]
	for i in index:
		vectors.append(X[i])
		labels.append(y[i])

	X=np.concatenate(vectors, axis=0)
	y=np.concatenate(labels, axis=0)
	return X,y
"""















