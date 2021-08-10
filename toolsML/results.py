import pickle
import sys

import pandas as pd
import numpy as np

from classification import foldclass, defaultclass

import matplotlib.pyplot as plt


def comparison( i, j, measure):
	a = np.array(measure[i])-np.array(measure[j])  
	p = sum(np.where(a > 0, 1, 0))
	n = sum(np.where(a < 0, 1, 0))
	e = sum(np.where(a == 0, 1, 0))
	better = -100
	if p > n:
		better = i
	if p < n:
		better = j 
	return p, n, e ,better


def case_test(rep, ntest, algo, param=None):
	stype = 'default'
	if ntest != 0:
		stype = 'folds'

	txt= rep+'data.txt'
	with open(txt) as f:
		files = f.readlines()

	DYADS = []
	for i in range(len(files)):
		with open(rep+files[i][:-1], "rb") as fp: 
			collabacts = pickle.load(fp)
		DYADS.append(collabacts)

	utterances=[]
	vectors=[]
	labels=[]
	for i in range(len(DYADS)):
		utterances.append(DYADS[i].utterances)
		vectors.append(DYADS[i].vectors)
		labels.append(DYADS[i].labels)

	accs = []
	kappas = []
	if stype == 'default':
		accs, kappas, folds = defaultclass(vectors, labels, algo, param)

	if stype == 'folds':
		accs, kappas, folds = foldclass(vectors, labels, ntest, algo, param)
		
	if stype == 'rfolds': #regfolds 
		utterances=np.concatenate(utterances, axis=0)	
		vectors=np.concatenate(vectors, axis=0)
		labels=np.concatenate(labels, axis=0)
	
		index = np.where(np.unique(utterances))
		utterances = utterances[index]
		vectors = vectors[index]
		labels = labels[index]

		index=np.array_split(range(vectors.shape[0]),24) #18

		X=[]
		y=[]
		for i in index:
			X.append(vectors[i])
			y.append(labels[i])
			
		accs, kappas, folds = foldclass(X, y, ntest, algo, param)
	
	return accs, kappas


def print_graph(hist, mtype='accuracy'):
	print('\nMean '+mtype+': ', np.mean(hist))
	print('Min '+mtype+': ', min(hist))
	print('Max '+mtype+': ', max(hist))

	plt.plot(hist,'m*', label=mtype)
	for i in range(len(hist)): 
		plt.axvline(x=i,linewidth=0.5, color='c', linestyle='-.')

	plt.legend(loc='upper left')
	plt.xlabel('Fold')
	plt.ylabel(mtype)
	plt.title(mtype+ ' computed for each Test according to the different folds')
	plt.show()
	
	#plt.save(mtype+str(np.mean(hist)))
