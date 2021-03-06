"""---------------------------------------------------------------------------
	step4.py -f folder
	OPT:
		-f : folder (./objects/) or set of folders
-------------------------------------------------------------------------------"""
import pickle
import sys
sys.path.insert(0, './toolsML')

import random

import pandas as pd
import numpy as np

import model
import embedding

args = sys.argv


n_args = 1
if '-f' in args:
	rep = args[args.index('-f')+1]
	n_args+=2
	

n_test= 17
algo='lrcn'

	
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
	

regular = True	
if regular:
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
		X.append(utterances[i])
		y.append(labels[i])
		
	utterances = X
	labels = y
	


folds = []
kappas =[]
accs = []
SIZE = len(utterances)
for n_fold in range(10):#range(SIZE-n_test):

	#fold_test = np.asarray(range(n_test),dtype=np.int32)+n_fold	
	fold_test = np.asarray(random.sample(range(0,SIZE), n_test),dtype=np.int32)	# random index
		
	fold_train = np.setdiff1d(np.array(range(SIZE)), fold_test)
	print(fold_test, fold_train)

	mod = embedding.Embedding()
	#mod.init(algo)
	mod.build()
		
	for fold in fold_train:
		
		X_train=utterances[fold]
		y_train=labels[fold]
			
		print(X_train.shape, y_train.shape)
		mod.train(X_train, y_train)

	for fold in fold_test:
		
		X_test=utterances[fold]
		y_test=labels[fold]
			
		y_pred = mod.test(X_test, y_test)
		#acc, kappa = mod.metrics(y_test, y_pred)
			
		#print(fold, ': ', acc, kappa)
			
		folds.append(fold)
		#accs.append(acc)
		#kappas.append(kappa)
	
	













