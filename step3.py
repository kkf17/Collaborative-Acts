"""---------------------------------------------------------------------------
	step3.py -r folder -t 0 -a algo
	OPT:
		-r : folder (./objects/) or set of folders
		-t : number of folds for test (n) or set of numbers of folds (min max)
		-a : chosen algorithm for classification or set of algorithms to test
				(lrn, lstm, lstmsteps)
-------------------------------------------------------------------------------"""
import pickle
import sys
sys.path.insert(0, './toolsML')

import pandas as pd
import numpy as np

from results import *

args = sys.argv


ntests = [0]
stype = 'default'
algos = ['nb']

nargs = 1
if '-r' in args:
	nargs += 1
	reps=[] 
	i=1
	while not '-' in args[args.index('-r')+i]:
		reps.append(args[args.index('-r')+i])
		nargs += 1
		i+=1
		if  nargs >= len(args):
			break
if '-t' in args:
	nargs += 1
	stype='folds'
	ntests=[] 
	i=1
	while not '-' in args[args.index('-t')+i]:
		ntests.append(int(args[args.index('-t')+i]))
		nargs += 1
		i+=1
		if  nargs >= len(args):
			break
	if len(ntests) != 1:
		ntests = range(ntests[1]+1)[ntests[0]:ntests[1]+1]

if '-a' in args:
	nargs += 1
	algos=[] 
	i=1
	while not '-' in args[args.index('-a')+i]:
		algos.append(args[args.index('-a')+i])
		nargs += 1
		i+=1
		if  nargs >= len(args):
			break

print('Folders: ',reps)
print('Number of test folds: ', np.array(ntests))
print('Algorithms: ', algos,'\n')


#rep=[]
#ntests=[2,3,4,10,11]
#algo=[]
params = [5,12,20] 

accs=[]
kappas=[]
i = 0
for rep in reps:
	for ntest in ntests: 
		for algo in algos:
			for param in params:
				print(i, ') ', rep, ntest, algo, param)
				acc, kappa = case_test(rep, ntest, algo, param)
				accs.append(acc)
				kappas.append(kappa)
				i+=1

for i in range(len(accs)):
	print_graph(accs[i], mtype='Accuracy')
	print_graph(kappas[i], mtype='Kappa score')
	
	
print('\n')
SIZE = len(accs)
for i in range(SIZE):
	for j in range(SIZE)[i+1:]:
		p, n, e, abetter = comparison(i, j, accs)
		p, n, e ,kbetter = comparison(i, j, kappas)
		print('Comparison between ', i ,' - ', j, 'Better: ', kbetter, abetter)
	print('\n')










