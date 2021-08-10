"""------------------------------------------------------------
	step0.py -f filename -d dyad -c
	OPT:
		-f : filename
		-d : given dyad
			number from 0 to number of dyads.
		-c : spell correction
--------------------------------------------------------------"""
import string
import csv
import sys
sys.path.insert(0, './toolsNLP')

import numpy as np
import pandas as pd

from spellnCorrection import spellncorrect

"""
NLP: class in .csv file ('./collaborativeActs.csv')
	0. Dyads
	1. Participant
	2. Id
	3. EAT	
	4. StartTime 
	5. EndTime
	6. Duration
	7. Utterances
	8. Subcategories	
	9. Categories
"""

args = sys.argv
n_args = 1

filename=''
d=0
c=False

if '-f' in args:
	filename = args[args.index('-f')+1]
if '-d' in args:
	d = args[args.index('-d')+1]
if '-c' in args:
	c=True


datafile = './data/dyad_'+d+'.csv'

print('-f: ',filename)
print('-d: ',d)
print('-c: ', c)


df = pd.read_csv(filename,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

X = np.array(df)
header= X[0]

dyads=np.delete(np.array(df[0]),0)
unique_dyads = np.unique(dyads)
dyads_index=[np.where(dyads == dyad) for dyad in unique_dyads]

X = np.delete(X, (0), axis=0)
X = X[dyads_index[int(d)]]


if c == False:
	print('Writting on file '+datafile+'. \n')
	with open(datafile, mode='w') as collab_file:
		writer = csv.writer(collab_file, delimiter="\t")#, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(header)
		for i in range(X.shape[0]):
			writer.writerow(X[i])

if c == True:
	spellncorrect(datafile)




