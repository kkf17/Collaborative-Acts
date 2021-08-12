"""------------------------------------------------------------
	step2.py -f filename -p 'npsl' -g [GRAM] -m 0 -v 'bow/ww' 
	OPT:
		-f : filename
		-p : type of preprocessing
			n: normalisation
			p: ponctuation removing
			s: stop words removing
			l: lemmatisation
		-g : type of grammar correction 
			(see preprocessing.py)
		-m : most predictive words
			number of most predictive words
		-v : type of vectorization
			bow : bag of words
			bowt : bag of words with TFIDF ponderation
			ww : word vector
			wm : mean of word vectors
			wwt : word vectors with TFIDF ponderation
--------------------------------------------------------------"""

import pickle
import sys
sys.path.insert(0, './tools')

import numpy as np

import CollaborativeActsDyad

from spellnCorrection import spellncorrect

args = sys.argv
n_args = 1

print(args)

filename=''
ptype=''
g=[]
m=0
vtype='ww'


if '-f' in args:
	filename = args[args.index('-f')+1]
	n_args+=2
if '-p' in args:
	n_args+=1
	if not '-' in args[args.index('-p')+1]:
		ptype = args[args.index('-p')+1]
		n_args+=1
if '-m' in args:
	n_args+=1
	if not '-' in args[args.index('-m')+1]:
		m = int(args[args.index('-m')+1])
		n_args+=1
if '-v' in args:
	n_args+=1 
	if not '-' in args[args.index('-v')+1]:
		vtype=args[args.index('-v')+1]
		n_args+=1

if '-g' in args:
	n_args+=1
	index=args.index('-g')+1
	g=[]
	for k in range(len(args)-n_args):
		if not '-' in args[index+k]: 
			g.append(args[index+k])

print('-f: ',filename)
print('-p: ',ptype)
print('-g: ',g)
print('-m: ',m)
print('-v: ',vtype)


collabacts=CollaborativeActsDyad.CollabActsDyad()
collabacts.init_model(filename)
collabacts.preprocessing(ptype, g, n=m)
collabacts.vectorization(vtype)


filename = './objects/'+(filename.split('.'))[1].split('/')[2]+'_'+ptype+''+"".join(g)+'_n'+str(m)+'_'+vtype
print('Writting on file '+filename+'. \n')	
with open(filename, "wb") as fp:  
	pickle.dump(collabacts, fp)







