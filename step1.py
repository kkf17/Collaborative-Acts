"""------------------------------------------------------------
	step1.py -f filename -n 0 -t [start_time] [end_time] 
	OPT:
		-f : filename
		-n : number of most predictive words
		-t : start_time, end_time: time laps to study the 
			interaction between the 2 participants
--------------------------------------------------------------"""

import sys

import pandas as pd
import numpy as np

import CollaborativeActsDyad

args = sys.argv
n_args = 1

filename=''
n=0
t=[]

if '-f' in args:
	filename = args[args.index('-f')+1]
	n_args+=2
if '-n' in args:
	n = int(args[args.index('-n')+1])
	n_args+=2
if '-t' in args:
	t.append(int(args[args.index('-t')+1]))
	t.append(int(args[args.index('-t')+2]))
	n_args+=3

print('-f: ',filename)
print('-n: ',n)
print('-t: ',t)

collabacts=CollaborativeActsDyad.CollabActsDyad()
collabacts.init_model(filename)
collabacts.visualize(n, t)



