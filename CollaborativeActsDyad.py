"""------------------------------------------------------------
	Class to store data for a given dyad
--------------------------------------------------------------"""

import pickle
import sys
sys.path.insert(0, './toolsNLP')

import pandas as pd
import numpy as np

#from tools import *

from visualize import visualize
from preprocessing import preprocessing, most_predictive_f_class #,most_predictive_chi2
from vectorization import vector


class CollabActsDyad(object):
	def __init__(self):
		pass

	def init_model(self, dataFile, dyad=0):
		df = pd.read_csv(dataFile,delimiter="\t",header=None,error_bad_lines=False, encoding="utf8")

		self.dyads=np.delete(np.array(df[0]),0)[0]
		
		#self.dyads_index = np.delete(np.array(df[0]),0)

		self.utterances = np.delete(np.array(df[7]),0)
		#subcategories = np.delete(np.array(df[8]),0)
		self.categories = np.delete(np.array(df[9]),0)

		self.collab_acts = ['Information','Interaction management','Other','Outside activity','Social relation','Task management','Tool','Transactivity']#np.unique(np.delete(np.array(df[9]),0)) chaque dyad

		self.labels = np.zeros(len(self.categories))
		for k in range(len(self.collab_acts)):
			index = np.where(self.categories == self.collab_acts[k])
			self.labels[index]=k

		self.participants=np.delete(np.array(df[1]),0)

		self.start_time=np.delete(np.array(df[4]),0).astype(np.float)
		self.duration=np.delete(np.array(df[6]),0).astype(np.float)

	def visualize(self, n=10, t=[]):
		visualize(self.dyads, self.utterances, self.categories, self.labels, self.collab_acts, self.participants, self.start_time, self.duration, n, t)

	def preprocessing(self, ptype='', g=[], n=10):
		print(self.utterances.shape, self.labels.shape)
		if 'u' in ptype:
			index = np.where(np.unique(self.utterances))
			self.utterances = self.utterances[index]
			self.labels = self.labels[index]
			print(self.utterances.shape, self.labels.shape)

		self.most_pred_words = most_predictive_f_class(self.utterances, self.labels, n)
				       #most_predictive_chi2(self.utterances, self.labels, n)
		if n !=0:
			w=self.most_pred_words
		else:
			w=[]
		token=[]
		for k in range(self.utterances.shape[0]):
				token.append(preprocessing(self.utterances[k], ptype, g, w))
		self.tokens=token

	def clean(self):
		if [] in self.tokens:
			token = []
			label = []
			for i in range(len(self.tokens)):
				#if self.tokens[i] == []:  # cas remplacer [] par [' '] -> meilleur
					#self.tokens[i] = [' ']

				if self.tokens[i] != []:  # cas retirer []
					token.append(self.tokens[i])
					label.append(self.labels[i])

			self.tokens = token
			self.labels = np.array(label)

	def vectorization(self, vtype='ww'):
		#clean [] et '' data : labels , tokens
		self.clean()
		self.vocab, self.vectors = vector(self.tokens, self.utterances, vtype)
		
		





