import numpy as np
import pandas as pd 

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.models import load_model, save_model

class LSTMSteps(object):
	def __init__(self):
		pass
		
	def split_sequences(self, sequences, n_steps):
		X, y = list(), list()
		for i in range(len(sequences)):
			# find the end of this pattern
			end_ix = i + n_steps
			# check if we are beyond the dataset
			if end_ix > len(sequences):
				break
			# gather input and output parts of the pattern
			seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
			#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[i:end_ix, -1]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)
		
	def build(self, features):
	
		self.n_features = 96#X.shape[1]
		self.n_steps = 2

		hidden_nodes = 20#95 
		model = Sequential()
		model.add(LSTM(hidden_nodes,return_sequences=True, input_shape=(self.n_steps, self.n_features), name='Layer1')) #return_sequences=True
		model.add(Dropout(0.25))
		model.add(LSTM(hidden_nodes, name='Layer2'))
		model.add(Dropout(0.25))

		model.add(Dense(8, activation='softmax', name='Dense1'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		model.save('./lstmsteps_model')


	def train(self, X_train, y_train, param=None): 

		Z_train = np.insert(X_train,  X_train[0].shape[0] ,y_train, axis=1) 
	
		X_train, y_train = self.split_sequences(Z_train, self.n_steps)
		y_train = to_categorical(y_train)

		model = load_model('./lstmsteps_model')
		
		#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=60, batch_size=10, verbose=0 ,shuffle=True)#, class_weight=weights)
		
		model.save('./lstmsteps_model')
		

	def test(self, X_test, y_test):
	
		Z_test = np.insert(X_test,  X_test[0].shape[0] ,y_test, axis=1)
		X_test, y_test = self.split_sequences(Z_test, self.n_steps)
		y_test_categ = to_categorical(y_test)
		
		model = load_model('./lstmsteps_model')
		y_pred = model.predict_classes(X_test)
		
		return y_pred







