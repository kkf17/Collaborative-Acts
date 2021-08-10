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


class LSTm(object):
	def __init__(self):
		pass
		
	def build(self, features):
	
		self.n_steps = 1
		self.n_features = features #X_train.shape[1]
		
		hidden_nodes = 20#95 
		model = Sequential()
		#model.add(LSTM(hidden_nodes, return_sequences=True,input_shape=(self.n_steps, self.n_features))) #return_sequences=True
		#model.add(Dropout(0.25))
		model.add(LSTM(hidden_nodes, input_shape=(self.n_steps, self.n_features)))
		#model.add(Dropout(0.25))

		model.add(Dense(8, activation='softmax'))
		
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
				
		model.save('./lstm_model')
		
	def train(self, X_train, y_train, param=None, weights=None):  
		
		X_train = np.reshape(X_train, (X_train.shape[0], self.n_steps, X_train.shape[1]))
		y_train = to_categorical(y_train)
		
		model = load_model('./lstm_model')
	
		#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=60, batch_size=10, verbose=0,shuffle=True)#, class_weight=weights) 
		
		model.save('./lstm_model')


	def test(self, X_test, y_test):
	
		X_test = np.reshape(X_test, (X_test.shape[0], self.n_steps, X_test.shape[1]))	
		y_test_categ = to_categorical(y_test)
		
		model = load_model('./lstm_model')
		
		y_pred = model.predict_classes(X_test)
			
		return y_pred

