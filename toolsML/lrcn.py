import numpy as np
import pandas as pd 

import tensorflow as tf
from tensorflow import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model, save_model


class LRCN(object):
	def __init__(self):
		pass
		
	def build(self, features):

		self.n_steps = 1
		self.n_features = features #X_train.shape[1]
		self.n_seq = 1
		
		hidden_nodes = 500#95 
		model = Sequential()
		
		model.add(TimeDistributed(Conv1D(filters=8, kernel_size=1, activation='relu'), input_shape=(None, self.n_steps, self.n_features)))
		model.add(TimeDistributed(MaxPooling1D(1)))
		model.add(TimeDistributed(Flatten()))
		
		hidden_nodes = 20 #96
		model.add(LSTM(hidden_nodes))
		model.add(Dense(8, activation='softmax'))
		
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		model.save('./lrcn_model')
		
	def train(self, X_train, y_train, param=None):
			
		X_train = X_train.reshape((X_train.shape[0], self.n_seq, self.n_steps, self.n_features))
		y_train = to_categorical(y_train)
		
		model = load_model('./lrcn_model')		
		
		#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=40, batch_size=200, verbose=0,shuffle=True)#, class_weight=param)
		
		model.save('./lrcn_model')


	def test(self, X_test, y_test):
		
		X_test = X_test.reshape((X_test.shape[0], self.n_seq, self.n_steps, self.n_features))	
		y_test_categ = to_categorical(y_test)
		
		model = load_model('./lrcn_model')

		y_pred = model.predict_classes(X_test)
		
		return y_pred
	
	
	
