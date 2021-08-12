import pickle
import sys

import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model, save_model

import tensorflow as tf

class Embedding(object):
	def __init__(self):
		pass

	def build(self):
		# integer encode the documents
		self.vocab_size = 200
		# pad documents to a max length of 4 words
		self.max_length = 20 # regular size of documents, a standard

		SIZE = 64

		self.model = Sequential()
		self.model.add(tf.keras.layers.Embedding(self.vocab_size, SIZE, input_length=self.max_length))
		#self.model.add(Flatten())
		#self.model.add(Dense(1, activation='sigmoid'))

		self.model.add(Bidirectional(LSTM(SIZE)))
		self.model.add(Dense(SIZE, activation='relu'))
		self.model.add(Dense(1))


		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		
		#model.save('./embedding')

	def train(self, X_train, y_train):
		# X_train = ['', '',''] == utterances
		X_train= [one_hot(d, self.vocab_size) for d in X_train]
		X_train = pad_sequences(X_train, maxlen=self.max_length, padding='post')
		
		#model.load('./embedding')

		self.model.fit(X_train, y_train, epochs=60, verbose=0)
		self.model.save('./embedding')

	def test(self, X_test, y_test):
		X_test = [one_hot(d, self.vocab_size) for d in X_test]
		X_test = pad_sequences(X_test, maxlen=self.max_length, padding='post')
		
		model = load_model('./embedding')

		loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
		print('Accuracy: %f' % (accuracy*100))
