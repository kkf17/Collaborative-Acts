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
#from keras.layers import TimeDistributed
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model, save_model


class Embedding(object):
	def __init__(self):
		pass

	def build(self):
		# integer encode the documents
		self.vocab_size = 200
		# pad documents to a max length of 4 words
		self.max_length = 20  # regular size of documents, a standard

		model = Sequential()
		model.add(Embedding(input_dim=200, output_dim=64, input_length=self.max_length))
		#model.add(Embedding(self.vocab_size))#, 64, self.max_length))
		#model.add(Flatten())
		#model.add(Dense(1, activation='sigmoid'))

		model.add(Bidirectional(LSTM(64)))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(1))

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		
		model.save('./embedding')

	def train(self, X_train, y_train):
		# X_train = ['', '',''] == utterances
		X_train= [one_hot(d, self.vocab_size) for d in X_train]
		X_train = pad_sequences(X_train, maxlen=self.max_length, padding='post')
		
		model.load('./embedding')

		model.fit(padded_docs, y_train, epochs=50, verbose=0)
		model.save('./embedding')

	def test(self, X_test, y_test):
		X_test = [one_hot(d, self.vocab_size) for d in X_test]
		X_test = pad_sequences(X_test, maxlen=self.max_length, padding='post')
		
		model.load('./embedding')

		loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
		print('Accuracy: %f' % (accuracy*100))
