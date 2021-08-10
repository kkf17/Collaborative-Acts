import numpy as np

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import spacy
from spacy import displacy
spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")


def vector(tokens, utterances, vtype='ww'):
	if vtype=='bow' or vtype=='bowt':
		vocab, vectors = bowvectorization(tokens, utterances, vtype)
	if vtype=='ww' or vtype=='wm' or vtype=='wwt':
		vocab, vectors = wwvectorization(tokens, utterances, vtype)
	return vocab, vectors


"""------- *** POS Tagging *** ----------"""

"""------- *** Bag of Words *** ----------"""
# PROBLEME : uniformiser size of vectors for classification

def bowvectorization(token,  utterances, vtype='ww'):
	tokens = []
	for k in range(len(token)):
		tokens.append(" ".join(token[k]))

	if  vtype=='bow':
		vocab, vectors = bagOfWords(tokens, n_gram=1)
	if vtype=='bowt':
		vocab, vectors = bagOfWordsTFIDF(tokens, smooth=True)
	return vocab, vectors

def bagOfWords(utterances, n_gram=1):
	print('bow')
	vectorizer = CountVectorizer(ngram_range=(1, n_gram)) 
	X = vectorizer.fit_transform(utterances)
	analyze = vectorizer.build_analyzer()
	vocab = vectorizer.get_feature_names()
	return vocab, X.toarray()

def bagOfWordsTFIDF(utterances, smooth=True):
	vectorizer = TfidfVectorizer(smooth_idf=smooth)
	X = vectorizer.fit_transform(utterances)
	analyze = vectorizer.build_analyzer()
	vocab = vectorizer.get_feature_names()
	return vocab, X.toarray() 


"""------- *** Word vectors *** ----------"""
def wwvectorization(tokens, utterances, vtype='ww'):
		vector = []
		vocab, tfidf = bagOfWordsTFIDF(utterances, smooth=True) 
		for k in range(len(tokens)):
			vector.append(wordvector(tokens[k],vtype,tfidf[k], vocab))
		vectors = np.asarray(vector,dtype=np.float64)
		return vocab, vectors


def wordvector(tokens, vtype='ww', tfidf=[], vocab=[]):
	vector = []
	if vtype=='ww':
		vector = wordVector(tokens)
	if vtype=='wm':
		vector = wordVectorMEAN(tokens)
	if vtype=='wwt': # tfidf
		vector = wordVectorTFIDF(tokens, tfidf, vocab)
	return vector

def wordVector(tokens):
	return nlp(" ".join(tokens)).vector


def wordVectorMEAN(tokens): 
	return 1/len(tokens)*sum([nlp(token).vector for token in tokens if token != ''])


def wordVectorTFIDF(tokens, tfidf, vocab): 
		tokens = [token for token in tokens if token in vocab]

		vect = [0]*96 # !! MODIF: cas tokens = []
		if tokens != []:
			vect = 1/len(tokens)*sum([(nlp(token)).vector*tfidf[vocab.index(token)] for token in tokens if token != ''])

		return vect

		




