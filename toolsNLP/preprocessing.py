import string 
import numpy as np

import spacy
from spacy import displacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif, chi2

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")


def preprocessing(utterance, ptype='', g=[], w=[]):
	if 'n' in ptype: 
		utterance = normalization(utterance)
	tokens = tokenization(utterance)
	if 'p' in ptype:
		tokens = ponctuation(tokens)
	if 's' in ptype:
		tokens = delete_stop_words(tokens)
	if w != []:
		tokens = most_predictive_tokens(tokens, w)
	if g != []:
		tokens = grammar(tokens, g)
	if 'l' in ptype:
		tokens = lemmatization(tokens)
	if not 'l' in ptype:
		tokens=[token.text for token in tokens]
	return tokens


"""------- *** Preprocessing *** ----------"""

def normalization(text):
	return text.lower()

def tokenization(text):
	doc = nlp(text)
	return [token for token in doc]

def ponctuation(tokens):
	return [token for token in tokens if (not token.text in string.punctuation or token.text=='?' or token.text=='!') ]

def delete_stop_words(tokens):
	return [token for token in tokens if not token.is_stop]

def most_predictive_tokens(tokens, words):
	return [token for token in tokens if token.text in words]

def lemmatization(tokens):
	return [token.lemma_ for token in tokens]

def grammar(token, grammar=[]):
	tokens = token
	if 'ADJ' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'ADJ']
	if 'ADP' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'ADP']
	if 'ADV' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'ADV']
	if 'AUX' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'AUX']
	if 'CONJ' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'CONJ']
	if 'DET' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'DET']
	if 'INTJ' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'INTJ']
	if 'NOUN' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'NOUN']
	if 'NUM' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'NUM']
	if 'PART' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'PART']
	if 'PRON' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'PRON']
	if 'PROPN' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'PROPN']
	if 'PUNCT' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'PUNCT']
	if 'SCONJ' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'SCONJ']
	if 'SYM' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'SYM']
	if 'VERB' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'VERB']
	if 'X' in grammar:
		tokens=[token for token in tokens if token.pos_ != 'X']
	return tokens 


"""------- *** Most predictive words *** ----------"""
def most_predictive_f_class(x, y, k):
	vectorizer =CountVectorizer()
	X = vectorizer.fit_transform(x)
	labels = np.array(vectorizer.get_feature_names())

	f_class, p_val= f_classif(X, y)
	n_values=np.array(f_class.argsort()[-k:][::-1])
	return labels[n_values]

def most_predictive_chi2(x, y, k):
	vectorizer =CountVectorizer()
	X = vectorizer.fit_transform(x)
	labels = np.array(vectorizer.get_feature_names())

	chi_sq, p_val = chi2(X, y)
	n_values=np.array(chi_sq.argsort()[-k:][::-1])
	return labels[n_values]







