import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif, chi2


def visualize(dyads, utterances, categories, labels, collab_acts, participants, start_time, duration, n, t):
	print('\nDyad: ',dyads,' Utterances: ',utterances.shape) 
	print('Categories: ', collab_acts)
	print_time((duration).astype(np.float))

	histogram(categories, dyads)
	dialog(start_time, participants, dyads, t[0], t[1], 'participants')
	dialog(start_time, categories, dyads, t[0], t[1], 'dialog')
	timestudy(duration, categories, collab_acts, dyads)

	print('With f_class:')
	print(most_predictive_f_class(utterances, labels, n))
	print('With chi2:')
	print(most_predictive_chi2(utterances, labels, n))


def print_time(time):
	minut, sec = divmod(int(sum(time)),60)
	millsec =  sum(time)-int(sum(time))
	print('Time:', sum(time), minut,':',sec,':',millsec)

def histogram(x, index):
	y, x, _ = plt.hist(x, density=True, color = 'magenta',
            edgecolor = 'black')# bins = 8, edgecolor='black')
	plt.xlabel('Collaborative Acts')
	plt.title('Repartition of collaborative Acts for dyad '+index)
	plt.show()
	#plt.savefig(index+'_0')
	return y,x

def dialog(X,y, index, start=0, end=20, type_graph='dialog'):
	X = X[np.arange(start, end)]
	y = y[np.arange(start, end)]	
	
	plt.plot(X,y,'m*')
	for i in range(X.shape[0]): 
		plt.axvline(x=X[i], linewidth=0.25, color='c', linestyle='-.')

	if type_graph=='dialog':
		plt.xlabel('Time (s)')
		plt.ylabel('Collaborativ Acts')
		plt.title('Types of utterances for dyad: '+index+' ( from '+ str(start)+' to '+str(end)+')')

	if type_graph=='participants':
		plt.xlabel('Time (s)')
		plt.ylabel('Participant')
		plt.title('Participation for dyad: '+index+' ( from '+ str(start)+' to '+str(end)+')')
	plt.show()
	#plt.savefig()

def timestudy(x, y , labels, index):
	labels_size = len(labels) #labels.shape[0]
	mean_time=np.zeros(labels_size)
	min_time=np.zeros(labels_size)
	max_time=np.zeros(labels_size)
	time_by_class=np.zeros(labels_size)
	for label in range(labels_size):
		ind=np.where(y == labels[label])
		time= x[ind]  
		if time != []:
			time_by_class[label] = sum(time)*100 / sum(x)
			print(labels[label],':')
			print('Time min (%): ', min(time))
			print('Time max (%): ', max(time))
			print('Time mean (%): ', np.mean(time))
			mean_time[label]=np.mean(time/ sum(x))*100 #/ sum(x)
			min_time[label]=min(time) * 100/ sum(x)
			max_time[label]=max(time)* 100/ sum(x)


	plt.plot(labels,mean_time,'m*', labels,min_time,'y*', labels,max_time,'g*')
	for i in range(labels_size): 
		plt.axvline(x=labels[i],linewidth=0.5, color='c', linestyle='-.')
	plt.xlabel('Collaborative Acts')
	plt.ylabel('Mean time of utterances (%)')
	plt.title('Mean time of utterances according to their category for dyad: '+index)
	plt.show()
	#plt.savefig(index+'_1')

	plt.plot(labels,time_by_class,'m*')
	#print(time_by_class)
	for i in range(labels_size): 
		plt.axvline(x=labels[i],linewidth=0.5, color='c', linestyle='-.')
	plt.xlabel('Collaborative Acts')
	plt.ylabel('Total time of utterances (%)')
	plt.title('Total time of utterances according to their category for dyad: '+index)
	plt.show()
	#plt.savefig(index+'_2')

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







