import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from subprocess import check_output
import pickle
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



df = pd.read_csv('input/movie_metadata.csv')
first_actors = set(df.actor_1_name.unique())
second_actors = set(df.actor_2_name.unique())
third_actors = set(df.actor_3_name.unique())

df.color = df.color.map({'Color': 1, 'Black and White': 0})

unique_genre_labels = set()
for genre_flags in df.genres.str.split('|').values:
	unique_genre_labels = unique_genre_labels.union(set(genre_flags))

for label in unique_genre_labels:
	df['Genre='+label] = df.genres.str.contains(label).astype(int)

df = df.drop('genres',  axis=1)

if len(df.drop_duplicates(subset=['movie_title',
	'title_year',
	'movie_imdb_link'])) < len(df):
	#print("duplicate title exist")
	duplicates = df[df.movie_title.map(df.movie_title.value_counts() > 1)]
	duplicates.sort('movie_title')[['movie_title', 'title_year']]
	df =  df.drop_duplicates(subset=['movie_title', 'title_year', 'movie_imdb_link'])
#print(df.info())

counts = df.language.value_counts()
df.language = df.language.map(counts)
#print(df.language)
count = df.country.value_counts()
df.country = df.country.map(count)

counts = df.content_rating.value_counts()
df.content_rating = df.content_rating.map(counts)

unique_words =  set()
for wordlist in df.plot_keywords.str.split('|').values:
	if wordlist is not np.nan:
		unique_words = unique_words.union(set(wordlist))
plot_wordbag = list(unique_words)
#print(plot_wordbag)
for word in plot_wordbag:
	df['plot_has_' + word.replace(' ', '-')] = df.plot_keywords.str.contains(word).astype(float)
df = df.drop('plot_keywords',  axis=1)
df.director_name = df.director_name.map(df.director_name.value_counts())
counts = pd.concat([df.actor_1_name, df.actor_2_name, df.actor_3_name]).value_counts()

df.actor_1_name = df.actor_1_name.map(counts)
df.actor_2_name = df.actor_2_name.map(counts)
df.actor_3_name = df.actor_3_name.map(counts)

df = df.drop(['movie_imdb_link'], axis=1)

#print(df.select_dtypes(include=['0']).columns)
#print(df.shape)

#new_style = {'grid': False}
#matplotlib.rc('axes', **new_style)
#plt.matshow(~df.isnull())
#plt.title('Missing values in data')
#plt.show()

nullcount = df.isnull().sum(axis=1)

	#df.to_pickle("movie_rec.pickle")

	#df = pd.read_pickle("movie_rec.pickle")


ndf = df.dropna(thresh=100)
#print(ndf.shape, df.shape)
#plt.matshow(~df.isnull())
#plt.title('Missing values in data')
#plt.show()



def reg_class_fill(df, column, classifier):
	ndf = df.dropna(subset=[col for col in df.columns if col != column])
	nullmask = ndf[column].isnull()
	train, test = ndf[~nullmask], ndf[nullmask]
	train_x, train_y = train.drop(column, axis=1), train[column]
	classifier.fit(train_x, train_y)
	if len(test) > 0:
		test_x, test_y = test.drop(column, axis=1), test[column]
		values = classifier.predict(test_x)
		test_y = values
		new_x, new_y = pd.concat([train_x, test_x]), pd.concat([train_x, test_y])
		newdf = new_x[column] = new_y
		return new_df
	else:
		return ndf


r, c = KNeighborsRegressor, KNeighborsClassifier
title_encoder = LabelEncoder()
title_encoder.fit(ndf.movie_title)
ndf.movie_title = title_encoder.transform(ndf.movie_title)



impute_order = [('director_name', c), ('title_year', c),
	            ('actor_1_name', c), ('actor_2_name', c), ('actor_3_name', c),
        	        ('gross', r), ('budget', r), ('aspect_ratio', r),
            	    ('content_rating', r), ('num_critic_for_reviews', r)]

for col, classifier in impute_order:
	ndf = reg_class_fill(ndf, col, classifier())

#print(ndf[ndf.columns[:25]].isnull().sum())
titles = title_encoder.inverse_transform(ndf.movie_title)
#print(titles)

	#ndf.to_pickle("movie_rec.pickle")
'''
else:
	ndf = pd.read_pickle("movie_rec.pickle")
	title_encoder = LabelEncoder()
	title_encoder.fit(ndf.movie_title)
	ndf.movie_title = title_encoder.transform(ndf.movie_title)
	titles = title_encoder.inverse_transform(ndf.movie_title)

'''

import time


def get_movies(names):
	movies = []
	found = [i for i in titles if names.lower() in i.lower()]
	if len(found) > 0:
		movies.append(found[0])
		print(names, ': ', found, 'added', movies[-1], 'to movies')
	else:
		print(names, ':', found)
	print('-'*10)
	print(movies)
	moviecodes = title_encoder.transform(movies)
	return moviecodes, movies




def recommend(movies, tree, titles, data):
	titles = list(titles)
	length, recommendations = len(movies) + 1, []
	for i, movie in enumerate(movies):
		weight = length - i
		dist, index = tree.query([data[titles.index(movie)]], k=3)
		for d,m in zip(dist[0], index[0]):
			recommendations.append((d*weight, titles[m]))

	recommendations.sort()

	rec = [i[1].strip() for i in recommendations if i[1] not in movies]
	rec = [i[1] for i in sorted([(v,k) for k, v in Counter(rec).items()], reverse=True)]

	return rec






from sklearn.neighbors import KDTree
from collections import Counter


from tkinter import *



def show():

	master.quit()

	#if e1.get() == null:

	s = ""
	names = e1.get()

	print(names)

	moviecodes, movies = get_movies(names)


	data =  ndf.drop('movie_title', axis=1)
	data = MinMaxScaler().fit_transform(data)


	tree = KDTree(data, leaf_size=2)


	rec = recommend(movies, tree, titles, data)
	for index,  movie in enumerate(rec[:9]):
		s = s + movie + "\n"

	print(s)
	Label(master, text=s).grid(row=7)

	
master = Tk()

Label(master, text="Write the name of your favourite movie").grid(row=0)

Label(master).grid(row=1)
Label(master).grid(row=2)

e1 = Entry(master)

e1.grid(row=3, column=0)

Label(master).grid(row=4)


Button(master, text='find movies which I can like', command=show).grid(row=5, column=0)


Label(master, text="hey").grid(row=6)

time.sleep(10)


mainloop()



