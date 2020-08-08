import json
import numpy as np
import pandas as pd
from sklearn import preprocessing  # LabelEncoder
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from dataframe import full_data

def scatter_2d():
	# get data
	data = full_data
	df_genres = data["genres"]
	data = data.drop(["genres", "production_companies", "production_countries", "release_date", "title"], axis=1)

	# parse genres column and get first genre
	genres = df_genres.values  # np array
	first_genres = []
	for s in genres:
		obj = json.loads(s.replace("'", "\""))  # replace single quotes so json.loads can parse it
		first_genres.append(obj[0]['name'])

	# run LabelEncoder on first_genres
	le = preprocessing.LabelEncoder()
	le.fit(first_genres)
	first_genres_id = le.transform(first_genres)  # np array of ints 0-19

	# histogram via bar graph
	labels, counts = np.unique(first_genres_id, return_counts=True)
	plt.bar(le.classes_, counts, align='center')
	plt.gca().set_xticks(labels)

	# 2d scatter matrix
	scatter_matrix(data, alpha=1, figsize=(8, 8), diagonal='hist', c=first_genres_id)
	plt.show()
