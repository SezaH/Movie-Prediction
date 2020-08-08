import dataframe
import json

def get_index_from_sample(sample):
	return sample.index.values[0]

def get_title_from_index(index):
	return dataframe.full_data.loc[index]["title"]

def get_unique_genres():
	genres = set([])

	for j in dataframe.full_data["genres"]:
		obj = json.loads(j.replace("'", "\""))  # replace single quotes so json.loads can parse it
		for entry in obj:
			genres.add(entry["name"])
	return genres

def get_genre_count(data):
	genres = {}

	for j in data["genres"]:
		obj = json.loads(j.replace("'", "\""))  # replace single quotes so json.loads can parse it
		for entry in obj:
			if entry["name"] in genres.keys():
				genres[entry["name"]] += 1
			else:
				genres[entry["name"]] = 1
	return genres
