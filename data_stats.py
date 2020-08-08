from dataframe import data, full_data
from helpers import get_genre_count
import numpy as np

def filter_budget_range(low, high):
	return full_data[(full_data["budget"] >= low) & (full_data["budget"] <= high)]

def genre_dict_top_n(genre_dict, n):
	top = []

	dict_copy = dict(genre_dict)

	for i in range(0, n):
		max_count = 0
		genre = "N/A"

		for key in dict_copy.keys():
			if dict_copy[key] > max_count:
				max_count = dict_copy[key]
				genre = key

		if genre != "N/A":
			top.append(genre)

		del dict_copy[genre]
	return top

def show_stats():
	print("average vote score:", np.mean(data["vote_average"]))
	print("median vote score:", np.median(data["vote_average"]))
	print("average vote count:", np.mean(data["vote_count"]))
	print("median vote count:", np.median(data["vote_count"]))
	print("average budget:", np.mean(data["budget"]))
	print("median budget:", np.median(data["budget"]))

	genres_0_100k_dict   = get_genre_count(filter_budget_range(       0,   100000))
	genres_100k_1m_dict  = get_genre_count(filter_budget_range(  100000,  1000000))
	genres_1m_5m_dict    = get_genre_count(filter_budget_range( 1000000,  5000000))
	genres_5m_10m_dict   = get_genre_count(filter_budget_range( 5000000, 10000000))
	genres_10m_20m_dict  = get_genre_count(filter_budget_range(10000000, 20000000))
	genres_20m_50m_dict  = get_genre_count(filter_budget_range(20000000, 50000000))
	genres_50m_plus_dict = get_genre_count(filter_budget_range(50000000, 1000000000000))

	number_top = 5
	print("top", number_top, "genres of movies in budget range of 0 and 100k:",  genre_dict_top_n(genres_0_100k_dict,   number_top))
	print("top", number_top, "genres of movies in budget range of 100k and 1m:", genre_dict_top_n(genres_100k_1m_dict,  number_top))
	print("top", number_top, "genres of movies in budget range of 1m and 5m:",   genre_dict_top_n(genres_1m_5m_dict,    number_top))
	print("top", number_top, "genres of movies in budget range of 5m and 10m:",  genre_dict_top_n(genres_5m_10m_dict,   number_top))
	print("top", number_top, "genres of movies in budget range of 10m and 20m:", genre_dict_top_n(genres_10m_20m_dict,  number_top))
	print("top", number_top, "genres of movies in budget range of 20m and 50m:", genre_dict_top_n(genres_20m_50m_dict,  number_top))
	print("top", number_top, "genres of movies in budget range above 50m:",      genre_dict_top_n(genres_50m_plus_dict, number_top))
