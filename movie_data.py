import pandas as pd
from tmdbv3api import TMDb
from tmdbv3api import Movie
from user_dataframe import movies, user_movie_matrix
import re

tmdb = TMDb()
tmdb.api_key = 'f773fb6bca4bab1b4a000c4c5e408298'

def sep_year_title(string):
	match = re.match(r'(.*) \((....)\)', string)

	if match == None:
		return ("N/A", "N/A")

	return (match.group(1), match.group(2))

class MovieData:
	def __init__(self, exact_title):
		self.movie = movies.loc[(movies['title'] == exact_title)].max()

		self.title, self.year = sep_year_title(self.movie.title)
		self.id = self.movie.movieId

		self.genres = self.movie.genres

		online_results = query_tmdb(self.title, self.year)

		self.poster_url = online_results[0]
		self.overview = online_results[1]

	def genres_as_str(self):
		return self.genres

def query_tmdb(title, release_year):
	movie = Movie()
	search = movie.search(title)
	image_url = ''
	summary = ''

	for res in search:
		if res.poster_path != None and res.release_date[0:4] == release_year:
			image_url ='http://image.tmdb.org/t/p/w500/' + res.poster_path
			summary = res.overview

	return [image_url, summary]
