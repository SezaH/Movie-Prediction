import pandas as pd  # To read csv file into DataFrame
import numpy as np

movies = pd.read_csv('data/ml-latest-small/movies.csv',
                     usecols=["movieId", "title", "genres"])
ratings = pd.read_csv('data/ml-latest-small/ratings.csv',
                      usecols=["userId", "movieId", "rating", "timestamp"])
links = pd.read_csv('data/ml-latest-small/links.csv',
                    usecols=["movieId", "imdbId", "tmdbId"])
tags = pd.read_csv('data/ml-latest-small/tags.csv',
                   usecols=["userId", "movieId", "tag", "timestamp"])

user_movie_matrix = ratings.pivot(
    index='userId', columns='movieId', values='rating').fillna(0)
