from user_dataframe import tags, movies, user_movie_matrix
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import warnings
import string
warnings.filterwarnings("ignore")

# create DF with movies and their corresponding tags
movies_tags = pd.merge(movies, tags, 'left', on='movieId').groupby(
    'movieId')['tag'].apply(list).reset_index()
# change tag data types from lists to strings
movies_tags['tag'] = movies_tags['tag'].apply(
    lambda tags: ' '.join([str(tag) for tag in tags]))
# replace nan values with empty strings for movies with no tags
movies_tags.loc[movies_tags.tag == "nan", 'tag'] = ""
# create DF with movies and strings of genres separated by whitespace
movies_genres = movies[['movieId', 'genres']]
movies_genres['genres'] = movies_genres['genres'].str.replace('|', ' ')
movie_data = movies_genres
# create new attribute data=genres+tags
movie_data.rename(columns={'genres': 'data'}, inplace=True)
movie_data['data'] += " " + movies_tags['tag']

# create IFIDF matrix
vectorizer = TfidfVectorizer()
TFIDF = vectorizer.fit_transform(movie_data['data'])
# compute similarity between movies
cosine_sim = cosine_similarity(TFIDF, TFIDF)

# set up for use in recommendations
movies = movies.reset_index()
titles = movies['title']
indices = pd.Series(movies.index, index=movies['movieId'])


def get_recommendations(userId=1, n_recommendations=10):
    user_index = userId - 1
    user_ratings = user_movie_matrix.iloc[user_index]
    already_rated_labels = list(
        user_ratings.iloc[user_ratings.nonzero()[0]].index)
    top_user_ratings = user_ratings.sort_values(
        ascending=False).to_frame().where(user_ratings >= 2.5).dropna()
    movie_scores = []
    for movId in top_user_ratings.index:
        movieId = movId
        idx = movies.loc[movies.movieId == movieId].index[0]
        # create list with movieId-cos_sim_vectors tuples
        movie_scores += list(zip(movies.movieId, cosine_sim[idx]))
    # sort by the cosine similarity value
    movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
    movie_scores = movie_scores[1:50]
    movie_ids = set([i[0] for i in movie_scores])
    movie_ids -= set(already_rated_labels)
    final_recs = movies.loc[movies['movieId'].isin(
        movie_ids)][['title', 'genres']].head(n_recommendations)
    user_ratings = user_movie_matrix.iloc[userId-1]

    print("USERS TOP 10 MOVIES:")
    top_user_ratings = user_ratings.sort_values(ascending=False).to_frame()
    print(pd.merge(top_user_ratings, movies, on='movieId').head(10).to_string())

    print(f'TOP {n_recommendations} RECOMMENDATIONS:')
    print(final_recs)
