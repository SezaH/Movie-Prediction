import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from user_dataframe import user_movie_matrix
from user_dataframe import movies
from sklearn.metrics.pairwise import cosine_similarity


def recommend(target_user_id, n_recommendations=10, num_neighbors=20, print_output=True):

    # KNN
    user_index = target_user_id - 1
    knn = NearestNeighbors(num_neighbors, metric='cosine').fit(user_movie_matrix)
    _, indices = knn.kneighbors(user_movie_matrix, num_neighbors)
    nbr_indices = indices[user_index]
    # knn_matrix is user_movie_matrix with n nearest neighbors for user id
    # first element is the target user
    knn_matrix = user_movie_matrix.iloc[nbr_indices, ]

    # user similarity
    similarity = cosine_similarity(knn_matrix)
    sum_similarity = np.absolute(similarity).sum(axis=0)
    pred_df = pd.DataFrame(similarity.dot(knn_matrix), columns=knn_matrix.columns).divide(sum_similarity,
                                                                                                        axis=0)
    # for a given user, select top recommendations for movies not already rated
    user_ratings = user_movie_matrix.iloc[user_index]
    already_rated_labels = list(user_ratings.iloc[user_ratings.nonzero()[0]].index)
    # drop already rated movies and then sort by top recommendations
    results_df = pred_df.iloc[0].drop(labels=already_rated_labels).sort_values(ascending=False).to_frame()

    final_recs = pd.merge(results_df, movies, on='movieId')[['movieId', 'title', 'genres']].head(n_recommendations)

    if print_output:
        # print output:
        print("USERS TOP MOVIES:")
        top_user_ratings = user_ratings.sort_values(ascending=False).to_frame()
        print(pd.merge(top_user_ratings, movies, on='movieId').head(15).to_string())

        print('TOP %d RECOMMENDATIONS:' % n_recommendations)
        print(final_recs.to_string())

    # return final recommendations ONLY
    return final_recs

