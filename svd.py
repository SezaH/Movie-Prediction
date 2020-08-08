import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from user_dataframe import movies, user_movie_matrix

def recommend(target_user_id, n_recommendations=10, latent_factors=50, print_output=True):
    # sparse matrix SVD approximation
    U, S, Vt = svds(user_movie_matrix, k=latent_factors)
    S = np.diag(S)
    # element (i, j) corresponds to ith user's predicted rating for jth movie
    predicted_ratings = np.dot(np.dot(U, S), Vt)
    pred_df = pd.DataFrame(predicted_ratings, columns=user_movie_matrix.columns)

    # for a given user, select top recommendations for movies not already rated
    user_index = target_user_id - 1
    user_ratings = user_movie_matrix.iloc[user_index]
    already_rated_labels = list(user_ratings.iloc[user_ratings.nonzero()[0]].index)
    # drop already rated movies and then sort by top recommendations
    results_df = pred_df.iloc[user_index].drop(labels=already_rated_labels).sort_values(ascending=False).to_frame()

    final_recs = pd.merge(results_df, movies, on='movieId')[['movieId', 'title', 'genres']].head(n_recommendations)

    if print_output:
        # print output:
        print("USERS TOP MOVIES:")
        top_user_ratings = user_ratings.sort_values(ascending=False).to_frame()
        print(pd.merge(top_user_ratings, movies, on='movieId').head(15).to_string())

        print('TOP %d RECOMMENDATIONS:' % n_recommendations)
        print(final_recs.to_string())

    # return prediction matrix and final recommendations
    return pred_df, final_recs
