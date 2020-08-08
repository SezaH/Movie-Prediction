import pandas as pd
import svd
import cf_knn
import numpy as np
from user_dataframe import ratings, user_movie_matrix, movies
import matplotlib.pyplot as plt

def cf_misc_stats():
    # unique totals
    print('Number of unique users: %d' % ratings['userId'].nunique())
    print('Number of unique movies: %d' % ratings['movieId'].nunique())

    # aggregate user stats
    print('Ratings per user:')
    user_info = user_movie_matrix.astype(bool).sum(axis=1)
    print(user_info.describe())

    # most frequent movies
    print('Most Frequently Rated Movies:')
    counts = ratings['movieId'].value_counts()
    counts.sort_index()
    count_df = counts.to_frame().reset_index().rename(index=str, columns={"index": "movieId", "movieId": "count"})
    print(pd.merge(count_df, movies, on='movieId').head(20).to_string())

    # COVERAGE
    userId = 47

    total_recs = []

    # coverage KNN
    for i in range(1, len(user_movie_matrix) + 1):
        print(i)
        final_recs = cf_knn.recommend(i, print_output=False)
        total_recs.append(final_recs['movieId'].values)

    # calculate coverage
    unique_recs = np.unique(total_recs)
    print('KNN Unique recs: %d' % unique_recs.size)

    total_recs = []

    # coverage SVD
    for i in range(1, len(user_movie_matrix) + 1):
        print(i)
        _, final_recs = svd.recommend(i, print_output=False)
        total_recs.append(final_recs['movieId'].values)

    # calculate coverage
    unique_recs = np.unique(total_recs)
    print('SVD Unique recs: %d' % unique_recs.size)


    # FREQUENCY PLOT
    counts.plot(kind='bar', width=1.0, stacked=False, color='blue')
    plt.xticks([])
    plt.xlabel('Movie Ids')
    plt.ylabel('Frequency')
    plt.title('Movie Rating Frequency')
    plt.show()
