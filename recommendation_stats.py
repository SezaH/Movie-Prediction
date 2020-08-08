import pandas as pd
from user_dataframe import ratings, user_movie_matrix, movies
import matplotlib.pyplot as plt

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

# frequency plot
counts.plot(kind='bar', width=1.0, stacked=False, color='blue')
plt.xticks([])
plt.xlabel('Movie Ids')
plt.ylabel('Frequency')
plt.title('Movie Rating Frequency')
plt.show()

