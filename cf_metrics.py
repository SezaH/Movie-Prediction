from user_dataframe import user_movie_matrix
from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

def cf_metrics():
	# Load the movielens-100k dataset (download it if needed).
	reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines = 1 )
	data = Dataset.load_from_file('data/ml-latest-small/ratings.csv', reader=reader)

	# Use the famous SVD algorithm.
	algo = SVD()

	# Run 5-fold cross-validation and print results.
	cross_validate(algo, data , measures=['RMSE', 'MAE'], cv=5, verbose=True)

	algo = KNNBasic()

	# Run 5-fold cross-validation and print results.
	cross_validate(algo, data , measures=['RMSE', 'MAE'], cv=5, verbose=True)
