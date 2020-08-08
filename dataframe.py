import pandas as pd # To read csv file into DataFrame
import numpy as np
from helpers import get_unique_genres
from sklearn.utils import shuffle
from scipy import stats
from sklearn import preprocessing
from actor_ratings import add_actor_ratings
import os

#read csv file into dataframe, parse certain columns as we read themn in 
data = pd.read_csv('data/movies_metadata.csv', usecols=['budget', 'genres', 'id', 'original_language', 'popularity', 
                                                        'production_companies', 'production_countries', 'release_date', 
                                                       'revenue', 'runtime', 'title', 'vote_average', 'vote_count'], 
                   low_memory=False)
data = data.dropna()
data = data.drop_duplicates(subset='id', keep='last')
data = data.drop(columns=['id', 'original_language'])
data['release_date'] = pd.to_datetime(data['release_date'], format='%Y-%m-%d')
data[['budget', 'popularity']] = data[['budget', 'popularity']].apply(pd.to_numeric)

data = data[(data['budget'] > 20000) & (data['revenue'] > 0) & (data['genres'] != '[]') & (data['runtime'] != 0) & 
           (data['vote_count'] > 10)]
data = data[['title', 'production_companies', 'production_countries', 'genres', 'release_date', 'budget', 'revenue', 
         'runtime', 'popularity', 'vote_count', 'vote_average']] #rearrange columns so numerical data are on the right

# remove samples with high zscore
zscore_threshold = 3
data = data[(np.abs(stats.zscore(data.iloc[:, 5:])) < zscore_threshold).all(axis=1)]
data = shuffle(data)
# copy data before removing / adding columns
full_data = data.copy()

data['month'] = pd.DatetimeIndex(data['release_date']).month
data['year'] = pd.DatetimeIndex(data['release_date']).year
data['was_summer'] = np.where((data['month'] >= 6) & (data['month'] <= 8), 1, 0) #June, July, August

genres = get_unique_genres()
for g in genres:
	data["is_genre_" + g.replace(" ","_")] = list(map(lambda x : int(g in str(x)), data["genres"]))

data = data.iloc[:, 5:]	#only contains numeric data

if os.environ.get("WITH_ACTORS"):
	data = data_with_actor_ratings = add_actor_ratings(data)
