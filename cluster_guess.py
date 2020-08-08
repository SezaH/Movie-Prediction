from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
#################################################################Slightly changed dataframe code
import pandas as pd # To read csv file into DataFrame
import numpy as np
from ast import literal_eval #forces pandas to evaluate the array as an array and not as a string
import datetime
from dataframe import full_data
from sklearn import preprocessing

from scipy import stats

def cluster_guess(data, show_plots = False):
    movieTitles = full_data["title"]

    ##############################################################################New code
    rows = ["" for x in range(len(data))]
    count = 0
    for index, row in data.iterrows():
        rows[count] = index
        count = count + 1

    data[["year", "month", "budget", "runtime", "revenue", "popularity", "vote_average", "vote_count"]] = preprocessing.MinMaxScaler().fit_transform(data[["year", "month", "budget", "runtime", "revenue", "popularity", "vote_average", "vote_count"]])

    # remove a lot of genre columns to reduce dimension
    data = data.drop(["is_genre_Mystery", "is_genre_Horror", "is_genre_Fantasy", "is_genre_Science_Fiction", "is_genre_Documentary", "is_genre_History", "is_genre_Music", "is_genre_TV_Movie", "is_genre_War", "is_genre_Adventure", "is_genre_Foreign", "is_genre_Western", "is_genre_Crime", "is_genre_Romance", "is_genre_Family", "is_genre_Animation"], axis=1)

    # generalize thriller genre a little bit
    data["is_thriller"] = data["is_genre_Action"] | data["is_genre_Thriller"]

    data = data.drop(["is_genre_Action", "is_genre_Thriller"], axis=1)

    # Finding best number of clusters
    if show_plots:
        dists = [0 for i in range(40)]
        for i in range(5,40):
            numClusters = i
            clusters = KMeans(n_clusters = numClusters, init = 'random', max_iter = 2000, precompute_distances = 'auto')
            clusters.fit(data)
            dists[i] = clusters.inertia_
        plt.plot(dists)
        plt.show()
        # Found ~ 15 is good

        silhouette = [0 for i in range(60)]
        for i in range(5,60):
            numClusters = i
            clusters = KMeans(n_clusters = numClusters, init = 'random', max_iter = 2000, precompute_distances = 'auto')
            predictions = clusters.fit_predict(data)
            silhouette[i] = silhouette_score(data, predictions, metric='euclidean')
        plt.plot(silhouette)
        plt.show()
        # Found ~ 15

    else:
        numClusters = 15
        moviesPerClust = 4
        clusters = KMeans(n_clusters = numClusters, init = 'random', max_iter = 2000, precompute_distances = 'auto')

        indices = clusters.fit_predict(data)

        clusterIndices = [[] for i in range(numClusters)]
        for i in range(len(data)):
            clusterIndices[indices[i]].append(rows[i])

        likeDislike = [0 for i in range(numClusters)]
        print("Answer with y-Yes, n-No, d-Don't know")
        for i in range(numClusters):
            movieIDs = random.sample(range(0, len(clusterIndices[i])), moviesPerClust)
            for j in range(moviesPerClust):
                answer = input("Do you like " + movieTitles[clusterIndices[i][movieIDs[j]]] + "? ")
                if (answer == 'y'):
                    likeDislike[i] = likeDislike[i] + 3
                if (answer == 'n'):
                    likeDislike[i] = likeDislike[i] - 2
                if (answer == 'd'):
                    likeDislike[i] = likeDislike[i] - 1

        bestCluster = likeDislike.index(max(likeDislike))
        print("You may also like: ")
        movieIDs = random.sample(range(0, len(clusterIndices[bestCluster])), 10)
        for i in range(10):
            print(movieTitles[clusterIndices[bestCluster][movieIDs[i]]])
