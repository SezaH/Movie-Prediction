from sklearn import ensemble
import matplotlib.pyplot as plt
from dataframe import data

for response_var in ['revenue', 'budget', 'vote_count', 'vote_average']:
    X = data.loc[:, data.columns != response_var]
    y = data.loc[:, response_var]

    rf = ensemble.RandomForestRegressor()
    model = rf.fit(X, y)
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]
    plt.figure()
    plt.title("Feature Importance for " + response_var)
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()
