from dataframe import data
import matplotlib.pyplot as plt 

def graph(feature1, feature2):
    x = data[feature1].values
    y = data[feature2].values
        
    #Basic Plotting
    plt.scatter(x,y)

    plt.xlabel(feature1)
    plt.ylabel(feature2)

    title = feature1 + ' vs ' + feature2
    plt.title(title)
    plt.show()
