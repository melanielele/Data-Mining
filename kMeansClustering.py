#  THIS CODE WAS OUR OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
#  CODE WRITTEN BY OTHER STUDENTS.
#  Zhangyi Ye & Zhiyue Zhao

import sys
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

# This function returns the Euclidean Distance between two vectors
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# This function takes in the data and number of clusters and returns the location of the k clusters
def initializeClusterCenter(data, k):
    dataCopy = deepcopy(data) # avoid operating directly on the data

    # randomly pick k data points as cluster centers
    index = np.random.randint(len(dataCopy))
    centroid = np.array([dataCopy[index]])
    #print(centroid)
    dataCopy = np.delete(dataCopy, index, 0) # delete ones just picked
    for i in range(k - 1):
        index = np.random.randint(len(dataCopy))
        centroid = np.append(centroid, [dataCopy[index]], axis=0)
        dataCopy = np.delete(dataCopy, index, 0)
    #print(centroid)
    return centroid

def kMeans(data, k):

    centroid = initializeClusterCenter(data, k) # initialize centroids

    OldC = np.zeros(centroid.shape)  # store old values of centroids
    clusters = np.zeros(len(X))  # store the cluster number for each data point
    err = dist(centroid, OldC, None)  # Distance between new centroids and old centroids

    # loop until centroids do not move
    while err != 0:
        # Assigning data points to the closest cluster
        for i in range(len(X)):
            distances = dist(X[i], centroid)
            clusters[i] = np.argmin(distances)
        # Storing the old centroid values
        clusters = clusters.astype(int)
        OldC = deepcopy(centroid)

        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            centroid[i] = np.mean(points, axis=0)
        err = dist(centroid, OldC, None) # update the error

    return clusters, centroid


# unpacking user inputs
inputFile = sys.argv[1]
k = int(sys.argv[2])
outputFile = sys.argv[3]

# Import the dataset and use numerical values only
df = pd.read_csv(inputFile, header=None)
#print(df.isnull().values.any())
df = df.replace('?',np.nan)
df = df.fillna(0)
X = np.array(df.drop([0], axis=1))
X = X.astype(np.float)
#print(X)
# Run k-means algorithm
clusters, centroids = kMeans(X, k)
#print(clusters)
# Output results
df.insert(len(df.columns), 'class', clusters, True)
#df.to_csv(outputFile, sep=',')


# scikit-learn results
'''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
c = kmeans.cluster_centers_ # centroid location
'''

#calculate SSE
sse = 0
for i in range(len(X)):
	mi = centroids[clusters[i]]
	dis = pow(np.linalg.norm(mi-X[i]),2)
	sse +=dis
print(sse)


#calculate  
silhouette_avg = silhouette_score(X, clusters)
print(silhouette_avg)
df = df.append(pd.Series(['sse',sse,'silhouette_avg',silhouette_avg,' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ', ' ',' ',' ',' ',' ','','','','','','','','','','','','','','','','','','','',''],index=df.columns),ignore_index=True)
df.to_csv(outputFile, sep=',')
# Print results
#print("Centroid values")
#print(centroids) # From myAlg

