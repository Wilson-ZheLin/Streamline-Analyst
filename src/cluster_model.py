from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

def KMeans_train(X_train, n_clusters=3, model_params=None):
    if model_params is None: model_params = {}
    kmeans = KMeans(n_clusters=n_clusters, **model_params)
    kmeans.fit(X_train)
    return kmeans

def DBSCAN_train(X_train, model_params=None):
    if model_params is None: model_params = {}
    dbscan = DBSCAN(**model_params)
    dbscan.fit(X_train)
    return dbscan

def GaussianMixture_train(X_train, n_components=1, model_params=None):
    if model_params is None: model_params = {}
    gmm = GaussianMixture(n_components=n_components, **model_params)
    gmm.fit(X_train)
    return gmm

