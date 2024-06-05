import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (
    v_measure_score,
    adjusted_rand_score,
)


def clustering(embeddings, y, num_class):
    kmeans_input = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    labels = y.cpu().numpy()
    #completeness = completeness_score(labels, pred)
    #hm = homogeneity_score(labels, pred)
    nmi = v_measure_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    auccuary = accuracy_score(labels, pred)
    return nmi, ari, auccuary


def node_clustering_eval(embeddings, y, num_class):

    kmeans_input = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    labels = y.cpu().numpy()
    # completeness = completeness_score(labels, pred)
    # hm = homogeneity_score(labels, pred)
    nmi = v_measure_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    auccuary = accuracy_score(labels, pred)
    return nmi, ari, auccuary