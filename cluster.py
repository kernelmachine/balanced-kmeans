import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.decomposition import PCA
from sklearn import datasets
from fairseq import libbase
import pandas as pd


def batchify(a, n=2):
    for i in range(a.shape[0] // n):
        yield a[n*i:n*(i+1)]


if __name__ == '__main__':
    seed = 18
    num_clusters = 5

    # set random seed
    np.random.seed(seed)

    n_samples = 2000

    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=num_clusters,  n_features = 64)

    X = torch.from_numpy(blobs[0][:1000])
    y = torch.from_numpy(blobs[0][1000:])

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    for batch in batchify(X, n=100):
        cluster_ids_x, cluster_centers = kmeans(
            X=batch, num_clusters=num_clusters, distance='cosine', device=device,  balanced=False
        )
    cluster_ids_y = kmeans_predict(
        y, cluster_centers, 'cosine', device=device
    )
    print(pd.Series(cluster_ids_y).value_counts())