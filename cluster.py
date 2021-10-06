import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import KMeans
from sklearn.decomposition import PCA
from sklearn import datasets
from fairseq import libbase
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns


def batchify(a, n=2):
    for i in np.array_split(a, n, axis=0):
        yield i


def get_clusters(X, cluster_size=None, kmeans=None, predict=False, balanced=False):
    if cluster_size is not None:
        n_clusters = int(np.ceil(len(X)/cluster_size))
    else:
        n_clusters = kmeans.n_clusters
    if not kmeans:
        kmeans = KMeans(n_clusters)
    
    if balanced:
        if predict:
            clusters = kmeans.predict(X)
        else:
            cs = []
            kmeans = kmeans.fit(X)
            centers = kmeans.cluster_centers_
            centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
            distance_matrix = cdist(X, centers)
            clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
            for index in range(n_clusters):
                selected = np.nonzero(clusters == index)[0]
                selected = X[selected,:]
                if selected.shape[0] == 0:
                    selected = batch[np.random.randint(len(batch)),:]
                kmeans.cluster_centers_[index] = selected.mean(dim=0)
    else:
        if predict:
            clusters = kmeans.predict(X)
        else:
            kmeans = kmeans.fit(X)
            clusters = kmeans.predict(X)
    return clusters, kmeans

def plot_blobs(data,cluster_centers, labels, plot_file):    
    plt.figure(figsize=(4, 3), dpi=160)
    pca = PCA(n_components=2)
    master = np.concatenate([data, cluster_centers], 0)
    pca = pca.fit(master)
    data = pca.transform(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    cluster_centers = pca.transform(cluster_centers)
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    #plt.axis([-1, 1, -1, 1])
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)


if __name__ == '__main__':
    seed = 235
    num_clusters = 8
    balanced = True
    # set random seed
    np.random.seed(seed)

    n_samples = 8000
    
    blobs = datasets.make_blobs(n_samples=n_samples, n_features=768, random_state=seed, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.6)

    X = torch.from_numpy(blobs[0][:n_samples//2])
    y = torch.from_numpy(blobs[0][n_samples//2:])

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    kmeans = KMeans(num_clusters=8)
    _ = kmeans.fit(
        X=X, num_clusters=num_clusters, distance='euclidean', device=device, iter_limit=25, balanced=balanced
    )


    

    kmeans.save("model.pkl")
    kmeans = kmeans.load("model.pkl")
    cluster_ids_y = kmeans.predict(
        X=y,  distance='euclidean', device=device
    )
    plot_blobs(y, kmeans.cluster_centers, cluster_ids_y, "balanced_clusters.pdf")
    # cluster_ids_x, kmeans = get_clusters(X, cluster_size=n_samples // num_clusters, predict=False, balanced=True)


    

    
    # cluster_ids_y, kmeans = get_clusters(y,  cluster_size=n_samples // num_clusters, predict=True, kmeans=kmeans, balanced=True)

    
    # plot_blobs(X, cluster_centers, cluster_ids_x, "balanced_clusters.pdf")
    # cx = pd.Series(cluster_ids_y)
    # cx = cx.value_counts().sort_index()

    # cluster_ids_x, kmeans = get_clusters(X, predict=False, kmeans=KMeans(n_clusters=num_clusters), balanced=False)

    # cluster_ids_y, kmeans = get_clusters(y, predict=True, kmeans=kmeans, balanced=False)


    # plot_blobs(y, kmeans.cluster_centers_, cluster_ids_y, "unbalanced_clusters.pdf")
    # cy = pd.Series(cluster_ids_y)
    # cy = cy.value_counts().sort_index()
    # df = pd.DataFrame({"balanced": cx,
    #                    "unbalanced": cy})
    # fig, ax = plt.subplots(1,1)
    # ax = sns.boxplot(data=df, ax=ax)
    # ax.set_ylabel("cluster size")
    # ax.set_xlabel("cluster type")
    # plt.savefig("boxplot.pdf", dpi=300)
    
