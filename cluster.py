import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import datasets
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


def batchify(a, batch_size=512):
    n = (len(a) // batch_size) + len(a) % batch_size
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


def get_initial_centroids(data, groups, num_clusters):
    new_data = torch.zeros((len(groups), data.shape[1]))
    for ix, group in enumerate(groups):
        new_data[ix, :] = torch.mean(data[group.id.values], 0)
    kmeans = KMeans(num_clusters=num_clusters, device=torch.device('cuda:0'), balanced=False)
    _ = kmeans.fit(
                X=new_data, distance='euclidean', iter_limit=1, balanced=False, tqdm_flag=False, online=False
    )
    # batched_X = batchify(new_data, batch_size=16)
    # counter = 0
    # for batch in tqdm(batched_X):
    #     try:
    #         _ = kmeans.fit(
    #             X=batch, distance='euclidean', iter_limit=1, balanced=False, tqdm_flag=False, online=False, iter_k=counter
    #         )
    #     except:
    #         import pdb; pdb.set_trace()
        # counter+= 1
    return kmeans

def featurize(df):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
    blobs = vectorizer.fit_transform(tqdm(df.text))
    svd = TruncatedSVD(n_components=50)
    svd = svd.fit(blobs)
    blobs = svd.transform(blobs)
    return blobs

if __name__ == '__main__':
    seed = 235
    num_clusters = 8
    balanced = True
    debug = True
    # set random seed
    np.random.seed(seed)

    if debug:
        n_samples = 8000
        blobs = datasets.make_blobs(n_samples=n_samples,
                                    random_state=seed,
                                    centers=[[1, 1], [-1, -1], [1, -1]],
                                    cluster_std=0.6)
        X = torch.from_numpy(blobs[0][:n_samples//2])
        y = torch.from_numpy(blobs[0][n_samples//2:])
    else:

        vecs = np.load('/private/home/suching/demix-data/demix_vecs_test/0.pt.vecs.npy')
        ids = np.load('/private/home/suching/demix-data/demix_vecs_test/0.pt.ids.npy')

        
        # df = pd.read_json("pop_sample_cc.jsonl", lines=True)
        # df['id'] = range(len(df))
        n_samples = vecs.shape[0]
        # groups = df.groupby('domain')
        # id_groups = [group for name, group in groups]
        # features = featurize(df)
        features = torch.from_numpy(vecs)
        # domain_kmeans = get_initial_centroids(features, id_groups, num_clusters)
        # metadata_clusters = domain_kmeans.predict(features, distance='euclidean')
        # m_c = torch.zeros(features.shape[0], num_clusters)
        # m_c[range(metadata_clusters.shape[0]), metadata_clusters] = 1
        # features = torch.cat([m_c, features],dim=1)

        X = features[:n_samples//2]
        y = features[n_samples//2:]

    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    kmeans = KMeans(n_clusters=8, cluster_centers=initial_centers, device=torch.device('cuda:0'), balanced=True)
    batched_X = batchify(X, batch_size=8)

    counter = 0
    # for batch in tqdm(batched_X):
    _ = kmeans.fit(
        X=X, distance='euclidean', iter_limit=100, tqdm_flag=True, online=False
    )
    # counter+= 1


    kmeans.save("model.pkl")
    kmeans = kmeans.load("model.pkl")

    batched_y = batchify(y, batch_size=8)
    cluster_ids_y_ = []

    cluster_ids_y = kmeans.predict(
            X=y
        )

    # for batch in tqdm(batched_y):
    #     output = kmeans.predict(
    #         X=batch,  distance='euclidean', balanced=balanced
    #     )
    #     cluster_ids_y_.append(output)
        
    # cluster_ids_y = torch.cat(cluster_ids_y_, 0).to('cpu')
    if balanced:
        output = 'balanced_clusters.pdf'
    else:
        output = 'unbalanced_clusters.pdf'
    plot_blobs(y, kmeans.cluster_centers.to('cpu'), cluster_ids_y, output)
    import pdb; pdb.set_trace()
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
    