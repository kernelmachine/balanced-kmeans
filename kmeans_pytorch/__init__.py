from functools import partial

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from .soft_dtw_cuda import SoftDTW
from scipy.optimize import linear_sum_assignment
from balanced_assignment import auction_lap
from sklearn.decomposition import PCA
import pickle


def batchify(a, n=2):
    for i in np.array_split(a, n, axis=0):
        yield i



class KMeans(object):
    def __init__(self, num_clusters, cluster_centers=None):
        self.num_clusters = num_clusters
        self.cluster_centers = cluster_centers
    
    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['num_clusters'], saved['cluster_centers'])
    
    def save(self, path_to_file):
        with open(path_to_file, 'wb+') as f :
            pickle.dump(self.__dict__, f)

    def initialize(self, X):
        """
        initialize cluster centers
        :param X: (torch.tensor) matrix
        :param num_clusters: (int) number of clusters
        :return: (np.array) initial state
        """
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        initial_state = X[indices]
        return initial_state
    
    def fit(
            self,
            X,
            num_clusters,
            distance='euclidean',
            cluster_centers=[],
            tol=1e-3,
            tqdm_flag=True,
            iter_limit=0,
            device=torch.device('cpu'),
            gamma_for_soft_dtw=0.001,
            balanced=False
    ):
        """
        perform kmeans
        :param X: (torch.tensor) matrix
        :param num_clusters: (int) number of clusters
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param tol: (float) threshold [default: 0.0001]
        :param device: (torch.device) device [default: cpu]
        :param tqdm_flag: Allows to turn logs on and off
        :param iter_limit: hard limit for max number of iterations
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
        """
        if tqdm_flag:
            print(f'running k-means on {device}..')

        if distance == 'euclidean':
            pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=device)
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        
        # transfer to device
        X = X.to(device)

        # initialize
        if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
            initial_state = self.initialize(X)
        else:
            if tqdm_flag:
                print('resuming')
            # find data point closest to the initial cluster center
            initial_state = cluster_centers
            dis = pairwise_distance_function(X, initial_state)
            
            
            # choice_points = torch.argmin(dis, dim=0)
            initial_state = X[choice_points]
            initial_state = initial_state.to(device)
            

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        # cluster = torch.arange(num_clusters).repeat_interleave(X.shape[0] // num_clusters).to(device)
        done=False
        while True:
            if balanced:
                distance_matrix = pairwise_distance_function(X, initial_state)
                cluster_assignments = auction_lap(-distance_matrix)
                
                # SCIPY LINEAR ASSIGNMENT SOLVER
                # cluster_assignments = linear_sum_assignment(-distance_matrix.cpu().numpy(), maximize=True)[1] // (X.shape[0] // num_clusters)   
                # cluster_assignments = torch.IntTensor(cluster_assignments).cuda()
                
            else:
                dis = pairwise_distance_function(X, initial_state)
                cluster_assignments = torch.argmin(dis, dim=1)
            
            initial_state_pre = initial_state.clone()
            for index in range(num_clusters):
                selected = torch.nonzero(cluster_assignments == index).squeeze().to(device)

                selected = torch.index_select(X, 0, selected)

                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                
                initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1

            # update tqdm meter
            if tqdm_flag:
                tqdm_meter.set_postfix(
                    iteration=f'{iteration}',
                    center_shift=f'{center_shift ** 2:0.6f}',
                    tol=f'{tol:0.6f}'
                )
                tqdm_meter.update()
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break
                
        self.cluster_centers = initial_state.cpu()

        return cluster_assignments.cpu()


    def plot(self, data, labels, plot_file):
        if self.cluster_centers is None:
            raise Exception("Fit the KMeans object first before plotting!")
        plt.figure(figsize=(4, 3), dpi=160)
        pca = PCA(n_components=2)
        master = np.concatenate([data, self.cluster_centers], 0)
        pca = pca.fit(master)
        data = pca.transform(data)
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        cluster_centers = pca.transform(cluster_centers)
        plt.scatter(
            self.cluster_centers[:, 0], self.cluster_centers[:, 1],
            c='white',
            alpha=0.6,
            edgecolors='black',
            linewidths=2
        )
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)


    def predict(
            self,
            X,
            distance='euclidean',
            device=torch.device('cpu'),
            gamma_for_soft_dtw=0.001,
            tqdm_flag=True
    ):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param cluster_centers: (torch.tensor) cluster centers
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param device: (torch.device) device [default: 'cpu']
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor) cluster ids
        """
        if tqdm_flag:
            print(f'predicting on {device}..')

        if distance == 'euclidean':
            pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=device)
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        # transfer to device
        X = X.to(device)
        distance_matrix = pairwise_distance_function(X, self.cluster_centers)
        cluster_assignments = auction_lap(-distance_matrix)
        # dis = pairwise_distance_function(X, cluster_centers)
        # cluster_assignments = torch.argmin(dis, dim=1)

        return cluster_assignments.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
    # if tqdm_flag:
        # print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def pairwise_soft_dtw(data1, data2, sdtw=None, device=torch.device('cpu')):
    if sdtw is None:
        raise ValueError('sdtw is None - initialize it with SoftDTW')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # (batch_size, seq_len, feature_dim=1)
    A = data1.unsqueeze(dim=2)

    # (cluster_size, seq_len, feature_dim=1)
    B = data2.unsqueeze(dim=2)

    distances = []
    for b in B:
        # (1, seq_len, 1)
        b = b.unsqueeze(dim=0)
        A, b = torch.broadcast_tensors(A, b)
        # (batch_size, 1)
        sdtw_distance = sdtw(b, A).view(-1, 1)
        distances.append(sdtw_distance)

    # (batch_size, cluster_size)
    dis = torch.cat(distances, dim=1)
    return dis
