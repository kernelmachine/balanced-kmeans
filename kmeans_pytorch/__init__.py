from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm
import pandas as pd
from .soft_dtw_cuda import SoftDTW
from scipy.optimize import linear_sum_assignment
# from balanced_assignment import auction_lap
from sklearn.decomposition import PCA
import pickle

import torch

def auction_lap(job_and_worker_to_score, return_token_to_worker=True):
    eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
    eps.clamp_min_(1e-04)
    assert not torch.isnan(job_and_worker_to_score).any()
    if torch.isnan(job_and_worker_to_score).any():
        raise Exception("NaN distance")
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    num_workers, num_jobs = worker_and_job_to_score.size()
    jobs_per_worker = num_jobs // num_workers
    value = torch.clone(worker_and_job_to_score)
    bids = torch.zeros((num_workers, num_jobs), dtype=worker_and_job_to_score.dtype, device=worker_and_job_to_score.device, requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,), dtype=worker_and_job_to_score.dtype, device=worker_and_job_to_score.device, requires_grad=False)
    while True:
        top_values, top_index = value.topk(jobs_per_worker + 1, dim=1)
        # Each worker bids the difference in value between that job and the k+1th job
        bid_increments = top_values[:,:-1] - top_values[:,-1:]  + eps
        assert bid_increments.size() == (num_workers, jobs_per_worker)
        bids.zero_()
        bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments)

        if counter < 100 and index is not None:
            # If we were successful on the last round, put in a minimal bid to retain
            # the job only if noone else bids. After N iterations, keep it anyway.
            bids.view(-1)[index] = eps
            # 
        if counter > 1000:
            bids.view(-1)[jobs_without_bidder] = eps
        # Find jobs that was a top choice for some worker
        jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
        jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)

        # Find the highest bidding worker per job
        high_bids, high_bidders = bids[:, jobs_with_bidder].max(dim=0)
        if high_bidders.size(0) == num_jobs:
            # All jobs were bid for
            break
        
        # Make popular items more expensive
        cost[:, jobs_with_bidder] += high_bids
        value = worker_and_job_to_score - cost

        # # Hack to make sure that this item will be in the winning worker's top-k next time
        index = (high_bidders * num_jobs) + jobs_with_bidder
        value.view(-1)[index] = worker_and_job_to_score.view(-1)[index]
        counter += 1
    

    if return_token_to_worker:
        return high_bidders
    _, sorting = torch.sort(high_bidders)
    assignment = jobs_with_bidder[sorting]
    assert len(assignment.unique()) == num_jobs

    return assignment.view(-1)



def batchify(a, n=2):
    for i in np.array_split(a, n, axis=0):
        yield i

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.shape[0] - 1))
    return t.kthvalue(k, dim=0).values
    

class KMeans(object):
    def __init__(self, n_clusters=None, cluster_centers=None, device = torch.device('cpu'), balanced=False):
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.device = device
        self.balanced = balanced
    
    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['n_clusters'], saved['cluster_centers'], torch.device('cpu'), saved['balanced'])
    
    def save(self, path_to_file):
        with open(path_to_file, 'wb+') as f :
            pickle.dump(self.__dict__, f)

    def initialize(self, X):
        """
        initialize cluster centers
        :param X: (torch.tensor) matrix
        :param n_clusters: (int) number of clusters
        :return: (np.array) initial state
        """
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state
    
    def fit(
            self,
            X,
            distance='euclidean',
            tol=1e-3,
            tqdm_flag=True,
            iter_limit=0,
            gamma_for_soft_dtw=0.001,
            online=False,
            iter_k=None
    ):
        """
        perform kmeans
        :param X: (torch.tensor) matrix
        :param n_clusters: (int) number of clusters
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param tol: (float) threshold [default: 0.0001]
        :param device: (torch.device) device [default: cpu]
        :param tqdm_flag: Allows to turn logs on and off
        :param iter_limit: hard limit for max number of iterations
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
        """
        if tqdm_flag:
            print(f'running k-means on {self.device}..')

        if distance == 'euclidean':
            pairwise_distance_function = partial(pairwise_distance, device=self.device, tqdm_flag=tqdm_flag)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=self.device)
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=self.device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        
        # transfer to device
        X = X.to(self.device)

        # initialize
        if not online or (online and iter_k == 0):  # ToDo: make this less annoyingly weird
            self.cluster_centers = self.initialize(X)
            # initial_state = self.cluster_centers
        # else:
            # if tqdm_flag:
            #     print('resuming')
            # find data point closest to the initial cluster center
            # initial_state = self.cluster_centers
            # dis = pairwise_distance_function(X, initial_state)
            
            
            # # choice_points = torch.argmin(dis, dim=0)
            # initial_state = X[choice_points]
            # initial_state = initial_state.to(self.device)
            

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        # cluster = torch.arange(n_clusters).repeat_interleave(X.shape[0] // n_clusters).to(device)
        done=False
        while True:
            if self.balanced:
                distance_matrix = pairwise_distance_function(X, self.cluster_centers)
                # far_away_points = torch.where(distance_matrix > percentile(distance_matrix, 90))[0]
                # close_points = torch.where(distance_matrix < percentile(distance_matrix, 90))[0]
                # cluster_assignments_1 = torch.argmin(distance_matrix[close_points, :], dim=1)
                cluster_assignments = auction_lap(-distance_matrix)
                
                # SCIPY LINEAR ASSIGNMENT SOLVER
                # cluster_assignments = linear_sum_assignment(-distance_matrix.cpu().numpy(), maximize=True)[1] // (X.shape[0] // self.n_clusters)   
                # cluster_assignments = torch.IntTensor(cluster_assignments).cuda()

            else:
                dis = pairwise_distance_function(X, self.cluster_centers)
                cluster_assignments = torch.argmin(dis, dim=1)
            
            initial_state_pre = self.cluster_centers.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(cluster_assignments == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                
                self.cluster_centers[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
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
        
        # cluster_assignments = auction_lap(-dis)

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
            gamma_for_soft_dtw=0.001,
            tqdm_flag=False,
            return_distances=False
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

        if distance == 'euclidean':
            pairwise_distance_function = partial(pairwise_distance, device=self.device, tqdm_flag=tqdm_flag)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=self.device)
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=self.device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()
        # transfer to device
        balanced = False
        X = X.to(self.device)
        if balanced:
            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            cluster_assignments = auction_lap(-distance_matrix)
            # cluster_assignments = linear_sum_assignment(-distance_matrix.cpu().numpy(), maximize=True)[1] // (X.shape[0] // self.n_clusters) 
            # cluster_assignments = torch.IntTensor(cluster_assignments).cuda()
        else:
            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            cluster_assignments = torch.argmin(distance_matrix, dim=1 if len(distance_matrix.shape) > 1 else 0)
            if len(distance_matrix.shape) == 1:
                cluster_assignments = cluster_assignments.unsqueeze(0)
        if return_distances:
            return cluster_assignments.cpu(),distance_matrix
        else:
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
