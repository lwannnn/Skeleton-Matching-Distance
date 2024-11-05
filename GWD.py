import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F
import torch_geometric.utils as PyG_utils
import numpy as np
import skimage
from skimage.morphology import skeletonize
import math

class GraphWassersteinDistance(nn.Module):
    def __init__(self,eps=0.01, thresh=0.1, max_iter=100,reduction='none'):
        super(GraphWassersteinDistance,self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh
        self.mask_matrix = None

    def forward(self,pred,target):
        """
        :param pred: skeleton numpy,size (x,y,z)
        :param target: same with 1.
        1.Build Graph Where Pred>0.5 TODO:Save mask graph in Hard Drive
        2.Calculate Cost Matrix basing on Euclidean distance
        3.Calculate Mask Matrix Basing on k-nn strategy
        4.Calculate Optimal Plan using Sinkhorn Iteration Algorithm
        :return:
        """
        assert pred.size() == target.size()
        self.normalized_num =  math.sqrt((target.size(0) - 1) ** 2 + (target.size(1) - 1) ** 2 + (target.size(2) - 1) ** 2)

        source_graph = self.build_graph_from_torch(pred).cuda()
        mask_graph = self.build_graph_from_torch(target, is_gt=True).cuda()

        cost_matrix = self.get_cost_matrix(source_graph, mask_graph)#（source_graph_node_num,mask_graph_node_num)
        mask_matrix = self.get_mask_matrix(cost_matrix).to_sparse()#二值的（source_graph_node_num,mask_graph_node_num)
        # mu = self.marginal_prob_unform(source_graph)#(source_graph_node_num,1)
        # nu = self.marginal_prob_unform(mask_graph)#(mask_graph_node_num)
        mu = self.marginal_prob_uniform(source_graph)
        nu = self.marginal_prob_uniform(mask_graph)
        gwd = self.get_dist(mu,nu,cost_matrix,mask_matrix)

        return gwd.item()

    def get_dist(self,mu,nu,C,A,mask=None):
        """
        :param mu: 预测图的分布
        :param nu: GT的分布
        :param C: cost 矩阵
        :param A: mask矩阵（k近邻，GTOT里的邻接矩阵）
        :param mask: （未知）
        :return:
        """
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        if A is not None:
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C
        actual_nits = 0    # To check if algorithm terminates because of threshold,or max iterations reached
        thresh = self.thresh  # Stopping criterion
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            if mask is None:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            else:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                u = mask * u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
                v = mask * v

            err = (u - u1).abs().sum(-1).max()# err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        U, V = u, v
        pi = self.exp_M(C, U, V, A=A)
        cost = torch.sum(pi * C, dim=(-2, -1))
        if torch.isnan(cost.sum()):
            print(pi)
            raise
        return cost


    def build_graph_from_torch(self,source,is_gt=False,threshold=0.5,max_num=3000):
        nonzero_indices= []
        if not is_gt:
            nonzero_indices = ( (source >= threshold)*1).nonzero(as_tuple=False)
        else:
            nonzero_indices = source.nonzero(as_tuple=False)

        if len(nonzero_indices) > max_num:
            # Get the indices of the top 5000 points based on their values in the source tensor
            sorted_indices = torch.argsort(
                source[nonzero_indices[:, 0], nonzero_indices[:, 1],  nonzero_indices[:, 2]],descending=True)
            # Use the first 5000 indices to filter the nonzero_indices
            top_indices = sorted_indices[:max_num]
            nonzero_indices = nonzero_indices[top_indices]
        if len(nonzero_indices) == 0:
            # If there are no non-zero points, create a default graph with a single node at (0, 0, 0)
            node_features = torch.tensor([1], dtype=torch.float).view(-1, 1)
            pos = torch.tensor([[0, 0, 0]], dtype=torch.float)
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1,1]], dtype=torch.float)  # Edge feature (for the single edge)
            degrees = torch.tensor([1], dtype=torch.float)
            data = Data(x=node_features, pos=pos, edge_index=edge_index, edge_attr=edge_attr, degrees=degrees)
            # print(data)
            return data

        node_features = source[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(-1, 1)
        pos = nonzero_indices.clone().detach().view(-1, 3)

        pairwise_distances = torch.cdist(pos.float(), pos.float(), p=2)# Calculate pairwise distances
        edge_mask = pairwise_distances <= 2**0.5# Create mask based on threshold
        edge_indices = edge_mask.nonzero(as_tuple=False)# Get indices of non-zero elements in the mask
        edge_index = torch.stack([edge_indices[:, 0], edge_indices[:, 1]], dim=0)# Create edge_index tensor
        edge_attr = torch.cat([node_features[edge_index[0]], node_features[edge_index[1]]], dim=1) # Extract edge features based on indices
        num_nodes = len(node_features)
        degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        data = Data(x=node_features, pos=pos, edge_index=edge_index, edge_attr=edge_attr,degrees=degrees)
        # data = Data(x=node_features, pos=pos)
        # print(data)
        return data

    def get_cost_matrix(self, source_graph, mask_graph):
        source_positions = source_graph.pos.float()
        mask_positions = mask_graph.pos.float()
        num_nodes_source, num_nodes_mask = len(source_positions), len(mask_positions)

        distances = torch.cdist(source_positions, mask_positions, p=2)
        normalized_distances = torch.log(distances+1) / torch.log(torch.tensor(self.normalized_num+1))
        # normalized_cost_matrix = torch.nn.functional.normalize(distances, p=2, dim=1)
        return normalized_distances

    def get_mask_matrix(self,cost_matrix,num_neighbors = 24):
        num_neighbors = min(num_neighbors, cost_matrix.size(1))
        top_values, top_indices = torch.topk(cost_matrix, num_neighbors, largest=False, dim=1)
        mask_matrix = torch.zeros_like(cost_matrix)
        mask_matrix.scatter_(1, top_indices, 1)# 将每行的前6个最小值的位置设为1
        # print(mask_matrix)

        num_neighbors = min(num_neighbors, cost_matrix.size(0))  # Number of rows
        top_values, top_indices = torch.topk(cost_matrix, num_neighbors, largest=True, dim=0)
        mask_matrix_T = torch.zeros_like(cost_matrix)
        mask_matrix_T.scatter_(0, top_indices, 1)  # 将每列的前6个最大值的位置设为1
        return ((mask_matrix+mask_matrix_T)>0)*1

    def marginal_prob_feature_based(self, source_graph):
        node_features = source_graph.x
        total_feature_sum = torch.sum(node_features, dim=0)
        mu = node_features / total_feature_sum
        mu_sum = torch.sum(mu, dim=0, keepdim=True)  # 在每行上进行求和并保持维度
        mu = mu / mu_sum  # 将mu中的每个元素除以对应行的和，确保mu加和=1
        return mu.view(1, -1)#转置

    def marginal_prob_degree_based(self, source_graph):
        node_degrees = source_graph.degrees
        mu = node_degrees.float()  # Consider degrees as the initial marginal probabilities
        mu_sum = torch.sum(mu)  # Compute the sum of degrees
        mu = mu / mu_sum  # Normalize the degrees to obtain probabilities
        # Convert the probabilities into a row vector and return the transposed tensor
        return mu.view(1, -1)  # Assuming mu is a 1D tensor, view it as a row vector

    def marginal_prob_uniform(self,source_graph):
        node_num = len(source_graph.pos)
        mu = torch.ones(node_num) / node_num
        return mu.cuda()

    def M(self, C, u, v, A=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return S

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-a).to(torch.bool),value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-A).to(torch.bool),value=0)

            return S
        elif self.mask_matrix is not None:
            return self.mask_matrix * torch.exp(self.M(C, u, v))
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1, mask=None):
        s = torch.sum(input_tensor, dim=dim)
        out = torch.log(1e-8 + s)
        if torch.isnan(out.sum()):
            raise
        if mask is not None:
            out = mask * out
        return out

if __name__=='__main__':
    loss=GraphWassersteinDistance()
    a = torch.tensor([[[1, 1],
         [1, 1]],

        [[2,  2],
         [ 3,  3]]])
    b = a
    print(a)
    gwd = loss(a.cuda(),b.cuda())
    print(gwd)


