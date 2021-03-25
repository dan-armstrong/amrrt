#Copyright (c) 2020 Ocado. All Rights Reserved.

import os
from scipy import sparse, stats, optimize
import numpy as np

from amrrt.grid_graph import GridGraph


class DiffusionMap:
    def __init__(self, space, grid_graph=None, tolerance=0.00001, dimensions=10, sample_path_amount=25, saved=None):
        """
        Generate diffusion matrix or load from saved data

        :param space: State space the diffusion map is created from
        :param grid_graph: Graph representation of the space, default None to generate from scratch
        :param tolerance: Tolerance to calculate eigenvalues within (the smaller the more accurate but more time consuming to find)
        :param dimensions: Number of dimensions left in diffusion map after dimensionality reduction
        :param sample_path_amount: Number of sample paths generated to test for optimal t value
        :param saved: Dictionary of saved information, default None to generate from scratch
        """
        if saved is None:
            self.grid_graph = grid_graph if grid_graph is not None else GridGraph(space)
            self.dimensions = dimensions
            P, L = self._eigenpairs(self._generate_similarity_matrix(), tolerance)
            t = self._generate_time_scale(P, L, self._generate_sample_paths(sample_path_amount))
            self.diffusion_matrix = P @ sparse.diags([P.shape[0]*L.diagonal()**t], [0], format="csr")
        else:
            self.grid_graph = grid_graph
            self.diffusion_matrix = saved["diffusion_matrix"]
            self.dimensions = self.diffusion_matrix.shape[1]

    @classmethod
    def from_saved(cls, space, map_name):
        """
        Load diffusion map from file

        :param space: State space the saved diffusion map was created within
        :param map_name: File name of saved diffusion map (should be a folder containing the relevant files)
        """
        grid_graph = GridGraph.from_saved(space, map_name)
        saved = {"diffusion_matrix": np.load(os.path.join(map_name, "diffusion_matrix.npy"))}
        return cls(space, grid_graph=grid_graph, saved=saved)

    def save(self, map_name):
        """
        Save diffusion map to file

        :param map_name: Folder name to save the diffusion map as
        """
        self.grid_graph.save(map_name)
        np.save(os.path.join(map_name, "diffusion_matrix.npy"), self.diffusion_matrix)

    def _generate_similarity_matrix(self):
        """
        Generate similarity matrix from adjacency matrix, using kernel described in Chen's paper on Diffusion Maps
        """
        rows, cols, data = sparse.find(self.grid_graph.adjacency_matrix)
        return sparse.csr_matrix((np.exp(-np.square(data)/(2*self.grid_graph.grid_size)), (rows, cols)), shape=(self.grid_graph.node_amount, self.grid_graph.node_amount))

    def _eigenpairs(self, A, tol):
        """
        Calculate eigenpairs of the transition matrix (generated from similarity matrix) as described in Chen's paper

        :param A: Similarity matrix
        :param tol: Tolerance to calculate eigenvalues within
        """
        row_sum = np.asarray(A.sum(axis=1)).squeeze()
        D = sparse.diags([row_sum], [0], format="csr")
        A1 = 0.5*A + 0.5*D
        DInv = sparse.diags([1/row_sum], [0], format="csr")
        A2 = DInv @ A1 @ DInv
        row_sum_2 = np.asarray(A2.sum(axis=1)).squeeze()
        D2 = sparse.diags([row_sum_2], [0], format="csr")
        D2Isqr = sparse.diags([1/np.sqrt(row_sum_2)], [0], format="csr")
        E = D2Isqr @ A2 @ D2Isqr
        eigenvalues, eigenvectors = sparse.linalg.eigsh(E, self.dimensions+1, which="LM", tol=tol)
        eigenvalues = eigenvalues[:self.dimensions]                             #REDUCE NUMBER OF DIMENSIONS
        eigenvectors = eigenvectors[:,:self.dimensions]
        P = D2Isqr @ eigenvectors
        L = sparse.diags([eigenvalues], [0], format="csr")
        return P, L

    def _generate_sample_paths(self, sample_path_amount):
        """
        Use Dijkstra to generate a set of optimal paths and their respective costs to test against

        :param sample_path_amount: Amount of points to generate sample paths between
        """
        start_positions = np.random.default_rng().choice(self.grid_graph.node_amount, size=min(sample_path_amount, self.grid_graph.node_amount), replace=False)
        end_positions = np.random.default_rng().choice(self.grid_graph.node_amount, size=min(sample_path_amount, self.grid_graph.node_amount), replace=False)
        return start_positions, end_positions, sparse.csgraph.dijkstra(self.grid_graph.adjacency_matrix, indices=start_positions)[:,end_positions]

    def _generate_time_scale(self, P, L, path_data, max_iter=1000):
        """
        Return time scale parameter that optimises the linearity of path lenghts in the diffusion map

        :param P: Eigenvectors of transition matrix
        :param L: Diagonal matrix of eigenvalues from transition matrix
        :param path_data: Set of optimal paths and their respective costs
        """
        max_t = 1000 * P.shape[0] ** 0.5
        best_t = 0
        best_score = -1
        score = lambda t : self._evaluate(P @ sparse.diags([P.shape[0]*L.diagonal()**t], [0], format="csr"), *path_data)
        for i in range(max_iter):
            t_score = score(max_t * i/max_iter)
            if t_score > best_score:
                best_t = max_t * i/max_iter
                best_score = t_score
        lower = max(0, best_t-max_t/max_iter)
        upper = best_t+max_t/max_iter
        return optimize.minimize_scalar(lambda x : -score(x), bounds=(lower, upper), method='bounded').x

    def _evaluate(self, eval_matrix, path_starts, path_ends, costs):
        """
        Return the linear correlation coefficient between diffusion distance and optimal distance by calculating PMCC

        :param eval_matrix: Diffusion space metric to evaluate
        :param path_starts: Array of path start nodes
        :param path_ends: Array of path end nodes
        :param costs: 2D array of with pos i,j the optimal distance between path_starts[i] and path_ends[j]
        """
        if np.count_nonzero(eval_matrix[:,0]) == 0:
            return -1
        estimates = []
        actual = []
        for start, cost_row in zip(path_starts, costs):
            estimates += list(np.linalg.norm(eval_matrix[start]-eval_matrix[path_ends], axis=1))
            actual += list(cost_row)
        return stats.pearsonr(actual, estimates)[0]

    def diffusion_pos(self, pos):
        """
        Map position in state space to respective position in the diffusion space
        """
        if self.grid_graph.grid_index(pos) is None:
            return np.array([np.inf] * self.dimensions)
        return self.diffusion_matrix[self.grid_graph.grid_index(pos)]
