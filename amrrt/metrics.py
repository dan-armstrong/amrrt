#Copyright (c) 2020 Ocado. All Rights Reserved.

import time
import numpy as np
from scipy import sparse


class EuclideanMetric:
    def distance(self, a, b):
        """
        Returns euclidean distance between states a and b
        """
        return np.linalg.norm(a.pos - b.pos)

    def steer(self, space, start, end, max_step, t_steer=0):
        """
        Return state projected along line from start to end as far as possible so that it remains obs-free, at max distance max_step
        """
        if space.free_path(start, end):
            p = min(1, max_step/self.distance(start, end))
            return space.create_state((1-p)*start.pos + p*end.pos)
        intersection_pos = space.env.closest_intersection(start.pos, end.pos)
        intersection = space.create_state(intersection_pos)
        p = min(0.99, max_step/self.distance(start, intersection))
        best = space.create_state((1-p)*start.pos + p*intersection.pos)
        if space.free_path(start, best):
            return best
        return None


class DiffusionMetric:
    def __init__(self, diffusion_map):
        self.diffusion_map = diffusion_map

    def distance(self, a, b):
        """
        Returns the diffusion distance between two states a and b
        """
        return np.linalg.norm(self.diffusion_map.diffusion_pos(a.pos) - self.diffusion_map.diffusion_pos(b.pos))

    def steer(self, space, start, end, max_step, t_steer):
        """
        Returns state accessible from start that is closer to end by diffusion distance via stochastic hill descent

        :param start: State to steer from
        :param end: State to steer towards
        :param max_step: Maximum distance the returned point can be start
        :param t_steer: Amount of time allocated to steering
        """
        best = None
        best_dist = np.inf
        radius = min(max_step, np.linalg.norm(start.pos - end.pos))
        start_time = time.time()
        while time.time() - start_time < t_steer:
            rand = space.choose_state_ellipse(start.pos, start.pos, radius, radius)
            rand_dist = self.distance(rand, end)
            if space.free_path(start, rand) and rand_dist < best_dist:
                best = rand
                best_dist = rand_dist
        return best


class GeodesicMetric:
    def __init__(self, grid_graph, distance_matrix=None):
        """
        :param grid_graph: Grid graph for which geodesic distances can be calculated
        :param distance_matrix: Matrix of geodesic distances, default None to generate from scratch
        """
        self.grid_graph = grid_graph
        self.distance_matrix = distance_matrix if distance_matrix is not None else sparse.csgraph.floyd_warshall(self.grid_graph.adjacency_matrix)

    def distance(self, a, b):
        """
        Returns geodesic distance between states a and b
        """
        a_index = self.grid_graph.grid_index(a.pos)
        b_index = self.grid_graph.grid_index(b.pos)
        if a_index is None or b_index is None:
            return np.inf
        return self.distance_matrix[a_index,b_index]

    def steer(self, space, start, end, max_step, t_steer):
        """
        Returns state accessible from start that is closer to end by geodesic distance via stochastic hill descent

        :param start: State to steer from
        :param end: State to steer towards
        :param max_step: Maximum distance the returned point can be start
        :param t_steer: Amount of time allocated to steering
        """
        best = None
        best_dist = np.inf
        radius = min(max_step, np.linalg.norm(start.pos - end.pos))
        start_time = time.time()
        while time.time() - start_time < t_steer:
            rand = space.choose_state_ellipse(start.pos, start.pos, radius, radius)
            rand_dist = self.distance(rand, end)
            if space.free_path(start, rand) and rand_dist < best_dist:
                best = rand
                best_dist = rand_dist
        return best
