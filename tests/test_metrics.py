#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.metrics import EuclideanMetric, DiffusionMetric


def test_distance_euclidean(maze_environment_info):
    space = maze_environment_info["space"]
    a = space.create_state(np.array([10,10]))
    b = space.create_state(np.array([20,20]))
    euclidean_metric = EuclideanMetric()
    assert (euclidean_metric.distance(a, b) == euclidean_metric.distance(b, a) == np.linalg.norm(a.pos - b.pos))


def test_distance_diffusion(maze_environment_info):
    space = maze_environment_info["space"]
    a = space.create_state(np.array([10,10]))
    b = space.create_state(np.array([20,20]))
    diffusion_metric = DiffusionMetric(maze_environment_info["diffusion_map"])
    assert diffusion_metric.distance(a, b) == diffusion_metric.distance(b, a)


def test_triangle_euclidean(maze_environment_info):
    space = maze_environment_info["space"]
    a = space.create_state(np.array([10,10]))
    b = space.create_state(np.array([20,20]))
    c = space.create_state(np.array([20,10]))
    euclidean_metric = EuclideanMetric()
    assert euclidean_metric.distance(a, b) + euclidean_metric.distance(b, c) >= euclidean_metric.distance(a, c)


def test_triangle_diffusion(maze_environment_info):
    space = maze_environment_info["space"]
    a = space.create_state(np.array([10,10]))
    b = space.create_state(np.array([20,20]))
    c = space.create_state(np.array([20,10]))
    diffusion_metric = DiffusionMetric(maze_environment_info["diffusion_map"])
    assert diffusion_metric.distance(a, b) + diffusion_metric.distance(b, c) >= diffusion_metric.distance(a, c)
