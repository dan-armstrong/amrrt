#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.metrics import GeodesicMetric


def test_distance_geodesic(maze_environment_info, maze_distance_matrix):
    space = maze_environment_info["space"]
    shortest_paths = [74.61747343047412, 84.61575781480384, 64.40995862204083, 98.07522464033143, 127.19815498031583, 95.70905438785839]
    waypoints = [[63.0, 51.0], [7.2, 6.6], [28.0, 81.0], [35.0, 37.0], [93.0, 91.0], [89.2,  7.0], [16.2, 37.2]]
    geodesic_metric = GeodesicMetric(maze_environment_info["space"], distance_matrix=maze_distance_matrix)
    for i in range(6):
        a = space.create_state(np.array(waypoints[i]))
        b = space.create_state(np.array(waypoints[i+1]))
        assert geodesic_metric.distance(a, b) == geodesic_metric.distance(b, a) >= np.linalg.norm(a.pos - b.pos)
        assert abs(geodesic_metric.distance(a, b) - shortest_paths[i]) / shortest_paths[i] < 0.05


def test_triangle_geodesic(maze_environment_info, maze_distance_matrix):
    space = maze_environment_info["space"]
    a = space.create_state(np.array([10,10]))
    b = space.create_state(np.array([20,20]))
    c = space.create_state(np.array([20,10]))
    geodesic_metric = GeodesicMetric(space, distance_matrix=maze_distance_matrix)
    assert geodesic_metric.distance(a, b) + geodesic_metric.distance(b, c) >= geodesic_metric.distance(a, c)
