#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import GridEnviroment, StateSpace
from amrrt.grid_graph import GridGraph
from amrrt.diffusion_map import DiffusionMap


def test_eigenpairs_simple():
    space = StateSpace(GridEnviroment(np.array([[0,0,0], [0,1,0]])))
    grid_graph = GridGraph(space, max_edge_length=1)
    diffusion_map = DiffusionMap(space, grid_graph=grid_graph, dimensions=3)
    P, L = diffusion_map._eigenpairs(diffusion_map._generate_similarity_matrix(), 0.0000001)
    assert (abs(L[0,0] - 0.1446) < 0.001 and abs(L[1,1] - 0.5667) < 0.001 and abs(L[2,2] - 0.9221) < 0.001 and abs(P[0,0] - 0.5708) < 0.001)


def test_eigenvalues_increasing_lt_one(maze_environment_info):
    cur = 1
    diffusion_map = maze_environment_info["diffusion_map"]
    _, L = diffusion_map._eigenpairs(diffusion_map._generate_similarity_matrix(), 0.00001)
    for i in range(9, 0, -1):
        assert (L[i,i] <= cur)
        cur = L[i,i]
