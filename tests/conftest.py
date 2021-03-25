#Copyright (c) 2020 Ocado. All Rights Reserved.

import pytest, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace
from amrrt.grid_graph import GridGraph
from amrrt.diffusion_map import DiffusionMap
from amrrt.metrics import GeodesicMetric


@pytest.fixture(scope="session", autouse=True)
def empty_environment_info():
    space = StateSpace.from_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "empty.png"))
    diffusion_map = DiffusionMap(space)
    return {"space": space, "diffusion_map": diffusion_map}


@pytest.fixture(scope="session", autouse=True)
def maze_environment_info():
    space = StateSpace.from_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "maze.png"))
    diffusion_map = DiffusionMap(space)
    return {"space": space, "diffusion_map": diffusion_map}


@pytest.fixture(scope="session", autouse=False)
def maze_distance_matrix():
    space = StateSpace.from_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "maze.png"))
    grid_graph = GridGraph(space)
    return GeodesicMetric(grid_graph).distance_matrix
