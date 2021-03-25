#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.metrics import DiffusionMetric
from amrrt.planners import RTRRTPlanner, AMRRTPlanner


def test_rewire_root_rtrrt(empty_environment_info):
    planner = RTRRTPlanner(empty_environment_info["space"], np.array([10,10]),
                           assisting_metric=DiffusionMetric(empty_environment_info["diffusion_map"]))
    planner.expand()
    costs = {}
    for node in planner.tree.nodes:
        costs[node] = planner.cost(node)
    planner.rewire_root()
    for node in planner.tree.nodes:
        assert planner.cost(node) <= costs[node]


def test_rewire_root_amrrt(empty_environment_info):
    planner = AMRRTPlanner(empty_environment_info["space"], np.array([10,10]),
                           assisting_metric=DiffusionMetric(empty_environment_info["diffusion_map"]))
    planner.expand()
    costs = {}
    for node in planner.tree.nodes:
        costs[node] = planner.cost(node)
    planner.rewire_root()
    for node in planner.tree.nodes:
        assert planner.cost(node) <= costs[node]


def test_rewire_rand_rtrrt(empty_environment_info):
    planner = RTRRTPlanner(empty_environment_info["space"], np.array([10,10]),
                           assisting_metric=DiffusionMetric(empty_environment_info["diffusion_map"]))
    planner.expand()
    costs = {}
    for node in planner.tree.nodes:
        costs[node] = planner.cost(node)
    planner.rewire_rand()
    for node in planner.tree.nodes:
        assert planner.cost(node) <= costs[node]


def test_rewire_goal_amrrt(empty_environment_info):
    planner = AMRRTPlanner(empty_environment_info["space"], np.array([10,10]),
                           assisting_metric=DiffusionMetric(empty_environment_info["diffusion_map"]))
    planner.set_goal(np.array([20,20]))
    while planner.goal not in planner.tree.nodes:
        planner.expand()
    costs = {}
    for node in planner.tree.nodes:
        costs[node] = planner.cost(node)
    planner.rewire_goal()
    for node in planner.tree.nodes:
        assert planner.cost(node) <= costs[node]
