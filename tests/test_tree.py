#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.tree import Tree


def test_add(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    new = space.create_state(np.array([5,5]))
    tree = Tree(root, space)
    tree.add_node(new, root)
    assert tree.parent(new) == root


def test_add_duplicate(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    tree = Tree(root, space)
    tree.add_node(root, root)
    assert tree.parent(root) == None


def test_update(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    a = space.create_state(np.array([11,10]))
    b = space.create_state(np.array([12,10]))
    tree = Tree(root, space)
    tree.add_node(a, root)
    tree.add_node(b, root)
    tree.update_edge(b, a)
    assert tree.parent(a) == b


def test_path(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    a = space.create_state(np.array([11,10]))
    b = space.create_state(np.array([12,10]))
    tree = Tree(root, space)
    tree.add_node(a, root)
    tree.add_node(b, a)
    assert tree.path(b) == [root, a, b]


def test_root_path(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    a = space.create_state(np.array([11,10]))
    b = space.create_state(np.array([12,10]))
    tree = Tree(root, space)
    tree.add_node(a, root)
    tree.add_node(b, a)
    assert tree.path(root) == [root]


def test_set_root(empty_environment_info):
    space = empty_environment_info["space"]
    old = space.create_state(np.array([10,10]))
    new = space.create_state(np.array([11,10]))
    tree = Tree(old, space)
    tree.add_node(new, old)
    tree.set_root(new)
    assert tree.path(old) == [new, old]


def test_nearest(empty_environment_info):
    space = empty_environment_info["space"]
    root = space.create_state(np.array([10,10]))
    seeds = [np.random.rand() for _ in range(50)]
    nodes = [space.create_state(space.bounds[:,0]*seed + space.bounds[:,1]*(1-seed)) for seed in seeds]
    tree = Tree(root, space)
    for node in nodes : tree.add_node(node)
    search = space.create_state(np.array([15,15]))
    assert np.linalg.norm(search.pos - tree.nearest(search).pos) == min([np.linalg.norm(search.pos - node.pos) for node in nodes])
