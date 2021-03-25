#Copyright (c) 2020 Ocado. All Rights Reserved.

import numpy as np

from amrrt.vptree import DynamicVPTree


class Tree:
    """
    Tree representation of the state space used for path planning
    """
    def __init__(self, root, space, assisting_dist_fn = lambda a, b: np.linalg.norm(a.pos - b.pos)):
        """
        Creates a tree representation as a set of nodes and a mapping from nodes to children
        Also creates two vp-trees for Euclidean and assisting metrics, used for nearest neighbour queries

        :param root: State representing the root of the tree (is added to the tree as the initial node)
        :param space: State space the tree is built within
        :param assisting_dist_fn: Distance function used for the assisting metric
        """
        self.root = root
        self.nodes = set()
        self.edges = {}
        self.back_edges = {}
        self.node_amount = 0
        self.space = space
        self.euclidean_vpt = DynamicVPTree(lambda a, b: np.linalg.norm(a.pos - b.pos))
        self.assisting_vpt = DynamicVPTree(assisting_dist_fn)
        self.add_node(root)

    def add_node(self, new, parent=None):
        """
        Add new node to the tree as a child of parent, if it is not already present

        :param new: State to be added to the tree
        :param parent: Parent node of new, default None for when new is the root
        """
        if new not in self.nodes:
            self.euclidean_vpt.insert(new)
            self.assisting_vpt.insert(new)
            self.node_amount += 1
            self.nodes.add(new)
            self.edges[new] = set()
            if parent is not None : self.edges[parent].add(new)
            self.back_edges[new] = parent

    def update_edge(self, new, child):
        """
        Update the parent of child to be new, replacing the mapping from child's old parent to child with new to child

        :param new: New parent node of child
        :param child: Child whose parent is to be updated
        """
        if child != self.root:
            self.edges[self.parent(child)].remove(child)
        self.edges[new].add(child)
        self.back_edges[child] = new

    def path(self, node):
        """
        Return path from the root to the given node
        """
        current = node
        nodes = [current]
        while current != self.root:
            current = self.parent(current)
            nodes.insert(0, current)
        return nodes

    def set_root(self, new):
        """
        Set root as new by reversing the direction of edges from the old root to new
        """
        reverse_path = self.path(new)
        for i in range(1, len(reverse_path)):
            self.update_edge(reverse_path[i], reverse_path[i-1])
        if new != self.root:
            self.edges[self.parent(new)].remove(new)
        self.back_edges[new] = None
        self.root = new

    def nearest(self, state):
        """
        Return node nearest to the given state by assisting metric
        """
        return self.assisting_vpt.nearest(state)

    def euclidean_nearest(self, state):
        """
        Return node nearest to the given state by Euclidean metric
        """
        return self.euclidean_vpt.nearest(state)

    def neighbourhood(self, state, radius):
        """
        Return all nodes within the given radius of the given state
        """
        return self.euclidean_vpt.neighbourhood(state, radius)

    def parent(self, node):
        """
        Return the parent of the given node
        """
        return self.back_edges[node]
