#Copyright (c) 2020 Ocado. All Rights Reserved.

import vptree, itertools
import numpy as np


class _ExtendedVPTree(vptree.VPTree):
    """
    VPTree class extended to include the list of points within the tree
    """
    def __init__(self, points, dist_fn):
        """
        :param points: List of points to add to the vp-tree
        :param dist_fn: Metric distance function
        """
        super().__init__(points, dist_fn)
        self.points = points
        self.size = len(points)

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """
        Override parent method to use <= when finding nearest neighbours to ensure a neighbour is returned even at infinite/nan distance
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be strictly positive integer')
        neighbors = vptree._AutoSortingList(max_size=n_neighbors)
        nodes_to_visit = [(self, 0)]
        furthest_d = np.inf
        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > furthest_d:
                continue
            d = self.dist_fn(query, node.vp)
            if d <= furthest_d:     #Replaced < with <=
                neighbors.append((d, node.vp))
                furthest_d, _ = neighbors[-1]
            if node._is_leaf():
                continue
            if node.left_min <= d <= node.left_max:
                nodes_to_visit.insert(0, (node.left, 0))
            elif node.left_min - furthest_d <= d <= node.left_max + furthest_d:
                nodes_to_visit.append((node.left,
                                       node.left_min - d if d < node.left_min
                                       else d - node.left_max))
            if node.right_min <= d <= node.right_max:
                nodes_to_visit.insert(0, (node.right, 0))
            elif node.right_min - furthest_d <= d <= node.right_max + furthest_d:
                nodes_to_visit.append((node.right,
                                       node.right_min - d if d < node.right_min
                                       else d - node.right_max))
        if len(neighbors) == 0:
            neighbors = [(np.nan, point) for point in self.points[:n_neighbors]] #Return any point(s) if query contains np.nan
        return list(neighbors)


class DynamicVPTree:
    """
    Dynamic vp-tree implemented using index folding
    """
    def __init__(self, dist_fn, min_tree_size=4):
        """
        :param dist_fn: Metric distance function used for vp-trees
        :param min_tree_size: Minimum number of nodes to form a tree (extra nodes are stored in a pool until the number is reached)
        """
        self.dist_fn = dist_fn
        self.trees = []
        self.pool = []
        self.min_tree_size = min_tree_size

    def insert(self, item):
        """
        Insert item into dynamic vp tree by first adding to pool, and then building a tree from the pool if min size reached
        Then merge trees of equal sizes so that there are at most log(log (n)) trees, with the largest tree having roughly n/2 nodes
        """
        self.pool.append(item)
        if len(self.pool) == self.min_tree_size:
            self.trees.append(_ExtendedVPTree(self.pool, self.dist_fn))
            self.pool = []
        while len(self.trees) > 1 and self.trees[-1].size == self.trees[-2].size:
            a = self.trees.pop()
            b = self.trees.pop()
            self.trees.append(_ExtendedVPTree(a.points + b.points, self.dist_fn))

    def nearest(self, query):
        """
        Return node nearest to query by finding nearest node in each tree and returning the global minimum (including nodes in pool)
        """
        nearest_trees = list(map(lambda t: t.get_nearest_neighbor(query), self.trees))
        distances_pool = list(zip(map(lambda x: self.dist_fn(x, query), self.pool), self.pool))
        best = None
        best_cost = np.inf
        for cost, near in nearest_trees + distances_pool:
            if cost <= best_cost:
                best = near
                best_cost = cost
        return best

    def neighbourhood(self, query, radius):
        """
        Return all nodes within distance radius of the given query, by collating neighbourhoods for each internal tree (and pool)
        """
        tree_neighbourhood = lambda tree: list(map(lambda x: x[1], tree.get_all_in_range(query, radius)))
        neighbourhood_trees = list(itertools.chain.from_iterable(map(tree_neighbourhood, self.trees)))
        return neighbourhood_trees + list(filter(lambda x: self.dist_fn(x, query) < radius, self.pool))
