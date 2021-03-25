#Copyright (c) 2020 Ocado. All Rights Reserved.

import os
import numpy as np
from scipy import sparse


class GridGraph:
    """
    Encapsulates graphical represtation of the state space using nodes in a grid layout
    """
    def __init__(self, space, grid_size=1, max_edge_length=2.5, saved=None):
        """
        :param space: State space the graph is created from
        :param grid_size: Spacing between nodes in grid
        :param max_edge_length: Maximum length of edge
        :param saved: Dictionary of saved information, default None to generate from scratch
        """
        self.space = space
        if saved is None:
            self.grid_size = grid_size
            self.max_edge_length = max_edge_length
            self.lattice = np.array([[[x,y] for x in self._frange(self.space.bounds[0][0]+self.grid_size/2, self.space.bounds[0][1], self.grid_size)]
                                            for y in self._frange(self.space.bounds[1][0]+self.grid_size/2, self.space.bounds[1][1], self.grid_size)])
            self.pos_to_index, self.index_to_pos, self.neighbours = self._reduce_components(self._neighbour_positions())
            self.node_amount = len(self.index_to_pos)
            self.adjacency_matrix = self._generate_adjacency_matrix()
        else:
            self.grid_size = saved["grid_size"]
            self.max_edge_length = saved["max_edge_length"]
            self.lattice = saved["lattice"]
            self.pos_to_index = saved["pos_to_index"]
            self.index_to_pos = saved["index_to_pos"]
            self.neighbours = saved["neighbours"]
            self.node_amount = len(self.index_to_pos)
            self.adjacency_matrix = saved["adjacency_matrix"]

    @classmethod
    def from_saved(cls, space, graph_name):
        """
        Load diffusion map from file

        :param space: State space the saved graph was created within
        :param graph_name: File name of saved graph (should be a folder containing the relevant files)
        """
        saved = {}
        with open(os.path.join(graph_name, "metadata.txt")) as f:
            metadata = f.read().split(",")
            saved["grid_size"] = float(metadata[0])
            saved["max_edge_length"] = float(metadata[1])
        saved["pos_to_index"] = {}
        with open(os.path.join(graph_name, "pos_to_index.txt")) as f:
            for line in f:
                if line != "":
                    data = line.split("-")
                    saved["pos_to_index"][tuple(map(float, data[0].split(",")))] = int(data[1])
        saved["neighbours"] = []
        with open(os.path.join(graph_name, "neighbours.txt")) as f:
            for line in f:
                if line != "":
                    saved["neighbours"].append(list(map(int, line.split(","))))
        saved["index_to_pos"] = np.load(os.path.join(graph_name, "index_to_pos.npy"))
        saved["lattice"] = np.load(os.path.join(graph_name, "lattice.npy"))
        saved["adjacency_matrix"] = sparse.load_npz(os.path.join(graph_name, "adjacency_matrix.npz"))
        return cls(space, saved=saved)

    def save(self, graph_name):
        """
        Save graph to file

        :param map_name: Folder name to save the graph as
        """
        if not os.path.exists(graph_name):
            os.mkdir(graph_name)
        metadata = str(self.grid_size) + "," + str(self.max_edge_length)
        with open(os.path.join(graph_name, "metadata.txt"), 'w') as f:
            f.write(metadata)
        with open(os.path.join(graph_name, "pos_to_index.txt"), 'w') as f:
            for pos, index in self.pos_to_index.items():
                f.write(",".join(map(str, pos)) + "-" + str(index) + "\n")
        with open(os.path.join(graph_name, "neighbours.txt"), 'w') as f:
            for nbr_list in self.neighbours:
                f.write(",".join(map(str, nbr_list)) + "\n")
        np.save(os.path.join(graph_name, "index_to_pos.npy"), self.index_to_pos)
        np.save(os.path.join(graph_name, "lattice.npy"), self.lattice)
        sparse.save_npz(os.path.join(graph_name, "adjacency_matrix.npz"), self.adjacency_matrix)

    def _neighbour_positions(self):
        """
        Returns dictionary mapping node positions to positions of their neighbours
        """
        neighbours = {}
        for row in range(self.lattice.shape[0]):
            for col in range(self.lattice.shape[1]):
                if self.space.free_position(self.lattice[row,col]):
                    neighbours[tuple(self.lattice[row,col])] = self._node_neighbours(row, col)
            print("Neighbours " + str(int(row/self.lattice.shape[0]*100)) + "% complete" + " "*5, end="\r")
        print("Neighbours completed" + " "*10)
        return neighbours

    def _node_data(self, neighbour_positions):
        """
        Indexes nodes from given neighbour positions and creates mappings from indexes to neighbours, positions to indexes & vice-versa

        :param neighbour_positions: Dictionary mapping node positions to positions of their neighbours
        """
        pos_to_index = {}
        index_to_pos = np.zeros((len(neighbour_positions), 2))
        for i, (pos, neighbours) in enumerate(neighbour_positions.items()):
            pos_to_index[tuple(pos)] = i
            index_to_pos[i] = pos
            print("Nodes " + str(int(i/len(neighbour_positions)*100)) + "% complete" + " "*5, end="\r")
        print("Nodes completed" + " "*10)
        neighbours = []
        for pos in index_to_pos:
            neighbours.append([pos_to_index[tuple(npos)] for npos in neighbour_positions[tuple(pos)]])
        return pos_to_index, index_to_pos, neighbours

    def _node_neighbours(self, row, col):
        """
        Returns actual positions of neighbours for the node at a given lattice location

        :param row: Row position of the node in the lattice
        :param col: Column position of the node in the lattice
        """
        neighbours = set()
        pos = self.lattice[row, col]
        lattice_radius = int(self.max_edge_length/self.grid_size)
        for nr in range(max(0, row-lattice_radius), min(self.lattice.shape[0], row+lattice_radius+1)):
            for nc in range(max(0, col-lattice_radius), min(self.lattice.shape[1], col+lattice_radius+1)):
                npos = self.lattice[nr, nc]
                if np.linalg.norm(pos-npos) <= self.max_edge_length and self.space.free_path(self.space.create_state(pos), self.space.create_state(npos)):
                    neighbours.add((nr,nc))
        return [self.lattice[row,col] for row, col in neighbours]

    def _reduce_components(self, all_neighbour_positions):
        """
        Returns the node-neighbour mapping representing the largest connected component of the graph represented by the given node-neighbour mapping

        :param all_neighbour_positions: Graph represented as a dictionary mapping node positions to positions of their neighbours
        """
        _, index_to_pos, neighbours = self._node_data(all_neighbour_positions)
        components = self._components(neighbours)
        sorted_indexes = sorted([(-len(components[i]), i) for i in range(len(components))])[:1]
        reduced_neighbour_positions = {}
        for _, i in sorted_indexes:
            for node in components[i]:
                reduced_neighbour_positions[tuple(index_to_pos[node])] = [index_to_pos[nbr] for nbr in neighbours[node]]
        return self._node_data(reduced_neighbour_positions)

    def _components(self, neighbours):
        """
        Returns list of connected components in graph, where each connected component is a list of the nodes within it

        :param neighbours: Dictionary mapping nodes to neighbours
        """
        seen = [False] * len(neighbours)
        components = []
        stack = []
        for node in range(len(neighbours)):
            if not seen[node]:
                component = []
                stack.append(node)
                seen[node] = True
                while stack != []:
                    cur = stack.pop()
                    component.append(cur)
                    for neighbour in neighbours[cur]:
                        if not seen[neighbour]:
                            stack.append(neighbour)
                            seen[neighbour] = True
                components.append(component)
        return components

    def grid_index(self, pos):
        """
        Maps a position to the index of its nearest grid point
        """
        i = max(0, min(self.lattice.shape[0]-1, int(round((pos[1]-self.lattice[0,0,1])/self.grid_size))))
        j = max(0, min(self.lattice.shape[1]-1, int(round((pos[0]-self.lattice[0,0,0])/self.grid_size))))
        if tuple(self.lattice[i,j]) not in self.pos_to_index:
            return None
        return self.pos_to_index[tuple(self.lattice[i,j])]

    def _generate_adjacency_matrix(self):
        """
        Generates the graph's adjacency matrix
        """
        data = []
        rows = []
        cols = []
        for node in range(len(self.index_to_pos)):
            data.append(0)
            rows.append(node)
            cols.append(node)
            for nbr in self.neighbours[node]:
                data.append(np.linalg.norm(self.index_to_pos[node] - self.index_to_pos[nbr]))
                rows.append(node)
                cols.append(nbr)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.node_amount, self.node_amount))

    def _frange(self, start, stop, step):
        """
        Returns the fractional range from start to stop (inclusive) by some given step
        """
        i = 0
        vals = []
        while start + i*step <= stop:
            vals.append(start + i*step)
            i += 1
        return vals
