#Copyright (c) 2020 Ocado. All Rights Reserved.

import random, skimage
from PIL import Image
import numpy as np
from skimage.filters import threshold_mean


class GridEnviroment:
    """
    2D environment represented as a discrete grid of empty/obstacle cells
    """
    def __init__(self, grid):
        """
        :param grid: Binary matrix representing the environment, with 0 = free & 1 = obstacle
        """
        self.grid = grid
        self.dimensions = 2
        nrows, ncols = self.grid.shape
        self.bounds = np.array([[0,ncols], [0,nrows]])
        self.freeness = 1-self.grid.mean()

    @classmethod
    def from_image(cls, image_name):
        """
        Load grid environment from bitmap image by thresholding, setting light pixels as free cells and dark as obstacles

        :param image_name: File name of image to load grid from
        """
        image = Image.open(image_name).convert("L")
        grid = np.asarray(image).copy()
        threshold = threshold_mean(grid)
        grid[grid < threshold] = 1
        grid[grid >= threshold] = 0
        return cls(grid)

    def free_point(self, point):
        """
        Check if given point is in free space
        """
        col, row = map(int, point)
        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
            return self.grid[row, col] == 0
        return False

    def intersections(self, a, b):
        """
        Return set of grid positions on the line between positions a and b that are occupied by obstacles
        """
        ac, ar = map(int, np.maximum(self.bounds[:,0], np.minimum(a, self.bounds[:,1]-1)))
        bc, br = map(int, np.maximum(self.bounds[:,0], np.minimum(b, self.bounds[:,1]-1)))
        rs, cs, _ = skimage.draw.line_aa(ar, ac, br, bc)
        args = np.nonzero(self.grid[rs, cs])
        return np.column_stack((cs[args], rs[args]))

    def closest_intersection(self, a, b):
        """
        Return intersection closest to position a of those on the line between a and b, or None if none exist
        """
        ints = self.intersections(a, b)
        if len(ints) == 0:
            return None
        return ints[0]

    def area(self):
        """
        Return the total free area in the environment
        """
        dimensions = [bound[1] - bound[0] for bound in self.bounds]
        return np.prod(dimensions) * self.freeness

    def display_image(self):
        """
        Return PIL image representing the grid, with free cells represented by a white pixel and occupied cells by black
        """
        image_grid = self.grid.copy()
        image_grid[image_grid == 0] = 255
        image_grid[image_grid == 1] = 0
        return Image.fromarray(image_grid).convert("RGB")


class StateSpace:
    """
    Generic state space class with an internal environment that can be queried for free spaces and obstacles
    """
    def __init__(self, env):
        """
        :param env: Environment the state space can interect with
        """
        self.env = env
        self.dimensions = self.env.dimensions
        self.bounds = self.env.bounds
        self.dynamic_obstacles = set()

    @classmethod
    def from_image(cls, image_name):
        """
        Load state space from image, using 2D grid environment
        """
        return cls(GridEnviroment.from_image(image_name))

    def choose_state_uniform(self):
        """
        Randomly sample free state in the space uniformly
        """
        while True:
            pos = np.random.uniform(self.env.bounds[:,0], self.env.bounds[:,1])
            if self.free_position(pos):
                return self.create_state(pos)

    def choose_state_line(self, a, b):
        """
        Randomly (uniformly) sample free state on the line between a and b
        """
        while True:
            p = np.random.rand()
            pos = (1-p)*a + p*b
            if self.free_position(pos):
                return self.create_state(pos)

    def choose_state_ellipse(self, a, b, major, minor):
        """
        Randomly (uniformly) sample free state in the defined ellipse (for 2D spaces)

        :param a: Position of first focus of the ellipse
        :param b: Position of second focus of the ellipse
        :param major: Length of major axis
        :param minor: Length of minor axis
        """
        if major == np.inf or minor == np.inf:
            return self.choose_state_uniform()
        while True:
            r = np.sqrt(np.random.rand())
            t = np.random.rand() * 2 * np.pi
            rot = np.arctan2((b-a)[1], (b-a)[0])
            xe = major * r * np.cos(t)
            ye = minor * r * np.sin(t)
            xr = xe * np.cos(rot) - ye * np.sin(rot)
            yr = xe * np.sin(rot) + ye * np.cos(rot)
            pos = np.array([xr,yr]) + 0.5*(a+b)
            if self.free_position(pos):
                return self.create_state(pos)

    def create_state(self, pos):
        """
        Create state object from given position
        """
        return State(pos)

    def free_position(self, pos):
        """
        Check if given position is in free space
        """
        if not self.dynamically_free_position(pos):
            return False
        for i in range(self.dimensions):
            if not (self.env.bounds[i][0] <= pos[i] <= self.env.bounds[i][1]):
                return False
        return self.env.free_point(pos)

    def free_path(self, start, end):
        """
        Check if the line between start and end is obstacle free
        """
        if not self.dynamically_free_path(start, end):
            return False
        if not self.free_position(start.pos) or not self.free_position(end.pos):
            return False
        return len(self.env.intersections(start.pos, end.pos)) == 0

    def dynamically_free_position(self, pos):
        """
        Check if given position is free from dynamic obstacles
        """
        for obstacle in self.dynamic_obstacles:
            if not obstacle.free_position(pos) : return False
        return True

    def dynamically_free_path(self, start, end):
        """
        Check if the line between start and end is free from dynamic obstacles
        """
        for obstacle in self.dynamic_obstacles:
            if not obstacle.free_path(start, end) : return False
        return True

    def area(self):
        """
        Return the total free area of the state space
        """
        return self.env.area()

    def display_image(self):
        """
        Return a PIL image representing the state space
        """
        return self.env.display_image()

    def add_dynamic_obstacle(self, pos, radius):
        """
        Add circular dynamic obstacle

        :param pos: Position of dynamic obstacle
        :param radius: Radius of dynamic obstacle
        """
        self.dynamic_obstacles.add(DynamicObstacle(pos, radius))


class State:
    """
    Class encapsulating a single state in the state space
    """
    def __init__(self, pos):
        """
        :param pos: Position that the state represents
        """
        self.pos = np.array(pos, dtype="float")

    def __eq__(self, other):
        if isinstance(other, State):
            return (self.pos == other.pos).all()
        return False

    def __hash__(self):
        return hash(tuple(self.pos))


class DynamicObstacle:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    def free_position(self, pos):
        """
        Check that the given pos is not within the obstacle
        """
        return np.linalg.norm(pos-self.pos) > self.radius

    def free_path(self, start, end):
        """
        Check if the line between start and end does not intersect the obstacle
        """
        a = start.pos[1] - end.pos[1]
        b = end.pos[0] - start.pos[0]
        c = -(a*start.pos[0] + b*start.pos[1])
        if a == b == 0 : return np.linalg.norm(start.pos-self.pos) > self.radius
        return abs(a*self.pos[0] + b*self.pos[1] + c) / (a**2 + b**2)**0.5 > self.radius

    def __eq__(self, other):
        if isinstance(other, DynamicObstacle):
            return (self.pos == other.pos).all() and self.radius == other.radius
        return False

    def __hash__(self):
        return hash((tuple(self.pos), self.radius))
