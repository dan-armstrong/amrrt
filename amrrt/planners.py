#Copyright (c) 2020 Ocado. All Rights Reserved.

import time, math, random
import numpy as np

from amrrt.tree import Tree
from amrrt.metrics import EuclideanMetric
from amrrt.cost import Cost


class RTSamplingPlanner:
    """
    Real time sampling planner, parent class for RTRRT and AMRRT
    """
    def __init__(self, space, agent_pos, assisting_metric=None, t_exp=0.15, t_root=0.003, t_steer=0.002):
        """
        :param space: State space planner operates in
        :param agent_pos: Initial position of agent
        :param assisting_metric: Assisting metric for the planner, default None for Euclidean
        :param t_exp: Time spent expanding and rewiring
        :param t_root: Time spent rewiring at the root
        :param t_steer: Time spent steering
        """
        self.space = space
        self.euclidean_metric = EuclideanMetric()
        self.assisting_metric = assisting_metric if assisting_metric is not None else self.euclidean_metric
        self.euclidean_distance = lambda a, b : self.euclidean_metric.distance(a, b)
        self.assisting_distance = lambda a, b : self.assisting_metric.distance(a, b)
        self.tree = Tree(self.space.create_state(agent_pos), self.space, assisting_dist_fn=self.assisting_distance)
        self.goal = None
        self.root_queue = []
        self.rewired_root = set()
        self.t_exp, self.t_root, self.t_steer = t_exp, t_root, t_steer

    def _nearest_node(self, node):
        """
        Find nearest node, first by Euclidean metric if the path is free or fallback on the assisting metric
        """
        euclidean = self.tree.euclidean_nearest(node)
        if self.space.free_path(node, euclidean) or self.euclidean_metric == self.assisting_metric:
            return euclidean
        return self.tree.nearest(node)

    def set_root(self, root):
        """
        Set the root node for the planner tree, and reset rewiring queue
        """
        self.tree.set_root(root)
        self.root_queue = [self.tree.root]
        self.rewired_root = {self.tree.root}

    def set_goal(self, pos):
        """
        Set new goal from position, try to add goal to tree if within range & path free
        """
        self.goal = self.space.create_state(pos)
        xnearest = self._nearest_node(self.goal)
        if self.euclidean_distance(xnearest, self.goal) < self.max_step and self.space.free_path(xnearest, self.goal):
            self.tree.add_node(self.goal, xnearest)

    def _add_node(self, xnew, xnearest, nearby):
        """
        Add new node to the tree, connecting it to the node in nearby that minimises cost
        Also attempt to add the goal to the tree via xnew if within range & free path

        :param xnew: New state to be added
        :param xnearest: Nearest node to xnew
        :param nearby: Set of nodes in the neighbourhood of xnew
        """
        xmin = xnearest
        cmin = self.cost(xnearest) + self.euclidean_distance(xnearest, xnew)
        for xnear in nearby:
            cnew = self.cost(xnear) + self.euclidean_distance(xnear, xnew)
            if cnew < cmin and self.space.free_path(xnear, xnew):
                xmin = xnear
                cmin = cnew
        self.tree.add_node(xnew, xmin)
        if self.goal is not None and self.euclidean_distance(xnew, self.goal) < self.max_step and self.space.free_path(xnew, self.goal):
            self.tree.add_node(self.goal, xnew)

    def goal_path(self):
        """
        Return path to the goal if one exists, if not a path to the nearest node
        """
        if self.goal is None:
            return []
        if self.goal in self.tree.nodes:
            return self.tree.path(self.goal)
        return self.tree.path(self._nearest_node(self.goal))

    def cost(self, node):
        """
        Returns path cost to given node (as Cost object)
        """
        if node == self.tree.root:
            return Cost(0, False)
        path = self.tree.path(node)
        running_cost = 0
        blocked = False
        for i in range(1, len(path)):
            if not self.space.dynamically_free_path(path[i-1], path[i]) : blocked = True
            running_cost += self.euclidean_distance(path[i-1], path[i])
        return Cost(running_cost, blocked)

    def add_dynamic_obstacle(self, pos, radius):
        """
        Add circular dynamic obstacle, and reset goal queues

        :param pos: Position of dynamic obstacle
        :param radius: Radius of dynamic obstacle
        """
        self.space.add_dynamic_obstacle(pos, radius)
        if self.goal is not None : self.set_goal(self.goal.pos)


class RTRRTPlanner(RTSamplingPlanner):
    def __init__(self, space, agent_pos, assisting_metric=None, t_exp=0.15, t_root=0.003, t_rand=0.003, t_steer=0.002):
        """
        :param space: State space planner operates in
        :param agent_pos: Initial position of agent
        :param assisting_metric: Assisting metric for the planner, default None for Euclidean
        :param t_exp: Time spent expanding and rewiring
        :param t_root: Time spent rewiring at the root
        :param t_rand: Time spent rewiring random nodes
        :param t_steer: Time spent steering
        """
        super().__init__(space, agent_pos, assisting_metric=assisting_metric, t_exp=t_exp, t_root=t_root, t_steer=t_steer)
        self.rand_queue = []
        self.t_rand = t_rand
        self.max_step = 5
        self.node_density = 12

    def plan(self, agent_pos):
        """
        Run main body of RT-RRT algorithm, explore & rewire to plan a path

        :param agent_pos: Agent's current position
        """
        agent = self.space.create_state(agent_pos)
        start = time.time()
        while time.time() - start < self.t_exp:
            self.expand()
        goal_path = self.goal_path()
        if self.euclidean_distance(agent, self.tree.root) < self.max_step/10 and len(goal_path) > 1:
            self.set_root(goal_path[1])
        return self.tree.root

    def expand(self):
        """
        Expand the tree through sampling & rewiring
        """
        xrand = self.sample_state()
        xnearest = self.tree.nearest(xrand)
        xnew = self.steer(xnearest, xrand)
        if xnew is not None:
            nearby = self.find_nodes_near(xnew)
            if len(nearby) < self.node_density or self.euclidean_distance(xnearest, xrand) > self.max_step or xnew == self.goal:
                self._add_node(xnew, xnearest, nearby)
                self.rand_queue.insert(0, xnew)
            else:
                self.rand_queue.insert(0, xnearest)
            self.rewire_rand()
        self.rewire_root()

    def find_nodes_near(self, node):
        radius = max(math.sqrt((self.space.area()*self.node_density) / (math.pi*self.tree.node_amount)), self.max_step)
        return self.tree.neighbourhood(node, radius)

    def rewire_root(self):
        """
        Rewire unseen nodes out from the root, resetting rewired_root once all nodes have been visited
        """
        if len(self.root_queue) == 0:
            self.root_queue.append(self.tree.root)
            self.rewired_root = {self.tree.root}
        start = time.time()
        while len(self.root_queue) > 0 and time.time() - start < self.t_root:
            xqueue = self.root_queue.pop()
            nearby = self.find_nodes_near(xqueue)
            for xnear in nearby:
                if self.cost(xnear) + self.euclidean_distance(xnear, xqueue) < self.cost(xqueue) and self.space.free_path(xnear, xqueue):
                    self.tree.update_edge(xnear, xqueue)
                if xnear not in self.rewired_root:
                    self.root_queue.insert(0, xnear)
                    self.rewired_root.add(xnear)

    def rewire_rand(self):
        """
        Rewire random portions of the graph via the rand_queue
        """
        start = time.time()
        while len(self.rand_queue) > 0 and time.time() - start < self.t_rand:
            xqueue = self.rand_queue.pop()
            nearby = self.find_nodes_near(xqueue)
            for xnear in nearby:
                if self.cost(xqueue) + self.euclidean_distance(xqueue, xnear) < self.cost(xnear) and self.space.free_path(xqueue, xnear):
                    self.tree.update_edge(xqueue, xnear)
                    self.rand_queue.insert(0, xnear)

    def sample_state(self, a=0.3, b=1.5):
        """
        Return a randomly sampled state from the state space

        :param a: Scales probability of sampling on the line between the goal and its current nearest neighbour
        :param b: Scales probability of sampling between entire state space and within rewire ellipse
        """
        p = np.random.rand()
        if p > 1-a and self.goal is not None:
            r = np.random.rand()
            return self.space.create_state(r * self.goal.pos + (1-r) * self.tree.nearest(self.goal).pos)
        elif p < (1-a)/b or self.goal is None or self.goal not in self.tree.nodes:
            return self.space.choose_state_uniform()
        else:
            cbest = self.cost(self.goal).to_float()
            cmin = self.euclidean_distance(self.tree.root, self.goal)
            return self.space.choose_state_ellipse(self.tree.root.pos, self.goal.pos, cbest, max(0, cbest**2 - cmin**2)**0.5)

    def steer(self, start, end):
        """
        Return a state that grows the tree towards end from start
        """
        return self.euclidean_metric.steer(self.space, start, end, np.inf, self.t_steer)


class AMRRTPlanner(RTSamplingPlanner):
    def __init__(self, space, agent_pos, assisting_metric=None, t_exp=0.15, t_root=0.002, t_goal=0.004, t_steer=0.002):
        """
        :param space: State space planner operates in
        :param agent_pos: Initial position of agent
        :param assisting_metric: Assisting metric for the planner, default None for Euclidean
        :param t_exp: Time spent expanding and rewiring
        :param t_root: Time spent rewiring at the root
        :param t_goal: Time spent rewiring at the goal
        :param t_steer: Time spent steering
        """
        super().__init__(space, agent_pos, assisting_metric=assisting_metric, t_exp=t_exp, t_root=t_root, t_steer=t_steer)
        self.goal_stack = []
        self.goal_queue = []
        self.t_goal = t_goal
        self.max_step = (self.space.bounds[0,1] - self.space.bounds[0,0]) * 0.05
        self.node_density = 20

    def plan(self, agent_pos):
        """
        Run main body of AM-RRT algorithm, explore & rewire to plan a path

        :param agent_pos: Agent's current position
        """
        agent = self.space.create_state(agent_pos)
        start = time.time()
        while time.time() - start < self.t_exp:
            self.expand()
        goal_path = self.goal_path()
        if self.euclidean_distance(agent, self.tree.root) < self.max_step/10 and len(goal_path) > 1:
            self.set_root(goal_path[1])
        return self.tree.root

    def expand(self):
        """
        Expand the tree through sampling & rewiring
        """
        xrand = self.sample_state()
        xnearest = self._nearest_node(xrand)
        xnew = self.steer(xnearest, xrand)
        if xnew is not None and xnew not in self.tree.nodes:
            nearby = self.tree.neighbourhood(xnew, self.max_step)
            if len(nearby) < self.node_density or self.euclidean_distance(xnearest, xrand) > self.max_step or xnew == self.goal:
                self._add_node(xnew, xnearest, nearby)
        self.rewire_root()
        if self.goal is not None and self.goal in self.tree.nodes:
            self.rewire_goal()

    def sample_state(self, a=0.3, b=1.5):
        """
        Return a randomly sampled state from the state space

        :param a: Scales probability of sampling on the line between the goal and its current nearest neighbour
        :param b: Scales probability of sampling between entire state space and within rewire ellipse
        """
        p = np.random.rand()
        if p > 1-a and self.goal is not None and self.goal not in self.tree.nodes:
            return self.space.create_state(self.goal.pos)
        elif p < (1-a)/b or self.goal is None or self.goal not in self.tree.nodes:
            return self.space.choose_state_uniform()
        else:
            cbest = self.cost(self.goal).to_float()
            cmin = self.euclidean_distance(self.tree.root, self.goal)
            return self.space.choose_state_ellipse(self.tree.root.pos, self.goal.pos, cbest, max(0, cbest**2 - cmin**2)**0.5)

    def rewire_root(self):
        """
        Rewire unseen nodes out from the root, resetting rewired_root once all nodes have been visited
        """
        if len(self.root_queue) == 0:
            self.root_queue.append(self.tree.root)
            self.rewired_root = {self.tree.root}
        start = time.time()
        while len(self.root_queue) > 0 and time.time() - start < self.t_root:
            xrewire = self.root_queue.pop()
            nearby = self.tree.neighbourhood(xrewire, self.max_step)
            for xnear in nearby:
                if self.cost(xnear) + self.euclidean_distance(xnear, xrewire) < self.cost(xrewire) and self.space.free_path(xnear, xrewire):
                    self.tree.update_edge(xnear, xrewire)
                if xnear not in self.rewired_root:
                    self.root_queue.insert(0, xnear)
                    self.rewired_root.add(xnear)

    def rewire_goal(self):
        """
        Rewire unseen nodes out from the root towards the goal along offshoots
        """
        if len(self.goal_stack) == 0 and len(self.goal_queue) == 0:
            self.goal_stack.append(self.tree.root)
            self.seen_goal = set()
        start = time.time()
        while time.time() - start < self.t_goal and (len(self.goal_stack) > 0 or len(self.goal_queue) > 0):
            if len(self.goal_stack) > 0 : xrewire = self.goal_stack.pop()
            else : xrewire = self.goal_queue.pop()
            cbest = self.cost(self.goal).to_float()
            cmin = self.euclidean_distance(self.tree.root, self.goal)
            if self._within_ellipse(xrewire.pos, self.tree.root.pos, self.goal.pos, cbest/2, max(0, cbest**2-cmin**2)**0.5/2):
                nearby = self.tree.neighbourhood(xrewire, self.max_step)
                rev_sorted_indexes = sorted([(-self.assisting_distance(self.goal, nearby[i]), i) for i in range(len(nearby))])
                nearby_sorted = [nearby[i] for _, i in rev_sorted_indexes]
                frontier = []
                for xnear in nearby_sorted:
                    if self.cost(xrewire) + self.euclidean_distance(xrewire, xnear) < self.cost(xnear) and self.space.free_path(xrewire, xnear):
                        self.tree.update_edge(xrewire, xnear)
                    if xnear not in self.seen_goal and self.space.free_path(xrewire, xnear):
                        frontier.append(xnear)
                        self.seen_goal.add(xnear)
                self.goal_stack = self.goal_stack + frontier
                self.goal_queue = frontier + self.goal_queue
            if len(self.goal_stack) > 0 and self.assisting_distance(self.goal_stack[-1], self.goal) > self.assisting_distance(xrewire, self.goal):
                self.goal_stack = []

    def steer(self, start, end):
        """
        Return a state that grows the tree towards end from start
        """
        if self.space.free_path(start, end):
            p = min(1, self.max_step/self.euclidean_distance(start, end))
            return self.space.create_state((1-p)*start.pos + p*end.pos)
        return self.assisting_metric.steer(self.space, start, end, self.max_step, self.t_steer)

    def set_goal(self, pos):
        """
        Set new goal from position, try to add goal to tree if within range & path free, and rest goal stack/queue
        """
        super().set_goal(pos)
        self.goal_stack = [self.tree.root]
        self.goal_queue = []
        self.seen_goal = {self.tree.root}

    def _within_ellipse(self, pos, fa, fb, major, minor):
        """
        Check if point falls within given ellipse

        :param pos: Position of point being checked
        :param fa: Position of first ellipse focus
        :param fb: Position of second ellipse focus
        :param major: Length of major axis
        :param minor: Length of minor axis
        """
        c = (fa + fb) / 2
        r = np.arctan2((fa-fb)[1], (fa-fb)[0])
        return (((np.cos(r)*(pos[0]-c[0]) + np.sin(r)*(pos[1]-c[1]))/major)**2 +
                ((np.sin(r)*(pos[0]-c[0]) - np.cos(r)*(pos[1]-c[1]))/minor)**2 <= 1)
