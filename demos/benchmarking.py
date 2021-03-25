#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, time, matplotlib, os, argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace
from amrrt.diffusion_map import DiffusionMap
from amrrt.grid_graph import GridGraph
from amrrt.metrics import DiffusionMetric, GeodesicMetric
from amrrt.planners import AMRRTPlanner, RTRRTPlanner


def planners(space, diffusion_map, distance_matrix, start_pos):
    rtrrt_e = RTRRTPlanner(space, start_pos)
    rtrrt_d = RTRRTPlanner(space, start_pos, assisting_metric=DiffusionMetric(diffusion_map))
    amrrt_e = AMRRTPlanner(space, start_pos)
    amrrt_d = AMRRTPlanner(space, start_pos, assisting_metric=DiffusionMetric(diffusion_map))
    if distance_matrix is not None:
        amrrt_g = AMRRTPlanner(space, start_pos, assisting_metric=GeodesicMetric(diffusion_map.grid_graph, distance_matrix=distance_matrix))
        return [rtrrt_e, rtrrt_d, amrrt_e, amrrt_d, amrrt_g]
    return [rtrrt_e, rtrrt_d, amrrt_e, amrrt_d]


def display_data(fig, ax, distances, times):
    labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
    avgs = np.array([np.nanmean(distances, axis=3), np.nanmean(times, axis=3)])
    x = np.arange(len(labels))
    bar_width = 0.1
    planner_amount = distances.shape[1]
    colours = ['#400082', '#00bdaa', '#ff0000', '#ffc800', '#00c800']
    planners = ['RT-RRT*', 'RT-RRT*(D)', 'AM-RRT*(E)', 'AM-RRT*(D)', 'AM-RRT*(G)']
    environments = ["Empty", "Bug Trap", "Maze", "Office"]

    for i in range(2):
        for j in range(4):
            ax[i,j].clear()
            for k in range(planner_amount):
                ax[i,j].bar(x + (k + 0.5 - planner_amount/2)*bar_width*1.15, avgs[i,j,k], bar_width, color=colours[k])
            ax[i,j].set_xticks(x)
            ax[i,j].set_xticklabels(labels)
            if i == 0:
                ax[i,j].text(0.5, 1.2, environments[j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round",
                         ec=(0,0,0,0), fc=(0,0,0,0)), transform=ax[i,j].transAxes)
            else:
                ax[i,j].set_yscale("log")
    fig.text(0.5, 0.013, "Goal Number", ha='center', va='center', fontsize=12)
    fig.text(0.015, 0.75, "Path length", ha='center', va='center', rotation='vertical')
    fig.text(0.015, 0.25, "Search time (s)", ha='center', va='center', rotation='vertical')
    fig.legend(tuple([matplotlib.lines.Line2D([], [], color=colours[i]) for i in range(planner_amount)]),
               tuple(planners[:planner_amount]), loc=(0.01,0.8), framealpha=0, markerfirst=True, labelspacing=0.2)
    plt.pause(0.01)


def generate_benchmarking_resources(images, include_geodesic_metric):
    spaces = [StateSpace.from_image(image) for image in images]
    grid_graphs = [GridGraph(space) for space in spaces]
    diffusion_maps = [DiffusionMap(space, grid_graph=grid_graph) for space, grid_graph in zip(spaces, grid_graphs)]
    if include_geodesic_metric:
        distance_matrices = [GeodesicMetric(grid_graph).distance_matrix for grid_graph in grid_graphs]
    else:
        distance_matrices = [None] * len(images)
    return spaces, diffusion_maps, distance_matrices


def load_position_data():
    start_positions = [np.array([50.0, 50.0]), np.array([35.0, 33.0]), np.array([63.0, 51.0]), np.array([79.0, 65.0])]
    goal_positions = np.array([[[13.0, 33.2], [15.0, 78.0], [92.4, 8.2], [92.6, 93.6], [32.1, 7.0], [4.9, 56.0]],
                               [[11.2, 49.2], [35.0, 70.0], [90.0, 59.2], [8.4, 8.2], [27.0, 90.0], [90.0, 10.0]],
                               [[7.2, 6.6], [28.0, 81.0], [35.0, 37.0], [93.0, 91.0], [89.2,  7.0], [16.2, 37.2]],
                               [[125.6, 141.6], [162.6, 13.4], [170.8, 128.0], [60.0, 185.8], [23.8, 48.6], [185.0, 178.2]]])
    intermediate_goal_paths = [[[],[],[],[],[],[]],
                               [[[56.3, 38.5], [56.3, 50.3]],
                                [[52.4, 50.3], [51.4, 68.7]],
                                [[54.8, 66.8], [51.9, 52.2], [14.9, 52.3], [13.4, 89.2], [89.4, 91.5]],
                                [[89.3, 9.2]],
                                [[7.9, 90.0]],
                                [[88.0, 89.1]]],
                               [[[52.1, 26.5], [32.6, 25.1], [31.8, 19.7], [7.3, 21.7]],
                                [[8.6, 80.3]],
                                [[11.5, 79.3], [8.8, 58.4], [25.6, 55.4], [25.3, 38.5]],
                                [[47.0, 30.2], [59.0, 45.3], [59.8, 88.9]],
                                [[60.2, 86.5], [59.9, 11.5]],
                                [[17.5, 7.1], [16.6, 24.9], [8.2, 27.5], [8.2, 39.5]]],
                               [[[87.2, 77.6], [88.4, 141.0]],
                                [[121.8, 89.2], [133.2, 49.4], [144.2, 11.0]],
                                [[187.6, 13.2], [189.4, 81.0], [179.6, 86.0]],
                                [[126.8, 131.0], [93.8, 152.0], [63.8, 154.2], [65.2, 163.4], [70.8, 166.6]],
                                [[70.6, 171.4], [67.6, 163.0], [54.4, 146.2], [57.6, 72.8], [68.0, 71.2], [67.0, 48.6]],
                                [[63.4, 47.6], [87.0, 67.8], [86.8, 148.4], [124.0, 150.4], [125.0, 170.2], [146.4, 169.4], [180.2, 151.4], [187.6, 150.2]]]]
    return start_positions, goal_positions, intermediate_goal_paths


def move_agent(agent_pos, waypoint, distance):
    prev_pos = agent_pos.copy()
    agent_pos += min(1, 0.7/np.linalg.norm(waypoint-agent_pos)) * (waypoint-agent_pos)
    return agent_pos, distance + np.linalg.norm(prev_pos-agent_pos)


def set_next_goal(planner, goal_count, goal_amount, goal_positions):
    if goal_count + 1 < goal_amount:
        planner.set_goal(goal_positions[goal_count+1].copy())
    return goal_count+1, 0, time.time()


def goal_planning_timed_out(planner, agent_pos, start_pos, goal_count, goal_positions, intermediate_goal_paths):
    prev_goal = start_pos.copy() if goal_count == 0 else goal_positions[goal_count-1].copy()
    cur_goal = goal_positions[goal_count].copy()
    intermediate_path = intermediate_goal_paths[goal_count].copy()
    add_intermediate_path(planner, prev_goal, cur_goal, intermediate_path)
    planner.set_root(space.create_state(goal_positions[goal_count].copy()))
    return goal_positions[goal_count].copy()


def add_intermediate_path(planner, start, goal, intermediate_path):
    current = start
    waypoints = list(map(np.array, intermediate_path + [goal]))
    for i, waypoint in enumerate(waypoints):
        while not (current == waypoints[i]).all():
            p = min(1, planner.max_step/np.linalg.norm(current - waypoints[i]))
            next = (1-p)*current + p*waypoints[i]
            planner.tree.add_node(planner.space.create_state(next), planner.space.create_state(current))
            current = next


def benchmark(start_pos, planner, goal_amount, goal_positions, intermediate_goal_paths):
    agent_pos = start_pos.copy()
    goal_count = 0
    planner.set_goal(goal_positions[0].copy())
    goal_distance = 0
    set_iter = False
    start_time = time.time()
    goal_start_time = start_time
    distances = np.full(goal_amount, np.nan)
    times = np.full(goal_amount, np.nan)
    while goal_count < goal_amount:
        waypoint = planner.plan(agent_pos).pos.copy()
        if planner.goal in planner.tree.nodes and np.isnan(times[goal_count]):
            times[goal_count] = time.time()-goal_start_time
        if planner.goal in planner.tree.nodes and time.time()-goal_start_time > 0.25:
            agent_pos, goal_distance = move_agent(agent_pos, waypoint, goal_distance)
        if (agent_pos == planner.goal.pos).all():
            distances[goal_count] = goal_distance
            goal_count, goal_distance, goal_start_time = set_next_goal(planner, goal_count, goal_amount, goal_positions)
        elif time.time()-goal_start_time > 1250:                    #PLANNING TIMED OUT, GO TO NEXT GOAL
            times[i,j,goal_count,r] = time.time()-goal_start_time
            agent_pos = goal_planning_timed_out(planner, agent_pos, start_pos, goal_count, goal_positions, intermediate_goal_paths)
            goal_count, goal_distance, goal_start_time = set_next_goal(planner, goal_count, goal_amount, goal_positions)
        print("Time to goal " + str(goal_count+1) + ":", time.time()-goal_start_time, "              ", end="\r")
    return distances, times


def benchmark_all(include_geodesic_metric):
    images = [os.path.join("demos", "resources", image + ".png") for image in ["empty", "bug_trap", "maze", "office"]]
    spaces, diffusion_maps, distance_matrices = generate_benchmarking_resources(images, include_geodesic_metric)
    start_positions, goal_positions, intermediate_goal_paths = load_position_data()

    repeats = 25
    planner_names = ['RT-RRT*', 'RT-RRT*(D)', 'AM-RRT*(E)', 'AM-RRT*(D)', 'AM-RRT*(G)']
    environments = ["Empty", "Bug Trap", "Maze", "Office"]
    goal_amount = goal_positions.shape[1]
    planner_amount = 5 if include_geodesic_metric else 4
    distances = np.full((len(spaces), planner_amount, goal_amount, repeats), np.nan)
    times = np.full((len(spaces), planner_amount, goal_amount, repeats), np.nan)
    fig, ax = plt.subplots(2, len(spaces), sharex='col', sharey='row', figsize=(15, 5))

    for r in range(repeats):
        for i, (space, diffusion_map, distance_matrix, start_pos) in enumerate(zip(spaces, diffusion_maps, distance_matrices, start_positions)):
            for j, planner in enumerate(planners(space, diffusion_map, distance_matrix, start_pos)):
                print("Repeat:", r, "   Planner:", planner_names[i], "   Environment:", environments[j], "                            ")
                distances[i,j,:,r], times[i,j,:,r] = benchmark(start_pos, planner, goal_amount, goal_positions[i], intermediate_goal_paths[i])
                if r > 0 : display_data(fig, ax, distances, times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AM-RRT* & RT-RRT* benchmarking')
    parser.add_argument('--include_geodesic', default=False, type=bool, help='Include AM-RRT*(G) in benchmarking (default: False)')
    args = parser.parse_args()
    benchmark_all(args.include_geodesic)
