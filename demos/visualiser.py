#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os, pygame, argparse
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace
from amrrt.diffusion_map import DiffusionMap, GridGraph
from amrrt.metrics import EuclideanMetric, DiffusionMetric, GeodesicMetric
from amrrt.planners import AMRRTPlanner, RTRRTPlanner


def display(screen, planner, agent_pos, path, image, scale):
    screen.blit(image, (0, 0))
    for u in planner.tree.edges:
        for v in planner.tree.edges[u]:
            a = (int(u.pos[0]*scale), int(u.pos[1]*scale))
            b = (int(v.pos[0]*scale), int(v.pos[1]*scale))
            pygame.draw.line(screen, (0,0,0), a, b, 1)
    display_path = [planner.space.create_state(agent_pos)] + path
    for i in range(len(display_path)-1):
        a = (int(display_path[i].pos[0]*scale), int(display_path[i].pos[1]*scale))
        b = (int(display_path[i+1].pos[0]*scale), int(display_path[i+1].pos[1]*scale))
        pygame.draw.line(screen, (30,30,255), a, b, 3)
    for obstacle in planner.space.dynamic_obstacles:
        pygame.draw.circle(screen, (0,0,0), tuple((obstacle.pos*scale).astype(int)), int(obstacle.radius*scale))
    pygame.draw.circle(screen, (255,30,30), tuple((agent_pos*scale).astype(int)), 4)
    if planner.goal is not None:
        pygame.draw.circle(screen, (30,255,30), tuple((planner.goal.pos*scale).astype(int)), 4)
    pygame.display.update()


def visualiser(space, planner_type, metric_type):
    pygame.init()
    grid_graph = GridGraph(space)
    if metric_type == "euclidean" : assisting_metric = EuclideanMetric()
    elif metric_type == "diffusion" : assisting_metric = DiffusionMetric(DiffusionMap(space, grid_graph=grid_graph))
    else : assisting_metric = GeodesicMetric(grid_graph)
    planner = None
    agent_pos = None
    image = space.display_image()
    image_size = 800
    scale = image_size/min(image.size[0],image.size[1])
    width, height = int(image.size[0]*scale), int(image.size[1]*scale)
    screen = pygame.display.set_mode((width, height))
    image = image.resize((width, height), resample=Image.NEAREST)
    image = pygame.image.frombuffer(image.tobytes(), image.size, image.mode)

    while planner is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = np.array(pygame.mouse.get_pos()) / scale
                if planner_type == "rtrrt" : planner = RTRRTPlanner(space, pos, assisting_metric=assisting_metric)
                else : planner = AMRRTPlanner(space, pos, assisting_metric=assisting_metric)
                agent_pos = pos
        screen.blit(image, (0, 0))
        pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = np.array(pygame.mouse.get_pos()) / scale
                if pygame.key.get_pressed()[pygame.K_d]:
                    planner.add_dynamic_obstacle(pos, 3)
                else:
                    planner.set_goal(pos)
        path = []
        waypoint = planner.plan(agent_pos).pos.copy()
        path = planner.goal_path()
        agent_pos += min(1, 0.7/np.linalg.norm(waypoint-agent_pos)) * (waypoint-agent_pos)
        display(screen, planner, agent_pos, path, image, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AM-RRT* & RT-RRT* graphical visualiser')
    parser.add_argument('image', type=str, help='Filename of an image containing the environment')
    parser.add_argument('planner', choices=['rtrrt', 'amrrt'], help='Name of planner (choices: "rtrrt", "amrrt")')
    parser.add_argument('metric_type', choices=['euclidean', 'diffusion', 'geodesic'], help='Name of assisting metric (choices: "euclidean", "diffusion", "geodesic")')
    args = parser.parse_args()
    visualiser(StateSpace.from_image(args.image), args.planner, args.metric_type)
