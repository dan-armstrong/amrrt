import sys, time, math, matplotlib, sys, os, pygame
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace
from amrrt.diffusion_map import DiffusionMap, GridGraph
from amrrt.metrics import EuclideanMetric, DiffusionMetric, GeodesicMetric
from amrrt.planners import AMRRTPlanner, RTRRTPlanner


if __name__ == "__main__":
    pygame.init()

    images = ["empty", "bug_trap", "maze", "office"]
    spaces = [StateSpace.from_image(os.path.join("demos", "resources", image + ".png")) for image in images]

    start_positions = [np.array([50.0, 50.0]), np.array([35.0, 33.0]), np.array([63.0, 51.0]), np.array([79.0, 65.0])]
    goal_positions = np.array([[[13.0, 33.2], [15.0, 78.0], [92.4, 8.2], [92.6, 93.6], [32.1, 7.0], [4.9, 56.0]],
                               [[11.2, 49.2], [35.0, 70.0], [90.0, 59.2], [8.4, 8.2], [27.0, 90.0], [90.0, 10.0]],
                               [[7.2, 6.6], [28.0, 81.0], [35.0, 37.0], [93.0, 91.0], [89.2,  7.0], [16.2, 37.2]],
                               [[125.6, 141.6], [162.6, 13.4], [170.8, 128.0], [60.0, 185.8], [23.8, 48.6], [185.0, 178.2]]])
    i = 1
    (space, start_pos, goal_positions) = list(zip(spaces, start_positions, goal_positions))[i]
    waypoints = np.array([start_pos] + list(goal_positions))


    planner = None
    agent_pos = None
    image = space.display_image()
    image_size = 800
    scale = image_size/min(image.size[0],image.size[1])
    width, height = int(image.size[0]*scale), int(image.size[1]*scale)
    screen = pygame.display.set_mode((width, height))
    image = image.resize((width, height), resample=Image.NEAREST)
    image = pygame.image.frombuffer(image.tobytes(), image.size, image.mode)

    shortest_paths = []
    pos = start_pos
    cur = 0
    i = 0


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                new = np.rint(np.array(pygame.mouse.get_pos()) / scale)
                print(new)
                cur += np.linalg.norm(new - pos)
                pos = new
            elif pygame.key.get_pressed()[pygame.K_d]:
                new = goal_positions[i]
                cur += np.linalg.norm(new - pos)
                pos = new
                shortest_paths.append(cur)
                cur = 0
                i += 1
                print(shortest_paths)
        screen.blit(image, (0, 0))
        pygame.draw.circle(screen, (255,30,30), tuple((waypoints[i]*scale).astype(int)), 4)
        pygame.draw.circle(screen, (255,30,30), tuple((waypoints[i+1]*scale).astype(int)), 4)
        pygame.display.update()
