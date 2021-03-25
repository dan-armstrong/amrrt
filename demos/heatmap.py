import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

import pygame
import numpy as np
import time
import math

from matplotlib import cm
from PIL import Image
import numpy as np

from amrrt.space import StateSpace
from amrrt.diffusion_map import DiffusionMap

def make_heatmap(space, diffusion_map, goal_pos):
    colour_map = cm.get_cmap('gist_rainbow', 1000)
    image = Image.new('RGB', (int(space.bounds[0][1])-int(space.bounds[0][0]), int(space.bounds[1][1])-int(space.bounds[1][0])))
    pixels = image.load()
    max_dist = 0
    distances = {}
    for x in range(int(space.bounds[0][0]), int(space.bounds[0][1])):
        for y in range(int(space.bounds[1][0]), int(space.bounds[1][1])):
            if space.free_position(np.array([x+0.5,y+0.5])):
                dist = np.linalg.norm(diffusion_map.diffusion_pos(np.array([x+0.5,y+0.5])) - diffusion_map.diffusion_pos(goal_pos))
                distances[(x,y)] = dist
                if dist < np.inf and dist > max_dist:
                    max_dist = dist
    for x in range(int(space.bounds[0][0]), int(space.bounds[0][1])):
        for y in range(int(space.bounds[1][0]), int(space.bounds[1][1])):
            if space.free_position(np.array([x+0.5,y+0.5])):
                red, green, blue, _ = colour_map(distances[(x,y)] / max_dist)
                pixels[x, y] = (int(red*255), int(green*255), int(blue*255))
    pixels[int(goal_pos[0]), int(goal_pos[1])] = (255, 255, 255)
    return image


if __name__ == "__main__":
    pygame.init()
    space = StateSpace.from_image("demos/resources/office.png")
    diffusion_map = DiffusionMap(space)
    heatmap = make_heatmap(space, diffusion_map, space.bounds[:,0].transpose())
    scale = 700/min(heatmap.size[0],heatmap.size[1])
    width, height = int(heatmap.size[0]*scale), int(heatmap.size[1]*scale)
    screen = pygame.display.set_mode((width, height))
    heatmap = heatmap.resize((width, height))
    screen.blit(pygame.image.frombuffer(heatmap.tobytes(), heatmap.size, heatmap.mode), (0,0))
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                heatmap = make_heatmap(space, diffusion_map, np.array([int(x/scale)+0.5,int(y/scale)+0.5])).resize((width,height))
                screen.blit(pygame.image.frombuffer(heatmap.tobytes(), heatmap.size, heatmap.mode), (0,0))
                pygame.display.update()
