#Copyright (c) 2020 Ocado. All Rights Reserved.

import matplotlib, os, sys
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace
from amrrt.grid_graph import GridGraph
from amrrt.metrics import GeodesicMetric

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
'''
fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.7, 'hspace':0.1, 'left':0, 'right':1, 'top':1, 'bottom':0}, figsize=(12, 2.7))
start_position = np.array([6.78531739, 91.99546438])*10
goal_position = np.array([83.0, 33.0])*10
for i in range(3):
    ax[i].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/rewire_' + ['rtrrt_euclidean', 'amrrt_euclidean', 'amrrt_diffusion'][i] + '.png')), zorder=-1)
    ax[i].text(start_position[0], start_position[1], "A", size=8, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,1)))
    ax[i].text(goal_position[0], goal_position[1], "G", size=8, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,1)))
    ax[i].axis('off')
plt.savefig('rewire_visuals.pgf')
plt.show()
'''

distances = np.load("distances.npy")
times = np.load("times.npy")

images = ["empty", "bug_trap", "maze", "office"]
spaces = [StateSpace.from_image(os.path.join("demos/resources", image + ".png")) for image in images]
grid_graphs = [GridGraph.from_saved(space, os.path.join("demos/resources", image)) for space, image in zip(spaces, images)]
distance_matrices = [np.load(os.path.join("demos/resources", image, "distance_matrix.npy")) for image in images]
geodesic_metrics = [GeodesicMetric(grid_graph, distance_matrix) for grid_graph, distance_matrix in zip(grid_graphs, distance_matrices)]

waypoints = np.array([[[50.0, 50.0], [13.0, 33.2], [15.0, 78.0], [92.4, 8.2], [92.6, 93.6], [32.1, 7.0], [4.9, 56.0]],
                      [[35.0, 33.0], [11.2, 49.2], [35.0, 70.0], [90.0, 59.2], [8.4, 8.2], [53.0, 90.0], [90.0, 47.0]],
                      [[63.0, 51.0], [7.2, 6.6], [28.0, 81.0], [35.0, 37.0], [93.0, 91.0], [89.2,  7.0], [16.2, 37.2]],
                      [[79.0, 65.0], [125.6, 141.6], [162.6, 13.4], [170.8, 128.0], [60.0, 185.8], [23.8, 48.6], [185.0, 178.2]]])

shortest_paths = [[np.linalg.norm(waypoints[0,i] - waypoints[0,i+1]) for i in range(distances.shape[2])],
                  [61.484680484261155, 63.12353620072288, 172.19053767418228, 116.21909186775824, 86.46336558701273, 128.02185403146035],
                  [74.61747343047412, 84.61575781480384, 64.40995862204083, 98.07522464033143, 127.19815498031583, 95.70905438785839],
                  [108.02716559200414, 138.82642333275155, 116.63626014222743, 164.60327397998313, 159.42910328956557, 233.444084954655]]

for i in range(distances.shape[0]):
    for j in range(distances.shape[2]):
        distances[i,:,j] /= shortest_paths[i][j]

start_positions = [np.array([50.0, 50.0]), np.array([35.0, 33.0]), np.array([63.0, 51.0]), np.array([79.0, 65.0])]
goal_positions = np.array([[[13.0, 33.2], [15.0, 78.0], [92.4, 9.2], [92.6, 91.6], [32.1, 9.0], [6.9, 56.0]],
                           [[11.2, 49.2], [35.0, 70.0], [90.0, 59.2], [8.4, 8.2], [27.0, 90.0], [90.0, 10.0]],
                           [[7.2, 7.6], [28.0, 83.0], [37.0, 37.0], [93.0, 91.0], [89.2,  7.5], [14.2, 39.2]],
                           [[125.6, 141.6], [162.6, 13.4], [170.8, 122.0], [65.0, 185.8], [24.8, 50.6], [185.0, 178.2]]])

fig, ax = plt.subplots(3, 4, figsize=(12, 5.4), subplot_kw={'xmargin': 0.04, 'ymargin':0.08}, gridspec_kw={'wspace':0.1, 'hspace':0.15, 'left':0.05, 'right':0.95, 'top':0.99, 'bottom':0.075, 'height_ratios': [0.6,1,1]})

labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
avgs = np.array([np.nanmean(distances, axis=3), np.nanmean(times, axis=3)])
bar_width = 0.12
for i in range(3):
    for j in range(4):
        if i > 0:
            x = np.arange(len(labels))
            bars1 = ax[i,j].bar(x - 2*bar_width*1.3, avgs[i-1,j,0], bar_width, color='#00bdaa')
            bars2 = ax[i,j].bar(x - 1*bar_width*1.3, avgs[i-1,j,1], bar_width, color='#400082')
            bars3 = ax[i,j].bar(x, avgs[i-1,j,2], bar_width, color='#ff0000')
            bars4 = ax[i,j].bar(x + 1*bar_width*1.3, avgs[i-1,j,3], bar_width, color='#00c800')
            bars5 = ax[i,j].bar(x + 2*bar_width*1.3, avgs[i-1,j,4], bar_width, color='#ffa600')
            ax[i,j].set_xticks(x)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            if i == 1:
                ax[i,j].set_xticklabels([])
                ax[i,j].set_ylim(0.75,1.75)
                ax[i,j].set_yticks([0.75,1,1.25,1.5,1.75])
                ax[i,j].text(2.5, 1.82, ['Empty', 'Bug Trap', 'Maze', 'Office'][j], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
            else:
                ax[i,j].set_xticklabels(labels)
                ax[i,j].set_ylim(0.01,1000)
                ax[i,j].set_yticks([0.01, 0.1, 1, 10, 100, 1000])
                ax[i,j].set_yscale("log")
            if j > 0 : ax[i,j].set_yticklabels([])
        else:
            ax[i,j].axis('off')
            ax[i,j].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/demos/resources/' + images[j] + '.png')), zorder=-1)
            ax[i,j].text(start_positions[j][0], start_positions[j][1], "S", size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
            for k, (x, y) in enumerate(goal_positions[j]):
                if j == 3 : x, y = x * 1, y * 1
                ax[i,j].text(x, y, ""+str(k+1), size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
"""for i in range(4):
    imax = fig.add_axes([0.787, 0.765 - 0.227*i, 0.19, 0.19], anchor='NE', zorder=-1)
    imax.imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/demos/resources/' + images[i] + '.png')), zorder=-1)
    imax.axis('off')
    imax.text(start_positions[i][0], start_positions[i][1], "S", size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    for k, (x, y) in enumerate(goal_positions[i]):
        imax.text(x, y, ""+str(k+1), size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    if i < 3 : imax.text(110, 50, ['Empty', 'Bug Trap', 'Maze', 'Office'][i], ha="center", va="center", rotation=-90)
    else : imax.text(220, 100, ['Empty', 'Bug Trap', 'Maze', 'Office'][i], ha="center", va="center", rotation=-90)
"""
fig.text(0.5, 0.013, "Goal Number", ha='center', va='center', size=11)
fig.text(0.0105, 0.596, "Path length ratio", ha='center', va='center', rotation='vertical', size=11)
fig.text(0.0105, 0.238, "Search time (s)", ha='center', va='center', rotation='vertical', size=11)
fig.legend((matplotlib.lines.Line2D([], [], color='#00bdaa'), matplotlib.lines.Line2D([], [], color='#400082'),
            matplotlib.lines.Line2D([], [], color='#ff0000'), matplotlib.lines.Line2D([], [], color='#00c800'), matplotlib.lines.Line2D([], [], color='#ffa600')),
            ('RT-RRT*', 'RT-RRT*(D)', 'AM-RRT*(E)', 'AM-RRT*(D)', 'AM-RRT*(G)'), loc=(0.057,0.61), markerfirst=True, labelspacing=0.2, prop={'size': 8}, markerscale=1.4)

for i in range(avgs.shape[2]):
    print(i,j,np.sum(avgs[0,:,i]),(np.sum(avgs[0,:,0])-np.sum(avgs[0,:,i])) / np.sum(avgs[0,:,0]) * 100)
    print(i,j,np.sum(avgs[1,:,i]),(np.sum(avgs[1,:,0])-np.sum(avgs[1,:,i])) / np.sum(avgs[1,:,0]) * 100)
    print(i,j,np.sum(avgs[0,3,i]),(np.sum(avgs[0,3,0])-np.sum(avgs[0,3,i])) / np.sum(avgs[0,3,0]) * 100)
    print(i,j,np.sum(avgs[1,3,i]),(np.sum(avgs[1,3,0])-np.sum(avgs[1,3,i])) / np.sum(avgs[1,3,0]) * 100)
    print()


print(np.sum(avgs[1,3,1]) / np.sum(avgs[1,3,3]))
print(np.sum(avgs[1,:,0]) / (26.1 + np.sum(avgs[1,:,3])))
print(np.mean(avgs[0,:,3]))

plt.show()
plt.savefig('all.pgf')

"""
fig, ax = plt.subplots(2, 5, figsize=(13, 4.7), subplot_kw={'xmargin': 0.08, 'ymargin':0.08}, gridspec_kw={'wspace':0.14, 'hspace':0.15, 'left':0.05, 'right':0.95, 'top':0.955, 'bottom':0.085, 'width_ratios': [2.3,2.3,2.3,2.3,0.5]})

labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
avgs = np.array([np.nanmean(distances, axis=3), np.nanmean(times, axis=3)])
bar_width = 0.1
for i in range(2):
    for j in range(5):
        if j < 4:
            x = np.arange(len(labels))
            bars1 = ax[i,j].bar(x - 2*bar_width, avgs[i,j,0], bar_width, color='#400082')
            bars2 = ax[i,j].bar(x - 1*bar_width, avgs[i,j,1], bar_width, color='#00bdaa')
            bars3 = ax[i,j].bar(x, avgs[i,j,2], bar_width, color='#ff0000')
            bars4 = ax[i,j].bar(x + 1*bar_width, avgs[i,j,3], bar_width, color='#ffc800')
            bars5 = ax[i,j].bar(x + 2*bar_width, avgs[i,j,4], bar_width, color='#00c800')
            ax[i,j].set_xticks(x)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            if i == 0:
                ax[i,j].set_xticklabels([])
                ax[i,j].set_ylim(0.75,1.75)
                ax[i,j].set_yticks([0.75,1,1.25,1.5,1.75])
                ax[i,j].text(2.5, 2.09, ['Empty', 'Bug Trap', 'Maze', 'Office'][j], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
            else:
                ax[i,j].set_xticklabels(labels)
                ax[i,j].set_ylim(0.01,1000)
                ax[i,j].set_yticks([0.01, 0.1, 1, 10, 100, 1000])
                ax[i,j].set_yscale("log")
            if j > 0 : ax[i,j].set_yticklabels([])
        else:
            ax[i,j].axis('off')
for i in range(4):
    imax = fig.add_axes([0.787, 0.765 - 0.227*i, 0.19, 0.19], anchor='NE', zorder=-1)
    imax.imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/demos/resources/' + images[i] + '.png')), zorder=-1)
    imax.axis('off')
    imax.text(start_positions[i][0], start_positions[i][1], "S", size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    for k, (x, y) in enumerate(goal_positions[i]):
        imax.text(x, y, ""+str(k+1), size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    if i < 3 : imax.text(110, 50, ['Empty', 'Bug Trap', 'Maze', 'Office'][i], ha="center", va="center", rotation=-90)
    else : imax.text(220, 100, ['Empty', 'Bug Trap', 'Maze', 'Office'][i], ha="center", va="center", rotation=-90)

fig.text(0.5, 0.013, "Goal Number", ha='center', va='center', size=11)
fig.text(0.015, 0.76, "Path length ratio", ha='center', va='center', rotation='vertical', size=11)
fig.text(0.015, 0.29, "Search time", ha='center', va='center', rotation='vertical', size=11)
fig.legend((matplotlib.lines.Line2D([], [], color='#400082'), matplotlib.lines.Line2D([], [], color='#00bdaa'),
            matplotlib.lines.Line2D([], [], color='#ff0000'), matplotlib.lines.Line2D([], [], color='#ffc800'), matplotlib.lines.Line2D([], [], color='#00c800')),
            ('RT-RRT*', 'RT-RRT*(D)', 'AM-RRT*(E)', 'AM-RRT*(D)', 'AM-RRT*(G)'), loc=(0.054,0.79), markerfirst=True, labelspacing=0.2, prop={'size': 8})

print(np.nonzero(times > 15000*0.15))
for i in range(avgs.shape[2]):
    print(i,j,np.sum(avgs[0,:,i]),(np.sum(avgs[0,:,0])-np.sum(avgs[0,:,i])) / np.sum(avgs[0,:,0]) * 100)
    print(i,j,np.sum(avgs[1,:,i]),(np.sum(avgs[1,:,0])-np.sum(avgs[1,:,i])) / np.sum(avgs[1,:,0]) * 100)
    print(i,j,np.sum(avgs[0,3,i]),(np.sum(avgs[0,3,0])-np.sum(avgs[0,3,i])) / np.sum(avgs[0,3,0]) * 100)
    print(i,j,np.sum(avgs[1,3,i]),(np.sum(avgs[1,3,0])-np.sum(avgs[1,3,i])) / np.sum(avgs[1,3,0]) * 100)
    print()

print(np.sum(avgs[1,3,1]) / np.sum(avgs[1,3,3]))
print(np.mean(avgs[0,:,3]))
plt.show()
plt.savefig('all.pgf')
"""
'''fig, ax = plt.subplots(1, 4, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.08, 'hspace':0.08, 'left':0.001, 'right':0.999, 'top':0.9, 'bottom':0.001}, figsize=(12, 3.2))
for i in range(4):
    ax[i].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/bm/' + images[i] + '.png')), zorder=-1)
    ax[i].text(50, -6, ['Empty', 'Bug Trap', 'Maze', 'Office'][i], size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
    ax[i].text(start_positions[i][0], start_positions[i][1], "S", size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    for k, (x, y) in enumerate(goal_positions[i]):
        ax[i].text(x, y, ""+str(k+1), size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    ax[i].axis('off')
plt.savefig('images.pgf')
'''

"""
        if i == 0:
            ax[i,j].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/bm/' + images[j] + '.png')), zorder=-1)
            if j == 3 : ax[i,j].text(100, -16, ['Empty', 'Bug Trap', 'Maze', 'Office'][j], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
            else : ax[i,j].text(50, -8, ['Empty', 'Bug Trap', 'Maze', 'Office'][j], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
            ax[i,j].text(start_positions[j][0], start_positions[j][1], "S", size=9, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
            for k, (x, y) in enumerate(goal_positions[j]):
                ax[i,j].text(x, y, ""+str(k+1), size=8, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
            ax[i,j].axis('off')
"""






#np.save("iterations.npy", iterations)

'''
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.08, 'hspace':0.105, 'left':0.08, 'right':0.92, 'top':0.994, 'bottom':0.08}, figsize=(7, 4.8))
for i in range(2):
    for j in range(2):
        ax[i, j].plot(range(coverages.shape[2]), coverages[2*i+j,0], '#400082')
        ax[i, j].plot(range(coverages.shape[2]), coverages[2*i+j,1], '#00bdaa')
        ax[i, j].plot(range(coverages.shape[2]), coverages[2*i+j,3], '#ffc800')
        if i == 0 and j == 0 : ax[i, j].plot(range(coverages.shape[2]), coverages[2*i+j,2], '#ff0000', linestyle=":")
        else : ax[i, j].plot(range(coverages.shape[2]), coverages[2*i+j,2], '#ff0000')
        ax[i, j].set_xticks([0,50,100,150,200,250])
        ax[i, j].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax[i, j].text(0.1, 0.964, "ABCD"[2*i+j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
fig.text(0.5, 0.011, 'Number of iterations', ha='center', va='center')
fig.text(0.015, 0.5, 'Fraction of map reachable', ha='center', va='center', rotation='vertical')
fig.legend((matplotlib.lines.Line2D([], [], color='#400082'), matplotlib.lines.Line2D([], [], color='#00bdaa'), matplotlib.lines.Line2D([], [], color='#ff0000'), matplotlib.lines.Line2D([], [], color='#FFE100')),
    ('RT-RRT*', 'RT-RRT*(D)', 'LM-RRT*(E)', 'LM-RRT*(D)'), loc=(0.755,0.08), framealpha=0, markerfirst=False, labelspacing=0.2, fontsize=8)
plt.savefig('coverages.pgf')
plt.show()
'''
'''
labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.08, 'hspace':0.105, 'left':0.08, 'right':0.92, 'top':0.994, 'bottom':0.08}, figsize=(7, 4.8))
for i in range(2):
    for j in range(2):
        bars1 = ax[i, j].bar(x - 1.5*width, distances[2*i+j,0], width, label='rtrrt', color='#400082')
        bars2 = ax[i, j].bar(x - 0.5*width, distances[2*i+j,1], width, label='dmrrt', color='#00bdaa')
        bars3 = ax[i, j].bar(x + 0.5*width, distances[2*i+j,2], width, label='emrrt', color='#ff0000')
        bars4 = ax[i, j].bar(x + 1.5*width, distances[2*i+j,3], width, label='emrrt', color='#ffc800')
        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels(labels)
        ax[i, j].autoscale(False)
        if i == 0:
            ax[i, j].set_ylim(0,200*1.05)
            ax[i, j].text(5.3, 200*0.99, "ABCD"[2*i+j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)), fontsize=10)
        else:
            ax[i, j].set_ylim(0,300*1.05)
            ax[i, j].text(5.3, 300*0.99, "ABCD"[2*i+j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)), fontsize=10)
fig.text(0.5, 0.013, 'Goal number', ha='center', va='center', fontsize=12)
fig.text(0.015, 0.5, 'Distance of path to goal', ha='center', va='center', rotation='vertical', fontsize=12)
fig.legend((matplotlib.lines.Line2D([], [], color='#400082'), matplotlib.lines.Line2D([], [], color='#00bdaa'), matplotlib.lines.Line2D([], [], color='#ff0000'), matplotlib.lines.Line2D([], [], color='#FFE100')),
    ('RT-RRT*', 'RT-RRT*(D)', 'LM-RRT*(E)', 'LM-RRT*(D)'), loc=(0.087,0.838), framealpha=0, markerfirst=True, labelspacing=0.2, fontsize=10)
plt.savefig('distances.pgf')
plt.show()


labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.08, 'hspace':0.105, 'left':0.08, 'right':0.92, 'top':0.994, 'bottom':0.08}, figsize=(7, 4.8))
for i in range(2):
    for j in range(2):
        bars1 = ax[i, j].bar(x - 1.5*width, iterations[2*i+j,0], width, label='rtrrt', color='#400082')
        bars2 = ax[i, j].bar(x - 0.5*width, iterations[2*i+j,1], width, label='dmrrt', color='#00bdaa')
        bars3 = ax[i, j].bar(x + 0.5*width, iterations[2*i+j,2], width, label='emrrt', color='#ff0000')
        bars4 = ax[i, j].bar(x + 1.5*width, iterations[2*i+j,3], width, label='emrrt', color='#ffc800')
        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels(labels)
        ax[i, j].set_yscale("log")
        ax[i, j].autoscale(False)
        if i == 0:
            ax[i, j].set_ylim(0.1,100*1.4)
            ax[i, j].set_yticks([0.1, 1, 10, 100])
            ax[i, j].text(5.3, 97, "ABCD"[2*i+j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)), fontsize=10)
        else:
            ax[i, j].set_ylim(0.1,1000*1.75)
            ax[i, j].text(5.3, 1090, "ABCD"[2*i+j], size=8.5, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)), fontsize=10)
fig.text(0.5, 0.013, 'Goal number', ha='center', va='center', fontsize=12)
fig.text(0.015, 0.5, 'Number of iterations to find path to goal', ha='center', va='center', rotation='vertical', fontsize=12)
fig.legend((matplotlib.lines.Line2D([], [], color='#400082'), matplotlib.lines.Line2D([], [], color='#00bdaa'), matplotlib.lines.Line2D([], [], color='#ff0000'), matplotlib.lines.Line2D([], [], color='#FFE100')),
    ('RT-RRT*', 'RT-RRT*(D)', 'LM-RRT*(E)', 'LM-RRT*(D)'), loc=(0.087,0.838), framealpha=0, markerfirst=True, labelspacing=0.2, fontsize=10)
plt.savefig('iterations.pgf')
plt.show()
'''
'''
fig, ax = plt.subplots(1, 4, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.08, 'hspace':0.08, 'left':0.001, 'right':0.999, 'top':0.9, 'bottom':0.001}, figsize=(12, 3.2))
start_positions = [np.array([50.0, 50.0]), np.array([63.0, 51.0]), np.array([35.0, 33.0]), np.array([79.0/2, 65.0/2])]
goal_positions = np.array([[[13.0, 33.2], [15.0, 78.0], [92.4, 8.2], [92.6, 93.6], [32.1, 7.0], [4.9, 56.0]],
                           [[7.2, 6.6], [28.0, 81.0], [35.0, 37.0], [93.0, 91.0], [89.2,  7.0], [16.2, 37.2]],
                           [[11.2, 49.2], [35.0, 70.0], [90.0, 59.2], [8.4, 8.2], [53.0, 90.0], [90.0, 47.0]],
                           [[125.6/2, 141.6/2], [162.6/2, 13.4/2], [170.8/2, 122.0/2], [60.0/2, 185.8/2], [23.8/2, 48.6/2], [185.0/2, 178.2/2]]])
for i in range(4):
    ax[i].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/Path Planning/bm/' + ['empty', 'maze', 'bug_trap', 'office'][i] + '.png')), zorder=-1)
    ax[i].text(50, -6, ['Empty (A)', 'Maze (B)', 'Bug Trap (C)', 'Office (D)'][i], size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,0,0,0)))
    ax[i].text(start_positions[i][0], start_positions[i][1], "S", size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    for k, (x, y) in enumerate(goal_positions[i]):
        ax[i].text(x, y, ""+str(k+1), size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(1,200/255,0,0)))
    ax[i].axis('off')
plt.savefig('images.pgf')
plt.show()
'''

"""fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', subplot_kw={'xmargin': 0.05, 'ymargin':0.05}, gridspec_kw={'wspace':0.3, 'hspace':0.1, 'left':0, 'right':1, 'top':1, 'bottom':0}, figsize=(12.4, 2.7))
start_position = np.array([25.0, 85.0])*10
goal_position = np.array([83.0, 20.0])*10
for i in range(3):
    ax[i].imshow(plt.imread(get_sample_data('/Users/danarmstrong/Desktop/AM-RRT*/' + ['rtrrt_e', 'amrrt_e', 'amrrt_d'][i] + '.png')), zorder=-1)
    ax[i].text(start_position[0], start_position[1], "A", size=8, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,189/255,170/255)))
    ax[i].text(goal_position[0], goal_position[1], "G", size=8, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0,0,0,0), fc=(0,189/255,170/255)))
    ax[i].axis('off')
plt.savefig('rewires.pgf')
plt.show()


"""

'''print(((distances[:,0] - distances[:,3]) / distances[:,3]).mean()*100)
print(((distances[:,0] - distances[:,2]) / distances[:,2]).mean()*100)
print(((distances[:,0] - distances[:,1]) / distances[:,1]).mean()*100)
print(((iterations[:,0] - iterations[:,3]) / iterations[:,3]).mean()*100)
print(((iterations[:,0] - iterations[:,2]) / iterations[:,2]).mean()*100)
print(((iterations[:,0] - iterations[:,1]) / iterations[:,1]).mean()*100)'''

'''rtd_dist = ((distances[:,0] - distances[:,1]) / distances[:,1]).mean()
lme_dist = ((distances[:,0] - distances[:,2]) / distances[:,2]).mean()
lmd_dist = ((distances[:,0] - distances[:,3]) / distances[:,3]).mean()
lmd_dist_office = ((distances[3,0] - distances[3,3]) / distances[3,3]).mean()
rtd_iter = ((iterations[:,0] - iterations[:,1]) / iterations[:,1]).mean()
lme_iter = ((iterations[:,0] - iterations[:,2]) / iterations[:,2]).mean()
lmd_iter = ((iterations[:,0] - iterations[:,3]) / iterations[:,3]).mean()
lmd_iter_office = ((iterations[3,0] - iterations[3,3]) / iterations[3,3]).mean()

print((distances[:,1] / distances[:,0]).mean())
print((distances[:,2] / distances[:,0]).mean())
print((iterations[:,2] / iterations[:,0]).mean())

print()

print(((iterations[:,0] - iterations[:,2]) / iterations[:,2]).mean())

print(rtd_dist/(rtd_dist+1)*100)
print(lme_dist/(lme_dist+1)*100)
print(lmd_dist/(lmd_dist+1)*100)
print(rtd_iter/(rtd_iter+1)*100)
print(lme_iter/(lme_iter+1)*100)
print(lmd_iter/(lmd_iter+1)*100)
'''
