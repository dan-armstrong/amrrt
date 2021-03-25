# AM-RRT*

This package is the official release of AM-RRT*, a novel algorithm that extends RRT* and RT-RRT* for online path planning in large, complex and dynamic environments. The associated paper describing our new method can be found here: [LINK]. The package also includes an implementation of RT-RRT*, in Python.


## Installation

Installation requires Python 3.6 or higher. Install Python requirements by running `python3 setup.py install` in project directory. To check the installation was successful, run `pytest` in project directory (this takes some time - for a quicker check run `pytest --ignore tests/test_geodesic_metric.py`).


## Usage

In [/demos](demos) there are two scripts: [benchmarking.py](demos/benchmarking.py) tests the performance of AM-RRT* against RT-RRT* in a variety of environments, and [visualiser.py](demos/visualiser.py) provides a GUI to visualise the behaviour of either planner in a chosen environment.

The benchmarking can be run with the command (along with the optional parament --include_geodesic):
```bash
python3 demos/benchmarking.py [--include_geodesic=False]
```

This compares the 4 planners as described in the results section of our paper: RT-RRT*, RT-RRT*(D), AM-RRT*(E), AM-RRT*(D) - see below to include AM-RRT*(G). RT-RRT*(D) is RT-RRT* with diffusion distance used as the metric to find nearest neighbours instead of Euclidean distance. AM-RRT*(E), AM-RRT*(D) and AM-RRT*(G) are AM-RRT* planners using Euclidean, diffusion and geodesic distances respectively as the assisting metric. The results will be displayed in a similar graph to the one in our paper, and will only be displayed after the first repeat has been completed. The current progress is printed to the terminal.

To also include RT-RRT*(G) in the results, run (`python3 demos/benchmarking.py --include_geodesic=True`). Computing the pairwise potentials necessary for the geodesic metric is an extremely costly process, so expect to wait a significant amount of time (upwards of an hour when run on Macbook Pro with 2.6GHz 6‑core Intel Core i7) after running the command before benchmarking actually starts. The geodesic metric is only included to give an upper bound on the performance of AM-RRT*.

The visualiser can be run with the command:

```bash
python3 demos/visualiser.py IMAGE PLANNER METRIC
```

This takes three arguments: IMAGE is the filename of an image containing a 2D environment (it will be thresholded to form a grid where light pixels are free space and dark pixels obstacles), PLANNER (either 'rtrrt' or 'amrrt') selects the planner to use, and METRIC (either 'euclidean', 'diffusion' or 'geodesic') selects the assisting metric (for RT-RRT*, the assisting metric is used to choose nearest neighbours). Click anywhere on the map to set the agents position, and then click again (any number of times) to set the goal position. The tree, agent position and goal position are displayed in the environment. There are some sample images under demos/resources.


## Common Problems

Pygame displays a blank screen on some versions of MacOS - fixes can be found at https://stackoverflow.com/questions/52718921/problems-getting-pygame-to-show-anything-but-a-blank-screen-on-macos-mojave.
