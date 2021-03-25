#Copyright (c) 2020 Ocado. All Rights Reserved.

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from amrrt.space import StateSpace


def within_ellipse(pos, fa, fb, major, minor):
    c = (fa + fb) / 2
    r = np.arctan2((fa-fb)[1], (fa-fb)[0])
    return (((np.cos(r)*(pos[0]-c[0]) + np.sin(r)*(pos[1]-c[1]))/major)**2 +
            ((np.sin(r)*(pos[0]-c[0]) - np.cos(r)*(pos[1]-c[1]))/minor)**2 <= 1)


def test_uniform_choice(empty_environment_info):
    space = empty_environment_info["space"]
    states = [space.choose_state_uniform() for _ in range(50)]
    freeness = [space.free_position(state.pos) for state in states]
    assert all(freeness)


def test_line_choice(empty_environment_info):
    space = empty_environment_info["space"]
    az = [space.choose_state_uniform().pos for _ in range(50)]
    bz = [space.choose_state_uniform().pos for _ in range(50)]
    states = [space.choose_state_line(a, b) for a, b in zip(az, bz)]
    cz = [state.pos for state in states]
    freeness = [space.free_position(state.pos) for state in states]
    on_line = [a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]) < 0.000001 for a, b, c in zip(az, bz, cz)]
    assert all(freeness) and False not in on_line


def test_ellipse_choice(empty_environment_info):
    space = empty_environment_info["space"]
    az = [space.choose_state_uniform().pos for _ in range(50)]
    bz = [space.choose_state_uniform().pos for _ in range(50)]
    mis = [np.linalg.norm(b-a) for a, b in zip(az, bz)]
    mas = [mi for mi in mis]
    states = [space.choose_state_ellipse(a, b, ma, mi) for a, b, ma, mi in zip(az, bz, mas, mis)]
    freeness = [space.free_position(state.pos) for state in states]
    in_ellipse = [within_ellipse(state.pos, a, b, ma, mi) for state, a, b, ma, mi in zip(states, az, bz, mas, mis)]
    assert all(freeness) and False not in in_ellipse


def test_free_position(empty_environment_info):
    space = empty_environment_info["space"]
    assert space.free_position(np.array([10,10])) and not space.free_position(np.array([0,0]))


def test_free_path(empty_environment_info):
    space = empty_environment_info["space"]
    a = space.create_state(np.array([0,0]))
    b = space.create_state(np.array([10,10]))
    c = space.create_state(np.array([20,20]))
    assert not space.free_path(a,b) and space.free_path(b,c)


def test_dynamic(empty_environment_info):
    space = StateSpace.from_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "empty.png"))
    space.add_dynamic_obstacle(np.array([10,10]), 5)
    assert (space.dynamically_free_position(np.array([4,10]))
            and not space.dynamically_free_position(np.array([7,10])))
