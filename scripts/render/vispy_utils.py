import os
import time
import numpy as np
import pickle
import cv2
import argparse
import math

import vispy.scene
from vispy import app
from vispy.visuals import transforms

particle_size = 0.002
border = 0.025
height = 1.3
y_rotate_deg = -45.0

def y_rotate(obj, deg=y_rotate_deg):
    tr = vispy.visuals.transforms.MatrixTransform()
    tr.rotate(deg, (0, 1, 0))
    obj.transform = tr

def add_floor(v):
    # add floor
    floor_length = 3.0
    w, h, d = floor_length, floor_length, border
    b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
    y_rotate(b1)
    v.add(b1)

    # adjust position of box
    mesh_b1 = b1.mesh.mesh_data
    v1 = mesh_b1.get_vertices()
    c1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
    mesh_b1.set_vertices(np.add(v1, c1))

    mesh_border_b1 = b1.border.mesh_data
    vv1 = mesh_border_b1.get_vertices()
    cc1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
    mesh_border_b1.set_vertices(np.add(vv1, cc1))

def update_box_states(boxes, last_states, curr_states):
    v = curr_states[0] - last_states[0]
    if args.verbose_data:
        print("box states:", last_states, curr_states)
        print("box velocity:", v)

    tr = vispy.visuals.transforms.MatrixTransform()
    tr.rotate(y_rotate_deg, (0, 1, 0))

    for i, box in enumerate(boxes):
        # use v to update box translation
        trans = (curr_states[i][0], curr_states[i][1], curr_states[i][2])
        box.transform = tr * vispy.visuals.transforms.STTransform(translate=trans)

def translate_box(b, x, y, z):
    mesh_b = b.mesh.mesh_data
    v = mesh_b.get_vertices()
    c = np.array([x, y, z], dtype=np.float32)
    mesh_b.set_vertices(np.add(v, c))

    mesh_border_b = b.border.mesh_data
    vv = mesh_border_b.get_vertices()
    cc = np.array([x, y, z], dtype=np.float32)
    mesh_border_b.set_vertices(np.add(vv, cc))

def add_box(v, w=0.1, h=0.1, d=0.1, x=0.0, y=0.0, z=0.0):
    """
    Add a box object to the scene view
    :param v: view to which the box should be added
    :param w: width
    :param h: height
    :param d: depth
    :param x: x center
    :param y: y center
    :param z: z center
    :return: None
    """
    # render background box
    b = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
    y_rotate(b)
    v.add(b)

    # adjust position of box
    translate_box(b, x, y, z)

    return b

def calc_box_init(x, z):
    boxes = []

    # floor
    boxes.append([x, z, border, 0., -particle_size / 2, 0.])

    # left wall
    boxes.append([border, z, (height + border), -particle_size / 2, 0., 0.])

    # right wall
    boxes.append([border, z, (height + border), particle_size / 2, 0., 0.])

    # back wall
    boxes.append([(x + border * 2), border, (height + border)])

    # front wall (disabled when colored)
    # boxes.append([(x + border * 2), border, (height + border)])

    return boxes

def add_container(v, box_x, box_z):
    boxes = calc_box_init(box_x, box_z)
    visuals = []
    for b in boxes:
        if len(b) == 3:
            visual = add_box(v, b[0], b[1], b[2])
        elif len(b) == 6:
            visual = add_box(v, b[0], b[1], b[2], b[3], b[4], b[5])
        else:
            raise AssertionError("Input should be either length 3 or length 6")
        visuals.append(visual)
    return visuals

def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 0., 1., 1.]])[:n]


def rotate(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def particlify_box(center, half_edge, quat):
    
    pos = []

    # initial spacing
    offset_height = 0.02
    offset_width = 0.02
    
    half_width = half_edge[0] # assume width = depth
    half_height = half_edge[1]

    particle_count_height = math.ceil(half_height * 2 / offset_height)
    particle_count_width = math.ceil(half_width * 2 / offset_width)

    offset_height = half_height * 2 / particle_count_height
    offset_width = half_width * 2 / particle_count_width


    local_bottom_corner_pos = np.array([-half_width, -half_height, - half_width])


    for h in range(0, particle_count_height + 1):
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([offset_width * w, offset_height * h, 0]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([half_width * 2, offset_height * h, offset_width * w]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([half_width * 2 - offset_width * w, offset_height * h, half_width * 2]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([0, offset_height * h, half_width * 2 - offset_width * w]))

    for r in range(1, particle_count_width):
        for c in range(1, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([offset_width * r, half_height * 2, offset_width * c]))
            pos.append(local_bottom_corner_pos + np.array([offset_width * r, 0, offset_width * c]))
        

    pos = np.asarray(pos, dtype=np.float64)
    
    for i in range(len(pos)):
        pos[i] = rotate(pos[i], quat)

    pos[:,0] += center[0]
    pos[:,1] += center[1]
    pos[:,2] += center[2]
        
    # pos = np.concatenate((pos, np.ones([len(pos), 1])), 1)
    
    return pos

def add_grips(positions, shape_states, half_edge):
    pos_all = []
    for r in range(len(positions)):
        pos_grip_iter = []
        for i in range(len(positions[0])):

            pos_grips = []

            for i_grip in range(len(shape_states[r, i])):

                pos = shape_states[r, i][i_grip][0:3]
                quat = shape_states[r, i][i_grip][6:10]
                pos_grip = particlify_box(pos, half_edge, quat)

                pos_grips.append(pos_grip)

            pos_grips = np.array(pos_grips)
            pos_grips = pos_grips.reshape(-1, pos_grips.shape[-1])


            pos_grip_iter.append(np.concatenate((positions[r,i], pos_grips), 0))
        pos_all.append(pos_grip_iter)

    pos_all = np.asarray(pos_all, dtype=np.float64)

    return pos_all   