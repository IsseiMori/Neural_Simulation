import os
import numpy as np
import time
import torch
import sys

import cv2
from PIL import Image
import re


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def sample_gripper_config(data_type="RiceGrip", random=True, scene_info=None):

    if random:
        dis = np.random.rand() * 0.5
        angle = np.random.rand() * np.pi * 2
        d_rand = np.random.rand()
    else:
        dis = 0.5 * 0.5
        angle = 0.35 * np.pi * 2
        d_rand = 0.5

    if data_type == "RiceGrip":

        x = np.cos(angle) * dis
        z = np.sin(angle) * dis
        d = d_rand * 0.1 + 0.7    # (0.6, 0.8)
        return x, z, d

    elif data_type == "RiceGripMulti":

        width1 = 1.0 - (max(scene_info[0][0], scene_info[0][2]) - 0.1) / 0.25
        width2 = 1.0 - (max(scene_info[1][0], scene_info[1][2]) - 0.1) / 0.25
        width_max = max(width1, width2)

        x = np.cos(angle) * dis
        y1 = 0.8 + max(0, 0.8 * np.random.rand() - 0.2) # horizontal press 1
        y2 = 0.8 + max(0, 0.8 * np.random.rand() - 0.2) # horizontal press 2
        z = np.sin(angle) * dis
        d = d_rand * 0.1 + 0.8 - width_max * 0.10   # (0.6, 0.8)
        d0 = d_rand * 0.4 + 0.4    # (0.6, 0.8)
        d1 = d_rand * 0.2 + 0.05 # vertical press 1
        d2 = d_rand * 0.2 + 0.05 # vertical press 2
        c = True if np.random.rand() < 0.5 else False
        return x, z, d, y1, y2, d0, d1, d2, c

    elif data_type == "PressDown":

        x = np.cos(angle) * dis
        z = np.sin(angle) * dis
        d = d_rand * 0.2 + 0.05    # (0.6, 0.8)
        return x, z, d

    elif data_type == "BendTube":

        x = np.cos(angle) * dis
        z = np.sin(angle) * dis
        d = d_rand * 0.1 + 0.7    # (0.6, 0.8)
        return x, z, d

    elif data_type == "CompressTube":

        x = np.cos(angle) * dis
        z = np.sin(angle) * dis
        d = d_rand * 1.0    # (0.6, 0.8)
        return x, z, d

    else:

        sys.exit('Invalid scene type')




def calc_shape_states(t, gripper_config, dim_shape_state, dt, data_type):

    if data_type == "RiceGrip":

        rest_gripper_dis = 1.8

        x, z, d = gripper_config
        s = (rest_gripper_dis - d) / 2.
        half_rest_gripper_dis = rest_gripper_dis / 2.

        time = max(0., t) * 5
        lastTime = max(0., t - dt) * 5

        states = np.zeros((2, dim_shape_state))

        dis = np.sqrt(x**2 + z**2)
        angle = np.array([-z / dis, x / dis])
        quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))

        e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
        e_1 = np.array([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

        e_0_curr = e_0 + angle * np.sin(time) * s
        e_1_curr = e_1 - angle * np.sin(time) * s
        e_0_last = e_0 + angle * np.sin(lastTime) * s
        e_1_last = e_1 - angle * np.sin(lastTime) * s

        states[0, :3] = np.array([e_0_curr[0], 0.8, e_0_curr[1]])
        states[0, 3:6] = np.array([e_0_last[0], 0.8, e_0_last[1]])
        states[0, 6:10] = quat
        states[0, 10:14] = quat

        states[1, :3] = np.array([e_1_curr[0], 0.8, e_1_curr[1]])
        states[1, 3:6] = np.array([e_1_last[0], 0.8, e_1_last[1]])
        states[1, 6:10] = quat
        states[1, 10:14] = quat

        return states

    elif data_type == "RiceGripMulti":

        x, z, d, y1, y2, d0, d1, d2, c = gripper_config

        if c:

            rest_gripper_dis = 2.2

            s = (rest_gripper_dis - d) / 2.
            half_rest_gripper_dis = rest_gripper_dis / 2.

            time = max(0., t) * 5
            lastTime = max(0., t - dt) * 5

            states = np.zeros((2, dim_shape_state))

            dis = np.sqrt(x**2 + z**2)
            angle = np.array([-z / dis, x / dis])
            quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))

            e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
            e_1 = np.array([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

            e_0_curr = e_0 + angle * np.sin(time) * s
            e_1_curr = e_1 - angle * np.sin(time) * s
            e_0_last = e_0 + angle * np.sin(lastTime) * s
            e_1_last = e_1 - angle * np.sin(lastTime) * s

            states[0, :3] = np.array([e_0_curr[0], y1, e_0_curr[1]])
            states[0, 3:6] = np.array([e_0_last[0], y1, e_0_last[1]])
            states[0, 6:10] = quat
            states[0, 10:14] = quat

            states[1, :3] = np.array([e_1_curr[0], y2, e_1_curr[1]])
            states[1, 3:6] = np.array([e_1_last[0], y2, e_1_last[1]])
            states[1, 6:10] = quat
            states[1, 10:14] = quat

        else:

            rest_gripper_dis_grip = 1.2
            rest_gripper_dis_press = 1.8

            s = (rest_gripper_dis_grip - d0) / 2.
            half_rest_gripper_dis = rest_gripper_dis_grip / 2.

            s1 = (rest_gripper_dis_press - d1) / 2.
            s2 = (rest_gripper_dis_press - d2) / 2.

            time = max(0., t) * 5
            lastTime = max(0., t - dt) * 5

            states = np.zeros((2, dim_shape_state))

            dis = np.sqrt(x**2 + z**2)
            angle = np.array([-z / dis, x / dis])
            quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))

            e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
            e_1 = np.array([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

            e_0_curr = e_0 + angle * 0.5 * s
            e_1_curr = e_1 - angle * 0.5 * s
            e_0_last = e_0 + angle * 0.5 * s
            e_1_last = e_1 - angle * 0.5 * s

            y0_curr = 1 * np.sin(time) * s1
            y0_last = 1 * np.sin(lastTime) * s1
            y1_curr = 1 * np.sin(time) * s2
            y1_last = 1 * np.sin(lastTime) * s2

            states[0, :3] = np.array([e_0_curr[0], 2. - y0_curr, e_0_curr[1]])
            states[0, 3:6] = np.array([e_0_last[0], 2. - y0_last, e_0_last[1]])
            states[0, 6:10] = quat
            states[0, 10:14] = quat

            states[1, :3] = np.array([e_1_curr[0], 2. - y1_curr, e_1_curr[1]])
            states[1, 3:6] = np.array([e_1_last[0], 2. - y1_last, e_1_last[1]])
            states[1, 6:10] = quat
            states[1, 10:14] = quat

        return states

    elif data_type == "PressDown":

        rest_gripper_dis = 1.8

        x, z, d = gripper_config
        s = (rest_gripper_dis - d) / 2.
        half_rest_gripper_dis = rest_gripper_dis / 2.

        time = max(0., t) * 5
        lastTime = max(0., t - dt) * 5

        states = np.zeros((1, dim_shape_state))

        dis = np.sqrt(x**2 + z**2)
        angle = np.array([-z / dis, x / dis])
        angle = np.array([np.abs(x / dis)])
        quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))
        
        e_0 = np.array([0])

        e_0_curr = e_0 + 1 * np.sin(time) * s
        e_0_last = e_0 + 1 * np.sin(lastTime) * s

        states[0, :3] = np.array([x, 1.5 - e_0_curr[0], z])
        states[0, 3:6] = np.array([x, 1.5 - e_0_last[0], z])
        states[0, 6:10] = quat
        states[0, 10:14] = quat


        return states

    elif data_type == "BendTube":

        rest_gripper_dis = 1.5

        x, z, d = gripper_config
        s = (rest_gripper_dis - d) / 2.
        half_rest_gripper_dis = rest_gripper_dis / 2.

        time = max(0., t) * 4
        lastTime = max(0., t - dt) * 4

        states = np.zeros((3, dim_shape_state))

        dis = np.sqrt(x**2 + z**2)
        angle = np.array([-z / dis, x / dis])
        quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))
        quat = quatFromAxisAngle(np.array([0., 1., 0.]), 0)

        e_0 = np.array([0])
        e_1 = np.array([0])
        e_2 = np.array([0])

        e_0_curr = e_0 + 1 * np.cos(time) * s
        e_1_curr = e_1 - 1 * np.cos(time) * s
        e_2_curr = e_2 - 1 * np.cos(time) * s
        e_0_last = e_0 + 1 * np.cos(lastTime) * s
        e_1_last = e_1 - 1 * np.cos(lastTime) * s
        e_2_last = e_2 - 1 * np.cos(lastTime) * s

        states[0, :3] = np.array([0.0, 0.4, e_0_curr[0]])
        states[0, 3:6] = np.array([0.0, 0.4, e_0_last[0]])
        states[0, 6:10] = quat
        states[0, 10:14] = quat

        states[1, :3] = np.array([1.0, 0.4, e_1_curr[0]])
        states[1, 3:6] = np.array([1.0, 0.4, e_1_last[0]])
        states[1, 6:10] = quat
        states[1, 10:14] = quat

        states[2, :3] = np.array([-1.0, 0.4, e_2_curr[0]])
        states[2, 3:6] = np.array([-1.0, 0.4, e_2_last[0]])
        states[2, 6:10] = quat
        states[2, 10:14] = quat

        return states

    elif data_type == "CompressTube":

        rest_gripper_dis = 1.9

        x, z, d = gripper_config
        s = (rest_gripper_dis - d) / 2.
        half_rest_gripper_dis = rest_gripper_dis / 2.

        time = max(0., t) * 2
        lastTime = max(0., t - dt) * 2

        states = np.zeros((3, dim_shape_state))

        dis = np.sqrt(x**2 + z**2)
        angle = np.array([-z / dis, x / dis])
        quat = quatFromAxisAngle(np.array([0., 1., 0.]), 0)

        e_0 = np.array([-1.4])
        e_1 = np.array([1.4])

        e_0_curr = e_0 + 1 * np.sin(time) * s
        e_1_curr = e_1 - 1 * np.sin(time) * s
        e_0_last = e_0 + 1 * np.sin(lastTime) * s
        e_1_last = e_1 - 1 * np.sin(lastTime) * s

        states[0, :3] = np.array([e_0_curr[0], 0.4, 0])
        states[0, 3:6] = np.array([e_0_curr[0], 0.4, 0])
        states[0, 6:10] = quat
        states[0, 10:14] = quat

        states[1, :3] = np.array([e_1_curr[0], 0.4, 0])
        states[1, 3:6] = np.array([e_1_curr[0], 0.4, 0])
        states[1, 6:10] = quat
        states[1, 10:14] = quat

        return states

    else:
        
        sys.exit('Invalid scene type')



def init_scene(pyflex, data_type, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep):

    if data_type == "RiceGrip":
        # x = 10
        # y = 10
        # z = 10

        x = rand_float(8.0, 10.0)
        y = rand_float(8.0, 10.0)
        z = rand_float(8.0, 10.0)

        scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
        pyflex.set_scene(5, scene_params, 0)

        halfEdge = np.array([0.15, 0.8, 0.15])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)

        return np.stack((halfEdge, halfEdge))

    elif data_type == "RiceGripMulti":
        # x = 10
        # y = 10
        # z = 10

        x = rand_float(8.0, 10.0)
        y = rand_float(8.0, 10.0)
        z = rand_float(8.0, 10.0)


        scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
        pyflex.set_scene(5, scene_params, 0)

        # np.random.rand()

        halfEdge1 = np.array([0.05 + 0.3 * np.random.rand(), 0.8, 0.05 + 0.3 * np.random.rand()])
        halfEdge2 = np.array([0.05 + 0.3 * np.random.rand(), 0.8, 0.05 + 0.3 * np.random.rand()])

        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge1, center, quat)
        pyflex.add_box(halfEdge2, center, quat)

        return np.stack((halfEdge1, halfEdge2))


    elif data_type == "PressDown":

        x = 10
        y = 10
        z = 10

        scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
        pyflex.set_scene(5, scene_params, 0)

        halfEdge = np.array([0.40, 0.10, 0.40])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge, center, quat)

        return halfEdge

    elif data_type == "BendTube":

        x = 30
        y = 5
        z = 5

        scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
        pyflex.set_scene(5, scene_params, 0)


        halfEdge = np.array([0.15, 0.4, 0.15])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)

        return halfEdge

    elif data_type == "CompressTube":

        x = 20
        y = 5
        z = 5

        scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
        pyflex.set_scene(5, scene_params, 0)

        halfEdge = np.array([0.15, 0.4, 0.4])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)

        return halfEdge

    else:

        sys.exit('Invalid scene type')


def init_scene_plb(env, data_type):

    if data_type == "RiceGripMulti":

        x = rand_float(8.0, 10.0)
        y = rand_float(8.0, 10.0)
        z = rand_float(8.0, 10.0)

        # halfEdge1 = np.array([0.05 + 0.3 * np.random.rand(), 0.8, 0.05 + 0.3 * np.random.rand()])
        # halfEdge2 = np.array([0.05 + 0.3 * np.random.rand(), 0.8, 0.05 + 0.3 * np.random.rand()])

        halfEdge1 = np.array([0.15 + 0.20 * np.random.rand(), 0.8, 0.15 + 0.20 * np.random.rand()])
        halfEdge2 = np.array([0.15 + 0.20 * np.random.rand(), 0.8, 0.15 + 0.20 * np.random.rand()])

        # halfEdge1 = np.array([0.1 + 0.05, 0.8, 0.1 + 0.25])
        # halfEdge2 = np.array([0.1 + 0.05, 0.8, 0.1 + 0.25])

        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        halfEdge1 /= 5
        halfEdge2 /= 5

        env.primitives.primitives[0].size[None] = tuple(halfEdge1.tolist())
        env.primitives.primitives[1].size[None] = tuple(halfEdge2.tolist())

        return np.stack((halfEdge1, halfEdge2))