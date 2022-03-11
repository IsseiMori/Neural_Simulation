import os
import numpy as np
import pyflex
import time
import torch

import scipy.spatial as spatial
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import re



dt = 1. / 60.
des_dir = 'PressDown'
os.system('mkdir -p ' + des_dir)
SAVE_VIDEO = True

if SAVE_VIDEO:
    os.makedirs("tmp", exist_ok=True)

# n_particles = 768
# n_shapes = 2
# n_rigidPositions = 2613
# n_rigids = 4
# np.random.seed(0)

grip_time = 1
time_step = 40
dim_position = 4
dim_velocity = 3
dim_shape_state = 14
rest_gripper_dis = 1.8


def sample_gripper_config():
    dis = np.random.rand() * 0.5
    angle = np.random.rand() * np.pi * 2.
    x = np.cos(angle) * dis
    z = np.sin(angle) * dis
    d = np.random.rand() * 0.2 + 0.05    # (0.6, 0.8)
    return x, z, d

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def calc_shape_states(t, gripper_config):
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
    # quat = quatFromAxisAngle(np.array([0., 1., 0.]), 0)

    # e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
    e_0 = np.array([0])

    e_0_curr = e_0 + 1 * np.sin(time) * s
    e_0_last = e_0 + 1 * np.sin(lastTime) * s

    states[0, :3] = np.array([x, 1.5 - e_0_curr[0], z])
    states[0, 3:6] = np.array([x, 1.5 - e_0_last[0], z])
    states[0, 6:10] = quat
    states[0, 10:14] = quat


    return states



pyflex.init()

use_gpu = torch.cuda.is_available()

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


for data_i in range(0, 1):

    ### set scene
    # x, y, z: [8.0, 10.0]
    # clusterStiffness: [0.4, 0.8]
    # clusterPlasticThreshold: [0.000005, 0.0001]
    # clusterPlasticCreep: [0.1, 0.3]
    x = 10
    y = 10
    z = 10
    clusterStiffness = rand_float(0.3, 0.7)
    clusterPlasticThreshold = rand_float(0.00001, 0.0005)
    clusterPlasticCreep = rand_float(0.1, 0.3)

    # clusterStiffness = 0.7
    # clusterPlasticThreshold = 0.1
    # clusterPlasticCreep = 0.3


    scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
    pyflex.set_scene(5, scene_params, 0)

    halfEdge = np.array([0.40, 0.10, 0.40])
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])

    pyflex.add_box(halfEdge, center, quat)
    # pyflex.add_box(halfEdge, center, quat)


    ### read scene info
    # print("Scene Upper:", pyflex.get_scene_upper())
    # print("Scene Lower:", pyflex.get_scene_lower())
    # print("Num particles:", pyflex.get_phases().reshape(-1, 1).shape[0])
    # print("Phases:", np.unique(pyflex.get_phases()))

    n_particles = pyflex.get_n_particles()
    n_shapes = pyflex.get_n_shapes()
    n_rigids = pyflex.get_n_rigids()
    n_rigidPositions = pyflex.get_n_rigidPositions()

    positions = np.zeros((grip_time, time_step, n_particles, dim_position))
    shape_states = np.zeros((grip_time, time_step, n_shapes, dim_shape_state))
    rigid_globalPositions = np.zeros((grip_time, time_step, n_particles, 3))
    data_positions = np.zeros((grip_time, time_step, n_particles, 6))


    for r in range(grip_time):
        gripper_config = sample_gripper_config()
        for i in range(time_step):
            shape_states_ = calc_shape_states(i * dt, gripper_config)
            pyflex.set_shape_states(shape_states_)

            positions[r, i] = pyflex.get_positions().reshape(-1, dim_position)
            shape_states[r, i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)
            rigid_globalPositions[r, i] = pyflex.get_rigidGlobalPositions().reshape(-1, 3)


            data_positions[r, i, :, :3] = rigid_globalPositions[r, i]
            data_positions[r, i, :, 3:] = positions[r, i][:, :3]


            if SAVE_VIDEO:
                pyflex.render(capture=1, path=os.path.join('tmp', 'render_%d.tga' % (r * time_step + i)))

            pyflex.step()


    states = {
        'positions': data_positions,
        'shape_states': shape_states,
        'clusterStiffness': clusterStiffness, 
        'clusterPlasticThreshold': clusterPlasticThreshold, 
        'clusterPlasticCreep': clusterPlasticCreep
        }

    with open(os.path.join(des_dir, '{:0>4}.npy'.format(str(data_i))), 'wb') as f:
        np.save(f, states)


    if SAVE_VIDEO:
        image_folder = 'tmp'
        video_name = os.path.join(des_dir, '{:0>4}.avi'.format(str(data_i)))

        images = [img for img in os.listdir(image_folder) if img.endswith(".tga")]
        images.sort(key = lambda f: int(re.sub('\D', '', f)))


        im = Image.open(os.path.join(image_folder, images[0])) 
        nimg = np.array(im)
        frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 60, (width,height))

        for image in images:
            im = Image.open(os.path.join(image_folder, image)) 
            nimg = np.array(im)
            frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            video.write(frame)

        cv2.destroyAllWindows()
        video.release()


pyflex.clean()
