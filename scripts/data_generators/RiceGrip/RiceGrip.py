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
des_dir = 'RiceGrip2'
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
    d = np.random.rand() * 0.1 + 0.7    # (0.6, 0.8)
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

    states[0, :3] = np.array([e_0_curr[0], 0.6, e_0_curr[1]])
    states[0, 3:6] = np.array([e_0_last[0], 0.6, e_0_last[1]])
    states[0, 6:10] = quat
    states[0, 10:14] = quat

    states[1, :3] = np.array([e_1_curr[0], 0.6, e_1_curr[1]])
    states[1, 3:6] = np.array([e_1_last[0], 0.6, e_1_last[1]])
    states[1, 6:10] = quat
    states[1, 10:14] = quat

    return states


def visualize_point_cloud(positions, idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], alpha = 0.5)
    ax.set_aspect('equal')
    plt.show()


def calc_surface_idx(positions):
    point_tree = spatial.cKDTree(positions)
    neighbors = point_tree.query(positions, 40, n_jobs=-1)[1]
    surface_mask = np.zeros(positions.shape[0])

    pca = PCA(n_components=3)

    for i in range(len(neighbors)):
        pca.fit(positions[neighbors[i]])
        # print(i, pca.explained_variance_ratio_)
        if pca.explained_variance_ratio_[0] > 0.45:
            surface_mask[i] = 1

    surface_idx = np.nonzero(surface_mask)[0]

    print("surface idx", surface_idx.shape)

    return surface_idx


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

    halfEdge = np.array([0.15, 0.8, 0.15])
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])

    pyflex.add_box(halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)


    ### read scene info
    # print("Scene Upper:", pyflex.get_scene_upper())
    # print("Scene Lower:", pyflex.get_scene_lower())
    # print("Num particles:", pyflex.get_phases().reshape(-1, 1).shape[0])
    # print("Phases:", np.unique(pyflex.get_phases()))

    n_particles = pyflex.get_n_particles()
    n_shapes = pyflex.get_n_shapes()
    n_rigids = pyflex.get_n_rigids()
    n_rigidPositions = pyflex.get_n_rigidPositions()

    # print("n_particles", n_particles)
    # print("n_shapes", n_shapes)
    # print("n_rigids", n_rigids)
    # print("n_rigidPositions", n_rigidPositions)

    positions = np.zeros((grip_time, time_step, n_particles, dim_position))
    restPositions = np.zeros((grip_time, time_step, n_particles, dim_position))
    velocities = np.zeros((grip_time, time_step, n_particles, dim_velocity))
    shape_states = np.zeros((grip_time, time_step, n_shapes, dim_shape_state))

    rigid_offsets = np.zeros((grip_time, time_step, n_rigids + 1, 1), dtype=np.int)
    rigid_indices = np.zeros((grip_time, time_step, n_rigidPositions, 1), dtype=np.int)
    rigid_localPositions = np.zeros((grip_time, time_step, n_rigidPositions, 3))
    rigid_globalPositions = np.zeros((grip_time, time_step, n_particles, 3))
    rigid_translations = np.zeros((grip_time, time_step, n_rigids, 3))
    rigid_rotations = np.zeros((grip_time, time_step, n_rigids, 4))

    data_positions = np.zeros((grip_time, time_step, n_particles, 6))
    data_velocities = np.zeros((grip_time, time_step, n_particles, 6))


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

    for r in range(grip_time):
        gripper_config = sample_gripper_config()
        # gripper_config = saved[r]
        # print(gripper_config)
        for i in range(time_step):
            shape_states_ = calc_shape_states(i * dt, gripper_config)
            pyflex.set_shape_states(shape_states_)

            positions[r, i] = pyflex.get_positions().reshape(-1, dim_position)

            if i == 0:
                surface_idx = calc_surface_idx(positions[r, i, :, :3])
                # visualize_point_cloud(positions[i, :, :3], surface_idx)

            restPositions[r, i] = pyflex.get_restPositions().reshape(-1, dim_position)
            velocities[r, i] = pyflex.get_velocities().reshape(-1, dim_velocity)
            shape_states[r, i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)

            rigid_offsets[r, i] = pyflex.get_rigidOffsets().reshape(-1, 1)
            rigid_indices[r, i] = pyflex.get_rigidIndices().reshape(-1, 1)
            rigid_localPositions[r, i] = pyflex.get_rigidLocalPositions().reshape(-1, 3)
            rigid_globalPositions[r, i] = pyflex.get_rigidGlobalPositions().reshape(-1, 3)
            rigid_rotations[r, i] = pyflex.get_rigidRotations().reshape(-1, 4)
            rigid_translations[r, i] = pyflex.get_rigidTranslations().reshape(-1, 3)


            data_positions[r, i, :, :3] = rigid_globalPositions[r, i]
            data_positions[r, i, :, 3:] = positions[r, i][:, :3]

            if i > 0:
            	data_velocities[r, i] = (data_positions[r, i] - data_positions[r, i-1]) / dt


            if SAVE_VIDEO:
                pyflex.render(capture=1, path=os.path.join('tmp', 'render_%d.tga' % (r * time_step + i)))

            # print(pyflex.get_sceneParams())
            # print(rigid_rotations[r, i])

            pyflex.step()

        # for i in range(time_step - 1):
        #     idx = 10
        #     cnt = 0
        #     for j in range(n_rigids):
        #         st, ed = rigid_offsets[r, i, j, 0], rigid_offsets[r, i, j + 1, 0]
        #         for k in range(st, ed):
        #             if rigid_indices[r, i, cnt, 0] == idx:
        #                 print(i, j, positions[r, i, rigid_indices[r, i, cnt, 0], :3],
        #                       rigid_globalPositions[r, i, rigid_indices[r, i, cnt, 0]],
        #                       rotate(rigid_localPositions[r, i, cnt], rigid_rotations[r, i, j]) + \
        #                       rigid_translations[r, i, j])
        #             cnt += 1


    # print(rigid_indices[r, i])
    # print(rigid_indices[r, i].shape)

    # print(rigid_globalPositions.shape)
    # print(rigid_rotations.shape)
    # print(rigid_localPositions.shape)
    # print(restPositions.shape)
    # print(rigid_translations.shape)

    # print(positions[0].std(axis=0).mean(axis=0))
    # print(rigid_globalPositions[0].std(axis=0).mean(axis=0))
    # print(rigid_localPositions[0].std(axis=0).mean(axis=0))
    # print(restPositions[0].std(axis=0).mean(axis=0))

    # print(data_positions[0, 0].mean(axis=0))
    # print(data_positions[0, 15].mean(axis=0))
    # print(data_positions[0, 15].std(axis=0))
    # print(data_velocities[0, 15].mean(axis=0))
    # print(data_velocities[0, 15].std(axis=0))

    # print((data_positions[0, 15] - data_positions[0, 0]).std(axis=0))


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
