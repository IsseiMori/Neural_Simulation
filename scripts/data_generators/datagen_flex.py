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

import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
parser.add_argument("--video", help="write video", action='store_true')
args = parser.parse_args()


dt = 1. / 60.
out_dir = args.out

os.system('mkdir -p ' + out_dir)

if args.video:
    os.makedirs("tmp", exist_ok=True)


grip_time = 1
dim_position = 4
dim_velocity = 3
dim_shape_state = 14

time_step = 40

if args.scene == "BendTube" or args.scene == "CompressTube":
    time_step = 80



pyflex.init()

use_gpu = torch.cuda.is_available()

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def simulate_scene(data_i, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep, data_name):

    # clusterStiffness = rand_float(0.3, 0.7)
    # clusterPlasticThreshold = rand_float(0.00001, 0.0005)
    # clusterPlasticCreep = rand_float(0.1, 0.3)

    # # clusterStiffness = 0.3
    # # clusterPlasticThreshold = 0.00001
    # # clusterPlasticCreep = 0.3
    # clusterStiffness = 0.7
    # clusterPlasticThreshold = 0.0005
    # clusterPlasticCreep = 0.1

    scene_info = init_scene(pyflex, args.scene, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep)


    n_particles = pyflex.get_n_particles()
    n_shapes = pyflex.get_n_shapes()
    n_rigids = pyflex.get_n_rigids()
    n_rigidPositions = pyflex.get_n_rigidPositions()

    positions = np.zeros((grip_time, time_step, n_particles, dim_position))
    shape_states = np.zeros((grip_time, time_step, n_shapes, dim_shape_state))
    rigid_globalPositions = np.zeros((grip_time, time_step, n_particles, 3))
    data_positions = np.zeros((grip_time, time_step, n_particles, 6))


    for r in range(grip_time):
        gripper_config = sample_gripper_config(args.scene, random=False)
        for i in range(time_step):
            shape_states_ = calc_shape_states(i * dt, gripper_config, dim_shape_state, dt, args.scene)
            pyflex.set_shape_states(shape_states_)

            positions[r, i] = pyflex.get_positions().reshape(-1, dim_position)
            shape_states[r, i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)
            rigid_globalPositions[r, i] = pyflex.get_rigidGlobalPositions().reshape(-1, 3)


            data_positions[r, i, :, :3] = rigid_globalPositions[r, i]
            data_positions[r, i, :, 3:] = positions[r, i][:, :3]


            if args.video:
                pyflex.render(capture=1, path=os.path.join('tmp', 'render_%d.tga' % (r * time_step + i)))

            pyflex.step()

    data_positions /= 5
    shape_states[:,:,:,0:6] /= 5

    data_positions[:,:,:,0] += 0.5
    data_positions[:,:,:,2] += 0.5
    data_positions[:,:,:,3] += 0.5
    data_positions[:,:,:,5] += 0.5

    shape_states[:,:,:,0] += 0.5
    shape_states[:,:,:,2] += 0.5
    shape_states[:,:,:,3] += 0.5
    shape_states[:,:,:,5] += 0.5

    scene_info /= 5

    states = {
        'positions': data_positions,
        'shape_states': shape_states,
        'clusterStiffness': clusterStiffness, 
        'clusterPlasticThreshold': clusterPlasticThreshold, 
        'clusterPlasticCreep': clusterPlasticCreep,
        'scene_info': scene_info
        }

    with open(os.path.join(out_dir, data_name + '.npy'), 'wb') as f:
        np.save(f, states)


    if args.video:
        image_folder = 'tmp'
        video_name = os.path.join(out_dir, data_name + '.avi')

        images = [img for img in os.listdir(image_folder) if img.endswith(".tga")]
        images.sort(key = lambda f: int(re.sub('\D', '', f)))


        im = Image.open(os.path.join(image_folder, images[0])) 
        nimg = np.array(im)
        frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 60, (width,height))

        for image in images[:time_step]:
            im = Image.open(os.path.join(image_folder, image)) 
            nimg = np.array(im)
            frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            video.write(frame)

        cv2.destroyAllWindows()
        video.release()


# for data_i in range(5000):
#     # clusterStiffness = rand_float(0.3, 0.7)
#     # clusterPlasticThreshold = rand_float(0.00001, 0.0005)
#     # clusterPlasticCreep = rand_float(0.1, 0.3)

#     clusterStiffness = rand_float(0.3, 0.7)
#     clusterPlasticThreshold = 0.00001
#     clusterPlasticCreep = 0.1

#     print(clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep)
#     simulate_scene(data_i, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep)



N_GRID = 5
params_range = np.array([[0.3, 0.7], [0.00001, 0.0005], [0.1, 0.3]])
params_offset = (params_range[:, 1] - params_range[:, 0]) / (N_GRID - 1)

data_i = 0
for p1 in range(N_GRID):
    for p2 in range(N_GRID):
        for p3 in range(N_GRID):
            clusterStiffness = params_range[0][0] + params_offset[0] * p1
            clusterPlasticThreshold = params_range[1][0] + params_offset[1] * p2
            clusterPlasticCreep = params_range[2][0] + params_offset[2] * p3
            data_name = str(p1) + "_" + str(p2) + "_" + str(p1)
            print(clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep)
            simulate_scene(data_i, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep, data_name)
            data_i += 1




pyflex.clean()
