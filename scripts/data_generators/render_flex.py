import os
import numpy as np
import pyflex
import time
import torch
import pickle

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
parser.add_argument("--data", help="data path", required=True, type=str)
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
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
        gripper_config = sample_gripper_config(args.scene, random=(not args.fixed_grip))
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

        for image in images[:time_step * grip_time]:
            im = Image.open(os.path.join(image_folder, image)) 
            nimg = np.array(im)
            frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            video.write(frame)

        cv2.destroyAllWindows()
        video.release()


with open(args.data, 'rb') as f:
    data = pickle.load(f)

x = 10.0
y = 10.0
z = 10.0

scene_params = np.array([x, y, z, data['global_context'][0][0], data['global_context'][0][1], data['global_context'][0][2]])
pyflex.set_scene(5, scene_params, 0)

halfEdge = np.array([0.15, 0.8, 0.15])
center = np.array([0., 0., 0.])
quat = np.array([1., 0., 0., 0.])

pyflex.add_box(halfEdge, center, quat)
pyflex.add_box(halfEdge, center, quat)


# print(pyflex.get_positions().reshape(-1, dim_position))



pred_positions = np.concatenate([data['initial_positions'], data['predicted_rollout']])
# pred_positions = data['ground_truth_rollout'][:]
pred_positions = pred_positions[:, :1060, 3:]
print(pred_positions.shape)

# shape_states = data['shape_states']

pred_positions[:,:,0] -= 0.5
pred_positions[:,:,2] -= 0.5

# shape_states[:,:,:,0] -= 0.5
# shape_states[:,:,:,2] -= 0.5
# shape_states[:,:,:,3] -= 0.5
# shape_states[:,:,:,5] -= 0.5

# shape_states[:,:,:,0:6] *= 5
pred_positions *= 5


for i in range(len(pred_positions)):
    pos = pyflex.get_positions().reshape(-1, dim_position)
    pos[:, :3] = pred_positions[i]
    pyflex.set_positions(pos)
    pyflex.render(capture=1, path=os.path.join('tmp', 'render_%d.tga' % (i)))

image_folder = 'tmp'
video_name = 'tmp.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".tga")]
images.sort(key = lambda f: int(re.sub('\D', '', f)))


im = Image.open(os.path.join(image_folder, images[0])) 
nimg = np.array(im)
frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width,height))

for image in images[:len(pred_positions)]:
    im = Image.open(os.path.join(image_folder, image)) 
    nimg = np.array(im)
    frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    video.write(frame)

cv2.destroyAllWindows()
video.release()

pyflex.clean()
