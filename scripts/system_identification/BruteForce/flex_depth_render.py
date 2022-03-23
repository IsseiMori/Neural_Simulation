import os
import json
import pickle
import numpy as np
import glob
import re
import torch
from glpointrast import perspective, PointRasterizer

import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--data", help="data dir", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
parser.add_argument("--predicted", help="data is predicted rollouts", action='store_true')
args = parser.parse_args()


from PIL import Image as im
def save_depth_image(depth_data, file_name):
    _min = np.amin(depth_data[depth_data != 0])
    _max = np.amax(depth_data[depth_data != 0])
    # print(_min)
    # print(_max)
    _min = -0.7
    _max = -0.4
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)


def sincos(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)

def rotx(a):
    s, c = sincos(a)
    return np.matrix([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])

def roty(a):
    s, c = sincos(a)
    return np.matrix([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])

def rotz(a):
    s, c = sincos(a)
    return np.matrix([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])


model_view = np.linalg.inv(
    np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, -0.5],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1],
    ]))


data_name = args.scene
flex_path = args.data
out_path = args.out

os.system('mkdir -p ' + out_path)


if args.predicted:
    files_flex = glob.glob(os.path.join(flex_path, "*.pkl"))
else:
    files_flex = glob.glob(os.path.join(flex_path, "*.npy"))


files_flex.sort(key = lambda f: int(re.sub('\\D', '', f)))


# Loop over FLEX data
for file_flex in files_flex:

    success = False
    while not success:

        try:

            depths = []

            if args.predicted:
                with open(file_flex, 'rb') as f:
                    d_flex = pickle.load(f)

                n_kinetic_particles = len(d_flex['particle_types'][d_flex['particle_types'] == 1])
                positions_flex = d_flex['predicted_rollout'][:,:n_kinetic_particles]

                dnum_flex = os.path.splitext(file_flex)[0].split('/')[-1]
                dnum_flex = dnum_flex.replace('rollout_', '')
                dnum_flex = f'{int(dnum_flex):05d}'
                depth_out_path = os.path.join(out_path, dnum_flex)
                os.system('mkdir -p ' + depth_out_path)

            else:
                d_flex = np.load(file_flex, allow_pickle=True).item()
                positions_flex = d_flex['positions'][0]

                dnum_flex = os.path.splitext(file_flex)[0].split('/')[-1]
                depth_out_path = os.path.join(out_path, dnum_flex)
                os.system('mkdir -p ' + depth_out_path)


            print(dnum_flex)

            for i_frame in range(len(positions_flex)):


                proj = perspective(np.pi / 3, 1, 0.1, 10)
                raster_func = PointRasterizer(128, 128, 0.01, model_view * rotx(90), proj)

                pos_flex = positions_flex[i_frame][:, :3].astype(np.float32)

                depth = raster_func.apply(torch.tensor(pos_flex).to('cuda'))

                if not len(depth[depth <= -100000]) > 0:
                    raise Exception("depth error 1")

                if depth.max() <= -100000:
                    raise Exception("depth error 2")

                depth[depth <= -100000] = 0
                depths.append(depth.to("cpu").detach().numpy())
                save_depth_image(depth.to("cpu").detach().numpy(), os.path.join(depth_out_path, str(i_frame) + ".png"))

            success = True


        except Exception as error:
            print(repr(error))
            continue
                    

        depths = np.array(depths)
        with open(os.path.join(depth_out_path, dnum_flex + ".npy"), 'wb') as f:
            np.save(f, depths)
