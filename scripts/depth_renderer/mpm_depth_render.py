import os
import json
import pickle
import numpy as np
import glob
import re

from utils import *

from scipy.spatial.transform import Rotation


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="file pattern for raw input data", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
parser.add_argument("--views", help="json file to specify view angles", required=True, type=str)
parser.add_argument("--n", help="number of data to use", default=10, type=int, required=False)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

with open(args.views, 'r') as f:
    views = json.load(f)

for view in views:
    os.makedirs(os.path.join(args.out, view['view']), exist_ok=True)

files = glob.glob(args.data)
files.sort(key = lambda f: int(re.sub('\\D', '', f)))

for file in files[:min(len(files), args.n)]:

    for view in views:

        out_depth_path = os.path.join(os.path.join(args.out, view['view']), os.path.splitext(file)[0].split('/')[-1])
        os.makedirs(out_depth_path, exist_ok=True)

        depths = []

        # ['states'] type data
        d = np.load(file, allow_pickle=True).item()
        positions = d['x']
        print(positions.shape)

        # numpy type data
        # d = np.load(file, allow_pickle=True)
        # positions = d

        for i_frame in range(0, len(positions)):

            rot = Rotation.from_euler('xyz', view['rotation'], degrees=True).as_quat()
            trans = np.array(view['translation'])
            
            pos = positions[i_frame][:,:3]
            pos[:, 0] -= 0.5
            pos[:, 1] -= 0.1
            pos[:, 2] -= 0.5


            depth = compute_depth_mc(pos, rot, trans, show_canvas=False)

            depths.append(depth)

            save_depth_image(depth, os.path.join(out_depth_path, str(i_frame) + ".png"))

        depths = np.array(depths)
        with open(os.path.join(out_depth_path, os.path.splitext(file)[0].split('/')[-1] + ".npy"), 'wb') as f:
            np.save(f, depths)