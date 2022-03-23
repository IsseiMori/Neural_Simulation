import os
import json
import pickle
import numpy as np
import glob
import re


from utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data dir", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
args = parser.parse_args()


os.system('mkdir -p ' + args.out)


files = glob.glob(os.path.join(args.data, "*.npy"))
files.sort(key = lambda f: int(re.sub('\\D', '', f)))


for file in files:

    depth_out_path = os.path.join(args.out, os.path.splitext(file)[0].split('/')[-1])
    os.system('mkdir -p ' + depth_out_path)

    depths = []

    d = np.load(file, allow_pickle=True).item()

    positions = d['positions'][0]

    for i_frame in range(len(positions)):

        rot = rotz_q(90)
        trans = np.array([-0.5, 0.5, -0.7])
        depth = compute_depth_mc(positions[i_frame][:,:3], rot, trans, show_canvas=False)

        depths.append(depth)

        save_depth_image(depth, os.path.join(depth_out_path, str(i_frame) + ".png"))


    depths = np.array(depths)
    with open(os.path.join(depth_out_path, os.path.splitext(file)[0].split('/')[-1] + ".npy"), 'wb') as f:
        np.save(f, depths)
