import os
import json
import pickle
import numpy as np
import glob
import re

from utils_simulator import *
from utils import *


root_dir = os.environ.get('NSIMROOT')
mpm_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/raw_mpm/*.npy")
out_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/finetuned/mpm_depth2")


files = glob.glob(mpm_path)
files.sort(key = lambda f: int(re.sub('\\D', '', f)))


for file in files:

    depths = []

    d = np.load(file, allow_pickle=True).item()

    positions = d['positions'][0]

    for i_frame in range(len(positions)):

        rot = rotz_q(90)
        trans = np.array([-0.5, 0.5, -0.7])
        depth = compute_depth_mc(positions[i_frame][:,:3], rot, trans, show_canvas=False)

        depths.append(depth)

        save_depth_image(depth, os.path.join(out_path, str(i_frame) + ".png"))


    depths = np.array(depths)
    with open(os.path.join(out_path, os.path.splitext(file)[0].split('/')[-1] + ".npy"), 'wb') as f:
        np.save(f, depths)

    exit()



# ds_mpm = prepare_data_from_tfds(data_path=mpm_path, split='4', is_rollout=True)

# num_inference_steps = 25

# depths = []

# for example_i, (features, labels) in enumerate(ds_mpm):

#     print(features['position'].shape)

#     for frame_i in range(num_inference_steps+1):
#         print(frame_i)

#         n_kinetic_particles = len(features['particle_type'][features['particle_type'] == 1])
#         points_true = features['position'][:n_kinetic_particles,frame_i, 3:]


#         depth = compute_depth_glpointrast(points_true)

#         depths.append(depth)

#         save_depth_image(depth, "images_mpm/" + str(frame_i) + ".png")

# depths = np.array(depths)
# print(depths.shape)
# with open('mpm_depth.npy', 'wb') as f:
#     np.save(f, depths)
