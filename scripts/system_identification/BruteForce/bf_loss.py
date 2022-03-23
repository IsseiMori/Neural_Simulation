import os
import json
import pickle
import numpy as np
import glob
import re
import math
import torch

from glpointrast import perspective, PointRasterizer

from pytorch3d.loss import chamfer_distance


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--flex_data", help="data dir", required=True, type=str)
parser.add_argument("--mpm_data", help="data dir", required=True, type=str)
parser.add_argument("--flex_depth", help="data dir", required=True, type=str)
parser.add_argument("--mpm_depth", help="data dir", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
args = parser.parse_args()

os.system('mkdir -p ' + args.out)




def unproject_torch(proj, proj_inv, depth_image):
    z = proj[2, 2] * depth_image + proj[2, 3]
    w = proj[3, 2] * depth_image
    z_ndc = z / w

    H, W = depth_image.shape
    ndc = torch.stack(
        [
            torch.tensor(x, dtype=depth_image.dtype, device=depth_image.device)
            for x in np.meshgrid(
                np.arange(0.5, W + 0.4) * 2 / W - 1,
                np.arange(0.5, H + 0.4)[::-1] * 2 / H - 1,
            )
        ]
        + [z_ndc, torch.ones_like(z_ndc)],
        axis=-1,
    )
    pos = ndc @ proj_inv.T
    return pos[..., :3] / pos[..., [3]]


proj_matrix = torch.Tensor(np.array([
    [ 1.73205081,  0.,          0.,          0.        ],
    [ 0.,          1.73205081,  0.,          0.        ],
    [ 0.,          0.,         -1.,         -0.1       ],
    [ 0.,          0.,         -1.,          0.        ],
])).to('cuda')

proj_matrix_inv = torch.Tensor(np.array([
    [  0.57735027,   0.,          -0.,           0.        ],
    [  0.,           0.57735027,  -0.,           0.        ],
    [  0.,           0.,          -0.,         -10.        ],
    [  0.,           0.,          -1.,          10.        ],
])).to('cuda')



print(args.flex_data)
files_flex = glob.glob(os.path.join(args.flex_data, "*.npy"))
files_flex.sort(key = lambda f: int(re.sub('\\D', '', f)))

print(files_flex)

loss_data = []


# Select corners
# files_flex = [
#                 files_flex[0], files_flex[4], files_flex[19], files_flex[24],
#                 files_flex[100], files_flex[104], files_flex[119], files_flex[124]
#             ]

# Loop over FLEX data
for file_flex in files_flex:

    dnum_flex = os.path.splitext(file_flex)[0].split('/')[-1]
    print(dnum_flex)
    
    d_flex = np.load(file_flex, allow_pickle=True).item()
    depth_flex = np.load(os.path.join(args.flex_depth, dnum_flex, dnum_flex + ".npy"), allow_pickle=True)
    depth_flex = torch.Tensor(depth_flex).to('cuda')


    files_mpm = glob.glob(os.path.join(args.mpm_data, "*.npy"))
    files_mpm.sort(key = lambda f: int(re.sub('\\D', '', f)))

    loss_data_mpm = []

    # Loop over MPM data
    for file_mpm in files_mpm:

        dnum_mpm = os.path.splitext(file_mpm)[0].split('/')[-1]
        print(dnum_mpm)

        d_mpm = np.load(file_mpm, allow_pickle=True).item()
        depth_mpm = np.load(os.path.join(args.mpm_depth, dnum_mpm, dnum_mpm + ".npy"), allow_pickle=True)
        depth_mpm = torch.Tensor(depth_mpm).to('cuda')

        # Set background to a reasonable depth
        depth_flex[depth_flex == 0] = -100
        depth_mpm[depth_mpm == 0] = -100

        loss = []
        for i_frame in range(len(depth_mpm)):

            points_projected_flex = unproject_torch(proj_matrix, proj_matrix_inv, depth_flex[i_frame])
            points_projected_mpm = unproject_torch(proj_matrix, proj_matrix_inv, depth_mpm[i_frame])

            points_projected_flex = torch.flatten(points_projected_flex, start_dim=0, end_dim=1)
            points_projected_mpm = torch.flatten(points_projected_mpm, start_dim=0, end_dim=1)
            points_projected_flex = points_projected_flex[None, :]
            points_projected_mpm = points_projected_mpm[None, :]

            loss.append(chamfer_distance(points_projected_flex, points_projected_mpm)[0].to("cpu").detach().numpy())

        loss = np.array(loss)
        loss_data_mpm.append(loss)
        
    loss_data.append(loss_data_mpm)

    loss_data_mpm = np.array(loss_data_mpm)
    with open(os.path.join(args.out, dnum_flex + ".npy"), 'wb') as f:
        np.save(f, loss_data_mpm)


loss_data = np.array(loss_data)
with open(os.path.join(args.out, "loss.npy"), 'wb') as f:
    np.save(f, loss_data)

