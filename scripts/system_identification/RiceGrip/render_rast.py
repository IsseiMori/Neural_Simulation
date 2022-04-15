import numpy as np
import torch
from glpointrast import perspective, PointRasterizer

import math

from scipy.spatial.transform import Rotation

from PIL import Image as im
def save_depth_image(depth_data, file_name):
    # _min = np.amin(depth_data[depth_data != 0])
    # _max = np.amax(depth_data[depth_data != 0])
    _min = -0.7
    _max = -0.4
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)




rot = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
mv_mat = np.zeros((4,4))
mv_mat[:3, :3] = rot
mv_mat[:, 3] = np.array([0, 0, -0.7, 1])
print(mv_mat)


proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(128, 128, 0.01, mv_mat, proj)

# points_true = np.load('points_true.npy')
points_true = np.load('/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/MPM/raw2/04000.npy', allow_pickle=True).item()['positions'][0, 20, :, :3]
points_true = points_true.astype(np.float32)
points_true[:, 0] -= 0.5
points_true[:, 1] -= 0.1
points_true[:, 2] -= 0.5
print(points_true.mean(axis=0))


# proj = perspective(np.pi / 3, 1, 0.1, 10)
# raster_func = PointRasterizer(512, 512, 0.03, model_view * rotx(0), proj)

depth_true = raster_func.apply(torch.tensor(points_true).to('cuda'))

depth_true[depth_true < -100000] = 0

save_depth_image(depth_true.to("cpu").detach().numpy(), 'tmp/depth_true.png')