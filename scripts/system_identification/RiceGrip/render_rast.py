import numpy as np
import torch
from glpointrast import perspective, PointRasterizer

import math

from PIL import Image as im
def save_depth_image(depth_data, file_name):
    # _min = np.amin(depth_data[depth_data != 0])
    # _max = np.amax(depth_data[depth_data != 0])
    _min = -0.5
    _max = -0.2
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
        [0, 0, 1, 0.5],
        [0, 0, 0, 1],
    ]))

points_true = np.load('points_true.npy')



proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(512, 512, 0.03, model_view * rotx(90), proj)

depth_true = raster_func.apply(torch.tensor(points_true).to('cuda'))

depth_true[depth_true < -100000] = 0

save_depth_image(depth_true.to("cpu").detach().numpy(), 'depth_true.png')