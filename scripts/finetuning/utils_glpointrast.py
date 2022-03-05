import numpy as np

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

import math
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