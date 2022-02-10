"""
python render_np_vispy --data=
"""


import os
import time
import numpy as np
import pickle
import cv2
import argparse
import math
import glob

import vispy.scene
from vispy import app
from vispy.visuals import transforms

import vispy_utils


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", help="data path", required=True, type=str,
)
args = parser.parse_args()


images = []
data = []

files = glob.glob(args.data)

for file in files:
    d = np.load(file, allow_pickle=True).item()
    d['new_positions'] = vispy_utils.add_grips(d['positions'][:,:,:,3:] , d['shape_states'])[:,:,:,:3]
    data.append(d)

c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = c.central_widget.add_view()

view.camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=71.5, elevation=84.5, distance=1, up='+y')

view.camera.scale_factor = 0.5259203606311046
view.camera.center = [0.02679920874137362, 0.038526414694705836, 0.6569861515101996]


# set instance colors
instance_colors = vispy_utils.create_instance_colors(1)

# render particles
p1 = vispy.scene.visuals.Markers()
p1.antialias = 0  # remove white edge
vispy_utils.y_rotate(p1)
view.add(p1)

frame = 0

total_frames = len(data) * (len(data[0]['new_positions'][0]) - 5)
seq_length = len(data[0]['new_positions'][0])-5

print(len(data[0]['new_positions'][0])-5)

def update(event):
    global frame
    global images
    global seq_length
    global total_frames

    data_i = math.floor(frame / seq_length)
    frame_i = frame % seq_length
    ppos = data[data_i]['new_positions'][0, frame_i].astype(np.float32) - np.array([0, 0, 0])
    ppos[:,0] -= data[data_i]['new_positions'][0, 0].mean(axis=0).astype(np.float32)[0]
    ppos[:,2] -= data[data_i]['new_positions'][0, 0].mean(axis=0).astype(np.float32)[2]
    ppos /= 5
    ppos[:,0] += 0.5
    ppos[:,2] += 0.5
    p1.set_data(ppos, edge_color='black', face_color='white')

    img = c.render()

    images.append(cv2.resize(img, dsize = (800, 600), interpolation = cv2.INTER_CUBIC))

    frame += 1

    if ( frame >= total_frames): 
        c.close()

# start animation
timer = app.Timer()
timer.connect(update)
timer.start(interval=1. / 3., iterations=len(data) * len(data[0]['positions'][0]))

c.show()
app.run()

print(args.data + ".avi")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_name = (args.data + ".avi").replace("*", "all")
out = cv2.VideoWriter(
    out_name,
    fourcc, 30, (800, 600))

for i in range(total_frames):
    img = images[i]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:, :] = img.astype(np.uint8)[:,:,0:3]
    # frame = cv2.putText(frame, "YS = " + str(data[datan][0]['state'][1]), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    out.write(frame)
out.release()



