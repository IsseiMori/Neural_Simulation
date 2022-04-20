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
import re

import vispy.scene
from vispy import app
from vispy.visuals import transforms

import vispy_utils


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data path", required=True, type=str)
parser.add_argument("--restpos", help="has rest pos in positions?", action='store_true')
parser.add_argument("--flex", help="Is this flex data?", action='store_true')
parser.add_argument("--mpm", help="Is this mpm data?", action='store_true')
args = parser.parse_args()

if ( (args.mpm and args.flex) or (not args.mpm and not args.flex) ):
    print("Please specify --mpm or --flex")
    exit()

images = []


data = np.load(args.data, allow_pickle=True)

particle_count = data.shape[1]

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

p2 = vispy.scene.visuals.Markers()
p2.antialias = 0  # remove white edge
vispy_utils.y_rotate(p2)
view.add(p2)

frame = 0

total_frames = (len(data) - 5)
seq_length = len(data)-5
# grip_seq_length = len(data[0]['new_positions'][0])
# seq_length = len(data[0]['new_positions']) * grip_seq_length - 5
# total_frames = len(data) * seq_length

def update(event):
    global frame
    global images
    global seq_length
    global total_frames

    data_i = math.floor(frame / seq_length)
    frame_i = frame % seq_length
    ppos = data[frame_i].astype(np.float32) - np.array([0, 0, 0])

    # data_i = math.floor(frame / seq_length)
    # grip_i = math.floor(frame % seq_length / grip_seq_length)
    # frame_i = frame % grip_seq_length
    # ppos = data[data_i]['new_positions'][grip_i, frame_i].astype(np.float32) - np.array([0, 0, 0])

    p1.set_data(ppos, edge_color='black', face_color='white')

    img = c.render()

    images.append(cv2.resize(img, dsize = (800, 600), interpolation = cv2.INTER_CUBIC))

    frame += 1

    if ( frame >= total_frames): 
        c.close()

# start animation
timer = app.Timer()
timer.connect(update)
timer.start(interval=1. / 30., iterations=total_frames)


c.show()
app.run()

print(args.data + ".avi")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_name = (args.data + ".avi").replace("*", "all")
out = cv2.VideoWriter(
    out_name,
    fourcc, 30, (800, 600))

for i in range(total_frames):
    data_i = math.floor(i / seq_length)
    img = images[i]
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:, :] = img.astype(np.uint8)[:,:,0:3]
    

    # frame = cv2.putText(frame, "YS = " + str(data[datan][0]['state'][1]), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    out.write(frame)
out.release()


# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out_name = (args.data + ".avi").replace("*", "all")
# out = cv2.VideoWriter(
#     out_name,
#     fourcc, 30, (800*2, 600))

# for i in range(1, len(data)):
#     for j in range(seq_length):
#         data_frame = seq_length * i + j

#         img_gt = images[j]
#         img_pd = images[data_frame]

#         frame = np.zeros((600, 800 * 2, 3), dtype=np.uint8)
#         frame[:, :800] = img_gt.astype(np.uint8)[:,:,0:3]
#         frame[:, 800:] = img_pd.astype(np.uint8)[:,:,0:3]

#         # frame = cv2.putText(frame, 'clusterStiffness = {:f}'.format(data[0]['clusterStiffness']), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         # frame = cv2.putText(frame, 'clusterPlasticThreshold = {:f}'.format(data[0]['clusterPlasticThreshold']), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         # frame = cv2.putText(frame, 'clusterPlasticCreep = {:f}'.format(data[0]['clusterPlasticCreep']), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
#         # frame = cv2.putText(frame, 'clusterStiffness = {:f}'.format(data[i]['clusterStiffness']), (800+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         # frame = cv2.putText(frame, 'clusterPlasticThreshold = {:f}'.format(data[i]['clusterPlasticThreshold']), (800+30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         # frame = cv2.putText(frame, 'clusterPlasticCreep = {:f}'.format(data[i]['clusterPlasticCreep']), (800+30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         frame = cv2.putText(frame, 'E = {:f}'.format(data[0]['E']), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         frame = cv2.putText(frame, 'YS = {:f}'.format(data[0]['YS']), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         frame = cv2.putText(frame, 'nu = {:f}'.format(data[0]['nu']), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
#         frame = cv2.putText(frame, 'E = {:f}'.format(data[i]['E']), (800+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         frame = cv2.putText(frame, 'YS = {:f}'.format(data[i]['YS']), (800+30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         frame = cv2.putText(frame, 'nu = {:f}'.format(data[i]['nu']), (800+30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


#         # frame = cv2.putText(frame, "YS = " + str(data[datan][0]['state'][1]), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         out.write(frame)
# out.release()



