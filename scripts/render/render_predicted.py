import os
import time
import numpy as np
import pickle
import cv2
import glob, os

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
    sys.error("Please specify --mpm or --flex")


data = []

files = glob.glob(args.data)

for file in files:

    with open(file, 'rb') as f:
        data.append(pickle.load(f))

c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')

view = c.central_widget.add_view()

view.camera = vispy.scene.cameras.TurntableCamera(fov=0, azimuth=71.5, elevation=84.5, distance=1, up='+y')
view.camera.scale_factor = 0.5259203606311046
view.camera.center = [0.02679920874137362, 0.038526414694705836, 0.6569861515101996]

# set instance colors
instance_colors = vispy_utils.create_instance_colors(1)

# render floor
vispy_utils.add_floor(view)

# render particles
p1 = vispy.scene.visuals.Markers()
p1.antialias = 0  # remove white edge

vispy_utils.y_rotate(p1)

view.add(p1)


images = []
seq_length = len(data[0]['ground_truth_rollout'])-1 # predicted rollout has 1 less frame

n_kinetic_particles = len(data[0]['particle_types'][data[0]['particle_types'] == 1])

timestep = 0
def update(event):
    global timestep
    global images
    global seq_length

    frame = timestep % (seq_length*2)
    data_n = int(timestep / (seq_length*2))

    print(f'Rollouts {data_n}: {frame} / {seq_length*2}', end="\r",)

    if (frame < seq_length ): 
        if args.restpos: ppos = data[data_n]['ground_truth_rollout'][frame][:,3:]
        else: ppos = data[data_n]['ground_truth_rollout'][frame][:,:3]
        p1.set_data(ppos, edge_color='black', face_color='white')

        img = c.render()
        images.append(img)

    else: 
        if args.restpos: ppos = data[data_n]['predicted_rollout'][frame-seq_length][:,3:]
        else: ppos = data[data_n]['predicted_rollout'][frame-seq_length][:,:3]
        p1.set_data(ppos, edge_color='black', face_color='white')

        img = c.render()
        images.append(img)

    timestep += 1

    if ( timestep >= seq_length*2*len(data)): 
        c.close()

# start animation
timer = app.Timer()
timer.connect(update)
timer.start(interval=1. / 30., iterations=seq_length*2*len(data))

#c.show()
app.run()


# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out_name = (args.data + ".avi").replace("*", "all")
# out = cv2.VideoWriter(
#     out_name,
#     fourcc, 30, (800*2, 600))


# for j in range(len(data)):
#     offset = j * seq_length * 2
#     for i in range(seq_length):
#         img_gt = images[offset+i]
#         img_pd = images[offset+i+seq_length]

#         frame = np.zeros((600, 800 * 2, 3), dtype=np.uint8)
#         frame[:, :800] = img_gt.astype(np.uint8)[:,:,0:3]
#         frame[:, 800:] = img_pd.astype(np.uint8)[:,:,0:3]

#         mse = ((data[j]['ground_truth_rollout'][:,:n_kinetic_particles] - data[j]['predicted_rollout'][:,:n_kinetic_particles])**2).mean()

#         frame = cv2.putText(frame, 'Ground Truth #{:d}'.format(j), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
#         if args.flex:
#             frame = cv2.putText(frame, 'clusterStiffness = {:f}'.format(data[j]['global_context'][0][0]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, 'clusterPlasticThreshold = {:f}'.format(data[j]['global_context'][0][1]), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, 'clusterPlasticCreep = {:f}'.format(data[j]['global_context'][0][2]), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         else:
#             frame = cv2.putText(frame, 'YS = {:f}'.format(data[j]['global_context'][0][0]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, 'E = {:f}'.format(data[j]['global_context'][0][1]), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, 'nu = {:f}'.format(data[j]['global_context'][0][2]), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         frame = cv2.putText(frame, 'Predicted Result #{:d} (MSE = {:.5f})'.format(j, mse), (800+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         frame = cv2.putText(frame, 'Frame = {:d}'.format(j), (800+30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         out.write(frame)

# out.release()


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_name = (args.data + ".avi").replace("*", "all")
out = cv2.VideoWriter(
    out_name,
    fourcc, 30, (800, 600))


for j in range(len(data)):
    offset = j * seq_length * 2
    for i in range(seq_length):
        img_gt = images[offset+i]
        img_pd = images[offset+i+seq_length]

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame = img_pd.astype(np.uint8)[:,:,0:3]

        # mse = ((data[j]['ground_truth_rollout'][:,:n_kinetic_particles] - data[j]['predicted_rollout'][:,:n_kinetic_particles])**2).mean()

        # frame = cv2.putText(frame, 'Ground Truth #{:d}'.format(j), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # if args.flex:
        #     frame = cv2.putText(frame, 'clusterStiffness = {:f}'.format(data[j]['global_context'][0][0]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #     frame = cv2.putText(frame, 'clusterPlasticThreshold = {:f}'.format(data[j]['global_context'][0][1]), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #     frame = cv2.putText(frame, 'clusterPlasticCreep = {:f}'.format(data[j]['global_context'][0][2]), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # else:
        #     frame = cv2.putText(frame, 'YS = {:f}'.format(data[j]['global_context'][0][0]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #     frame = cv2.putText(frame, 'E = {:f}'.format(data[j]['global_context'][0][1]), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #     frame = cv2.putText(frame, 'nu = {:f}'.format(data[j]['global_context'][0][2]), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # frame = cv2.putText(frame, 'Predicted Result #{:d} (MSE = {:.5f})'.format(j, mse), (800+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # frame = cv2.putText(frame, 'Frame = {:d}'.format(j), (800+30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(frame)

out.release()