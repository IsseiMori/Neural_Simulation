import collections
import functools
import json
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import glob, os
import re

tf.enable_eager_execution()
tf.executing_eagerly()


def _read_metadata(data_path):
  with open(os.path.join(data_path), 'rt') as fp:
    return json.loads(fp.read())


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data dir", required=True, type=str)
parser.add_argument("--out", help="out dir", required=True, type=str)
parser.add_argument("--simulator", help="data type", required=True, type=str)
parser.add_argument("--restpos", help="include respos?", action='store_true')

parser.add_argument("--num_data", help="number of data to include", required=False, default=100, type=int)
parser.add_argument("--offset", help="data load offset", required=False, default=0, type=int)
parser.add_argument("--has_context", help="has context?", required=False, default=True, type=bool)


args = parser.parse_args()

if not args.simulator == 'mpm' and not args.simulator == 'flex':
    sys.error("--simulator must be mpm or flex")



def main():

    os.system('mkdir -p ' + args.out)
    os.system('mkdir -p ' + args.out + '/rollouts')


    velocities = []
    positions = []
    accelerations = []
    contexts0 = []
    contexts1 = []
    contexts2 = []


    files = glob.glob(os.path.join(args.data, "*.npy"))
    files = files[args.offset:args.offset+args.num_data]
    files.sort(key = lambda f: int(re.sub('\D', '', f)))



    loaded = np.load(files[0], allow_pickle=True).item()
    loaded_positions = loaded['positions'][0] 
        
    pos_mean = np.zeros_like(loaded_positions[0, :]).astype(np.float64)
    vel_mean = np.zeros_like(loaded_positions[0, :]).astype(np.float64)
    acc_mean = np.zeros_like(loaded_positions[0, :]).astype(np.float64)
    ys_mean = np.zeros_like(loaded_positions[0, :]).astype(np.float64)

    total_frames = 0




    total_frames = 0
    for file in files:

        loaded = np.load(file, allow_pickle=True).item()
        loaded_positions = loaded['positions'][0] 

        
        print(f'{file}', end="\r",)
        
        for i in range(len(loaded_positions)-2):
            prev_vel = loaded_positions[i+1].astype(np.float32) - loaded_positions[i].astype(np.float32)
            next_vel = loaded_positions[i+2].astype(np.float32) - loaded_positions[i+1].astype(np.float32)
            curr_acc = next_vel - prev_vel

            pos_mean += loaded_positions[i].astype(np.float64)
            vel_mean += prev_vel
            acc_mean += curr_acc
            
            total_frames += 1
        
        if args.has_context:
            if args.simulator == 'mpm':
                contexts0.append(loaded['YS'])
                contexts1.append(loaded['E'])
                contexts2.append(loaded['nu'])
            else:
                contexts0.append(loaded['clusterStiffness'])
                contexts1.append(loaded['clusterPlasticThreshold'])
                contexts2.append(loaded['clusterPlasticCreep'])

    pos_mean /= total_frames
    vel_mean /= total_frames
    acc_mean /= total_frames

    if args.has_context:
        contexts0 = np.vstack(contexts0)
        contexts1 = np.vstack(contexts1)
        contexts2 = np.vstack(contexts2)




    pos_var = np.zeros_like(loaded_positions[0]).astype(np.float64)
    vel_var = np.zeros_like(loaded_positions[0]).astype(np.float64)
    acc_var = np.zeros_like(loaded_positions[0]).astype(np.float64)

    for file in files:
        loaded = np.load(file, allow_pickle=True).item()
        loaded_positions = loaded['positions'][0]

        
        print(f'{file}', end="\r",)
        
        for i in range(len(loaded_positions)-2):
            prev_vel = loaded_positions[i+1].astype(np.float32) - loaded_positions[i].astype(np.float32)
            next_vel = loaded_positions[i+2].astype(np.float32) - loaded_positions[i+1].astype(np.float32)
            curr_acc = next_vel - prev_vel

            vel_var += (prev_vel - vel_mean)**2
            acc_var += (curr_acc - acc_mean)**2




    vel_std = np.sqrt(vel_var / total_frames)
    acc_std = np.sqrt(acc_var / total_frames)
    vel_std = np.std(vel_std, axis=0)
    acc_std = np.std(acc_std, axis=0)
    vel_mean_final = np.mean(vel_mean, axis=0)
    acc_mean_final = np.mean(acc_mean, axis=0)




    velmean = list(vel_mean_final)
    velstd = list(vel_std)

    accmean = list(acc_mean_final)
    accstd = list(acc_std)

    if args.has_context: context_mean = [np.mean(contexts0.astype(dtype=np.float64)), np.mean(contexts1.astype(dtype=np.float64)), np.mean(contexts2.astype(dtype=np.float64))]
    if args.has_context: context_std = [np.std(contexts0.astype(dtype=np.float64)), np.std(contexts1.astype(dtype=np.float64)), np.std(contexts2.astype(dtype=np.float64))]




    metadata = {}
    theoretical_min = 0 # 1/64 * 3
    bounds_min = theoretical_min
    bounds_max = 1 - theoretical_min

    if args.restpos:
        metadata['bounds'] = [[bounds_min, bounds_max], [bounds_min, bounds_max], [bounds_min, bounds_max], [bounds_min, bounds_max], [bounds_min, bounds_max], [bounds_min, bounds_max]]
        metadata['dim'] = 6
    else:
        metadata['bounds'] = [[bounds_min, bounds_max], [bounds_min, bounds_max], [bounds_min, bounds_max]]
        metadata['dim'] = 3

    metadata['sequence_length'] = len(loaded_positions)-1

    if args.simulator == 'mpm':

        if args.restpos:
            velmean.append(velmean)
            velstd.append(velstd)
            accmean.append(accmean)
            accstd.append(accstd)

        metadata['vel_mean'] = velmean
        metadata['vel_std'] = velstd
        metadata['acc_mean'] = accmean
        metadata['acc_std'] = accstd

    else:
        metadata['vel_mean'] = velmean if args.restpos else velmean[3:]
        metadata['vel_std'] = velstd if args.restpos else velstd[3:]
        metadata['acc_mean'] = accmean if args.restpos else accmean[3:]
        metadata['acc_std'] = accstd if args.restpos else accstd[3:]


    metadata['default_connectivity_radius'] = 0.025

    if args.has_context: metadata['context_mean'] = context_mean
    if args.has_context: metadata['context_std'] = context_std

    with open(args.out + '/metadata.json', 'w') as f:
        json.dump(metadata, f)

    with open(args.out + '/rollouts/metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()