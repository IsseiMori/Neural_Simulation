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
parser.add_argument("--flex", help="Is this flex data?", action='store_true')
parser.add_argument("--mpm", help="Is this mpm data?", action='store_true')
parser.add_argument("--restpos", help="include respos?", action='store_true')

parser.add_argument("--num_data", help="number of data to include", required=False, default=-1, type=int)
parser.add_argument("--offset", help="data load offset", required=False, default=0, type=int)
parser.add_argument("--has_context", help="has context?", required=False, default=True, type=bool)


args = parser.parse_args()

if ( (args.mpm and args.flex) or (not args.mpm and not args.flex) ):
    sys.error("Please specify --mpm or --flex")



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
    files.sort(key = lambda f: int(re.sub('\D', '', f)))

    if args.num_data != -1:
        files = files[args.offset:args.offset+args.num_data]



    loaded = np.load(files[0], allow_pickle=True).item()

    # Ad-hoc method to concate 5 grip iterations in a sequence
    d = loaded
    positions_collapsed = d['positions'].reshape(-1, d['positions'].shape[2], d['positions'].shape[3])
    d['positions'] = np.zeros((1, d['positions'].shape[0] * d['positions'].shape[1], d['positions'].shape[2], d['positions'].shape[3]))
    d['positions'][0] = positions_collapsed
    shape_states_collapsed = d['shape_states'].reshape(-1, d['shape_states'].shape[2], d['shape_states'].shape[3])
    d['shape_states'] = np.zeros((1, d['shape_states'].shape[0] * d['shape_states'].shape[1], d['shape_states'].shape[2], d['shape_states'].shape[3]))
    d['shape_states'][0] = shape_states_collapsed
    loaded = d


    loaded_positions = loaded['positions'][0] 
    if args.mpm: loaded_positions = loaded_positions[:,:,:3]

    pos_mean = np.zeros_like(loaded_positions[0, 0, :]).astype(np.float64)
    vel_mean = np.zeros_like(loaded_positions[0, 0, :]).astype(np.float64)
    acc_mean = np.zeros_like(loaded_positions[0, 0, :]).astype(np.float64)
    ys_mean = np.zeros_like(loaded_positions[0, 0, :]).astype(np.float64)

    vels = []



    total_frames = 0
    for file in files:

        loaded = np.load(file, allow_pickle=True).item()

        # Ad-hoc method to concate 5 grip iterations in a sequence
        d = loaded
        positions_collapsed = d['positions'].reshape(-1, d['positions'].shape[2], d['positions'].shape[3])
        d['positions'] = np.zeros((1, d['positions'].shape[0] * d['positions'].shape[1], d['positions'].shape[2], d['positions'].shape[3]))
        d['positions'][0] = positions_collapsed
        shape_states_collapsed = d['shape_states'].reshape(-1, d['shape_states'].shape[2], d['shape_states'].shape[3])
        d['shape_states'] = np.zeros((1, d['shape_states'].shape[0] * d['shape_states'].shape[1], d['shape_states'].shape[2], d['shape_states'].shape[3]))
        d['shape_states'][0] = shape_states_collapsed
        loaded = d

        loaded_positions = loaded['positions'][0]
        if args.mpm: loaded_positions = loaded_positions[:,:,:3]

        
        print(f'{file}', end="\r",)
        
        for i in range(len(loaded_positions)-2):
            prev_vel = loaded_positions[i+1].astype(np.float32) - loaded_positions[i].astype(np.float32)
            next_vel = loaded_positions[i+2].astype(np.float32) - loaded_positions[i+1].astype(np.float32)
            curr_acc = next_vel - prev_vel

            pos_mean += np.mean(loaded_positions[i].astype(np.float64), axis=0)
            vel_mean += np.mean(prev_vel, axis=0)
            acc_mean += np.mean(curr_acc, axis=0)

            vels.append(prev_vel)
            
            total_frames += 1
        
        if args.has_context:
            if args.mpm:
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



    # vel_var = np.zeros_like(loaded_positions[0]).astype(np.float64)
    # acc_var = np.zeros_like(loaded_positions[0]).astype(np.float64)

    vel_var = np.zeros(loaded_positions.shape[-1]).astype(np.float64)
    acc_var = np.zeros(loaded_positions.shape[-1]).astype(np.float64)

    test_std = []

    total_particles = 0

    for file in files:
        loaded = np.load(file, allow_pickle=True).item()

        # Ad-hoc method to concate 5 grip iterations in a sequence
        d = loaded
        positions_collapsed = d['positions'].reshape(-1, d['positions'].shape[2], d['positions'].shape[3])
        d['positions'] = np.zeros((1, d['positions'].shape[0] * d['positions'].shape[1], d['positions'].shape[2], d['positions'].shape[3]))
        d['positions'][0] = positions_collapsed
        shape_states_collapsed = d['shape_states'].reshape(-1, d['shape_states'].shape[2], d['shape_states'].shape[3])
        d['shape_states'] = np.zeros((1, d['shape_states'].shape[0] * d['shape_states'].shape[1], d['shape_states'].shape[2], d['shape_states'].shape[3]))
        d['shape_states'][0] = shape_states_collapsed
        loaded = d

        loaded_positions = loaded['positions'][0]
        if args.mpm: loaded_positions = loaded_positions[:,:,:3]

        
        print(f'{file}', end="\r",)
        
        for i in range(len(loaded_positions)-2):
            prev_vel = loaded_positions[i+1].astype(np.float32) - loaded_positions[i].astype(np.float32)
            next_vel = loaded_positions[i+2].astype(np.float32) - loaded_positions[i+1].astype(np.float32)
            curr_acc = next_vel - prev_vel

            vel_var += np.sum((prev_vel - vel_mean)**2, axis=0)
            acc_var += np.sum((curr_acc - acc_mean)**2, axis=0)

            total_particles += prev_vel.shape[0]

            test_std.append(prev_vel)


    vel_std = np.sqrt(vel_var / total_particles)
    acc_std = np.sqrt(acc_var / total_particles)
    vel_mean_final = vel_mean
    acc_mean_final = acc_mean




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

    if args.mpm:

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