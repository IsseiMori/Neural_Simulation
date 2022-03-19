import collections
import functools
import json
import os
import pickle
import glob
import re
import math
import sys

import numpy as np
import tensorflow.compat.v1 as tf


tf.enable_eager_execution()
tf.executing_eagerly()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data dir", required=True, type=str)
parser.add_argument("--out", help="out dir", required=True, type=str)
parser.add_argument("--flex", help="Is this flex data?", action='store_true')
parser.add_argument("--mpm", help="Is this mpm data?", action='store_true')
parser.add_argument("--restpos", help="include respos?", action='store_true')

parser.add_argument("--num_data", help="number of data to include", required=False, default=100, type=int)
parser.add_argument("--offset", help="data load offset", required=False, default=0, type=int)
parser.add_argument("--name", help="name of the output file", required=False, default="train", type=str)
parser.add_argument("--has_context", help="has context?", required=False, default=True, type=bool)


args = parser.parse_args()

if ( (args.mpm and args.flex) or (not args.mpm and not args.flex) ):
    sys.error("Please specify --mpm or --flex")



def rotate(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)


def particlify_box(center, half_edge, quat):
    
    pos = []

    # initial spacing
    offset_height = 0.02
    offset_width = 0.02
    
    half_width = half_edge[0] # assume width = depth
    half_height = half_edge[1]

    particle_count_height = math.ceil(half_height * 2 / offset_height)
    particle_count_width = math.ceil(half_width * 2 / offset_width)

    offset_height = half_height * 2 / particle_count_height
    offset_width = half_width * 2 / particle_count_width


    local_bottom_corner_pos = np.array([-half_width, -half_height, - half_width])


    for h in range(0, particle_count_height + 1):
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([offset_width * w, offset_height * h, 0]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([half_width * 2, offset_height * h, offset_width * w]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([half_width * 2 - offset_width * w, offset_height * h, half_width * 2]))
        for w in range(0, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([0, offset_height * h, half_width * 2 - offset_width * w]))

    for r in range(1, particle_count_width):
        for c in range(1, particle_count_width):
            pos.append(local_bottom_corner_pos + np.array([offset_width * r, half_height * 2, offset_width * c]))
            pos.append(local_bottom_corner_pos + np.array([offset_width * r, 0, offset_width * c]))
        

    pos = np.asarray(pos, dtype=np.float64)
    
    for i in range(len(pos)):
        pos[i] = rotate(pos[i], quat)

    pos[:,0] += center[0]
    pos[:,1] += center[1]
    pos[:,2] += center[2]
        
    # pos = np.concatenate((pos, np.ones([len(pos), 1])), 1)
    
    return pos


"""
respos: if restpos is True, grip particles will be expanded to 6 dim
"""
def add_grips(positions, shape_states, half_edge, restpos=False):
    pos_all = []
    for r in range(len(positions)):
        pos_grip_iter = []
        for i in range(len(positions[0])):

            pos_grips = []

            for i_grip in range(len(shape_states[r, i])):

                pos = shape_states[r, i][i_grip][0:3]
                quat = shape_states[r, i][i_grip][6:10]
                pos_grip = particlify_box(pos, half_edge, quat)

                if restpos: pos_grips.append(np.concatenate([pos_grip, pos_grip], axis=1))
                else : pos_grips.append(pos_grip)

            pos_grips = np.array(pos_grips)
            pos_grips = pos_grips.reshape(-1, pos_grips.shape[-1])


            pos_grip_iter.append(np.concatenate((positions[r,i], pos_grips), 0))

        pos_all.append(pos_grip_iter)

    pos_all = np.asarray(pos_all, dtype=np.float64)

    return pos_all

def generate_tfrecord_plb(data_name, writer_name, idx_start, idx_end, _HAS_CONTEXT=True, is_mpm=True, restpos=False):

    writer = tf.python_io.TFRecordWriter(writer_name)

    files = glob.glob(os.path.join(data_name, "*.npy"))
    files.sort(key = lambda f: int(re.sub('\D', '', f)))
    files = files[idx_start:idx_end]
    
    i = 0
    for file in files:
        print(f'{file}', end="\r",)
        d = np.load(file, allow_pickle=True).item()


        # Random sample to match MPM with FLEX
        num_particles = 1060
        random_indices = np.random.randint(d['positions'].shape[2], size=num_particles)
        d['positions'] = d['positions'][:,:,random_indices]

        # If MPM and --restpos, need to extend position to 6 dim
        # If FLEX and not --restpos, remove restpos
        if is_mpm:
            if restpos:
                d_pos = np.concatenate([d['positions'][:,:,:,:3], d['positions'][:,:,:,:3]], axis=3)
            else:
                d_pos = d['positions'][:,:,:,:3]
        else:
            if not restpos:
                d_pos = d['positions'][:,:,:,3:]
            else:
                d_pos = d['positions']


        d['new_positions'] = add_grips(d_pos , d['shape_states'], d['scene_info'], restpos)

        
        n_particle_plasticine = len(d['positions'][0,0])
        
        positions = d['new_positions'][0].astype(np.float32)


        step_contexts = []
        for _ in range(0, len(positions)):
            if is_mpm:
                step_contexts.append(d['YS'])
                step_contexts.append(d['E'])
                step_contexts.append(d['nu'])
            else:
                step_contexts.append(d['clusterStiffness'])
                step_contexts.append(d['clusterPlasticThreshold'])
                step_contexts.append(d['clusterPlasticCreep'])


        positions = np.asarray(positions)
        step_contexts = np.asarray(step_contexts)

        # Create feature list
        positions_bytes_list = []
        for pos in positions: # per frame
            positions_bytes = pos.tobytes()
            positions_bytes = tf.train.Feature(bytes_list = tf.train.BytesList(value=[positions_bytes]))
            positions_bytes_list.append(positions_bytes)

        step_context_bytes_list = []
        for step_context in step_contexts: # per frame
            step_context_bytes = np.float32(step_context).tobytes()
            step_context_bytes = tf.train.Feature(bytes_list = tf.train.BytesList(value=[step_context_bytes]))
            step_context_bytes_list.append(step_context_bytes)

        positions_feature_list = tf.train.FeatureList(feature=positions_bytes_list)
        if _HAS_CONTEXT:
            step_context_feature_list = tf.train.FeatureList(feature=step_context_bytes_list)

        particle_type = np.ones([positions[0].shape[0]], dtype=np.int64)
        particle_type[n_particle_plasticine:] += 2
        particle_type = particle_type.tobytes()
        particle_type_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[particle_type]))

        key = np.int64(i)
        key_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[key]))

        sequence_dict = {'position': positions_feature_list, 'step_context': step_context_feature_list}

        context_dict = {'key': key_feature, 'particle_type': particle_type_feature}

        sequence_context = tf.train.Features(feature=context_dict)
        # now create a list of feature lists contained within dictionary
        sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)

        example = tf.train.SequenceExample(context=sequence_context, feature_lists=sequence_list)

        writer.write(example.SerializeToString())

        i += 1

    writer.close()


def main():

    os.system('mkdir -p ' + args.out)

    generate_tfrecord_plb(
        args.data, 
        os.path.join(args.out, args.name + ".tfrecord"), 
        args.offset, 
        args.offset + args.num_data, 
        args.has_context,
        args.mpm,
        args.restpos
    )



if __name__ == "__main__":
    main()