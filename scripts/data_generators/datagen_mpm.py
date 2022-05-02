from plb.engine.taichi_env import TaichiEnv
import taichi as ti
import numpy as np
from plb.config import load

import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
parser.add_argument("--flex", help="flex data path", required=True, type=str)
parser.add_argument("--video", help="write video", action='store_true')
parser.add_argument("--data", help="write video", required=True, type=str)
args = parser.parse_args()

out_dir = args.out
os.system('mkdir -p ' + out_dir)


cfg = load(f"envs/" + args.scene + ".yml") # you can find most default config is at plb/config/default_config.py


cfg['RENDERER']['sdf_threshold'] = 0.20720000000000002*2

env = TaichiEnv(cfg, nn=False, loss=False)

def animate(imgs, filename='animation.webm', _return=True, fps=60):
    print(f'animating {filename}')
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=True)

def set_parameters(env: TaichiEnv, yield_stress, E, nu):
    env.simulator.yield_stress.fill(yield_stress)
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    env.simulator.mu.fill(_mu)
    env.simulator.lam.fill(_lam)


def simulate_scene(data_i, params, data_name):


    
    env.initialize()

    set_parameters(env, params[0], params[1], params[2])

    state = env.get_state()


    gripper_config = sample_gripper_config("BendTube", random=False)

    # d = np.load(os.path.join(args.flex, data_name + ".npy"), allow_pickle=True).item()
    d = np.load(os.path.join(args.flex, args.data + ".npy"), allow_pickle=True).item()
    # d = np.load(os.path.join(args.flex, "0000.npy"), allow_pickle=True).item()
    
    # x, v, F, C, p1, p2, p3 = state['state']
    states_xvfcp = state['state']
    n_grips = d['shape_states'].shape[2]
    n_frames = d['shape_states'].shape[1]

    shape_states_ = d['shape_states'][0][0]
    
    for i_grip in range(n_grips):
        states_xvfcp[4+i_grip][:3] = shape_states_[i_grip][0:3]
        states_xvfcp[4+i_grip][3:] = shape_states_[i_grip][6:10]


    new_state = {
        'state': states_xvfcp,
        'is_copy': state['is_copy'],
        'softness': state['softness'],
    }
    env.set_state(**new_state)


    positions = []
    shape_states = []


    images = []
    states = []
    gird_m = []
    ppos = []

    for frame in range(n_frames-1):
        print(f'{frame}', end="\r",)

        state = env.get_state()

        env.simulator.compute_grid_m_kernel(0)
        gird_m.append(env.simulator.grid_m.to_numpy())

        # ppos.append(env.simulator.x.to_numpy()[0])
        # ppos.append(env.simulator.x.to_numpy())
        
        positions.append(np.concatenate((state['state'][0], np.ones([len(state['state'][0]), 1])), 1))
        sts = np.array(state['state'][4:])
        shape_states.append(np.concatenate([sts[:, :3], sts[:, :3], sts[:, 3:], sts[:, 3:]], axis=1))
        

        # env.step(calc_shape_states_dx(frame * dt, gripper_config))
        env.step((d['shape_states'][0][frame+1][:, 0:3] - d['shape_states'][0][frame][:, 0:3]).flatten() * 100)

        if args.video:
            images.append(env.render('rgb_array'))

        ppos.append(env.simulator.x.to_numpy()[0])

    state = env.get_state()

    env.simulator.compute_grid_m_kernel(0)
    gird_m.append(env.simulator.grid_m.to_numpy())

    # ppos1.append(env.simulator.x.to_numpy()[0])
    # ppos.append(env.simulator.x.to_numpy())

    positions.append(np.concatenate((state['state'][0], np.ones([len(state['state'][0]), 1])), 1))

    sts = np.array(state['state'][4:])
    shape_states.append(np.concatenate([sts[:, :3], sts[:, :3], sts[:, 3:], sts[:, 3:]], axis=1))

    states = {
            'positions': np.array([positions]),
            'shape_states': np.array([shape_states]),
            'YS': params[0], 
            'E': params[1], 
            'nu': params[2],
            'scene_info': d['scene_info']
            }

    with open(os.path.join(out_dir, args.scene + '-action-' + args.data + '.npy'), 'wb') as f:
            np.save(f, states)

    # with open(os.path.join(out_dir, 'gridm_' + data_name + '.npy'), 'wb') as f:
    #         np.save(f, np.array(gird_m))

    # ppos = env.simulator.x.to_numpy()
    # print(ppos.shape)
    with open(os.path.join(out_dir, args.scene + '-ppos-' + args.data + '.npy'), 'wb') as f:
            np.save(f, np.array(ppos))

    if args.video:
        animate(images, os.path.join(out_dir, args.scene + '-' + args.data + '.webm'))


for data_i in range(1):
    # YS = 5 + np.random.random()*195
    # E = 100 + np.random.random()*2900
    # nu = 0 + np.random.random()*0.45

    # YS = 5
    # E = 100 + np.random.random()*2900
    # nu = 0

    N_GRID = 5
    params_range = np.array([[5, 200], [100, 3000], [0, 0.45]])
    params_offset = (params_range[:, 1] - params_range[:, 0]) / (N_GRID - 1)
    # YS = params_range[0][0] + params_offset[0] * 3
    # E = params_range[1][0] + params_offset[1] * 1
    # nu = params_range[2][0] + params_offset[2] * 3

    # For finetuning
    YS = params_range[0][0] + params_offset[0] * 4
    E = params_range[1][0] + params_offset[1] * 1
    nu = params_range[2][0] + params_offset[2] * 4


    params = []
    params.append(YS)
    params.append(E)
    params.append(nu)
    print(params)
    data_name = f'{data_i:05d}'
    simulate_scene(data_i, params, data_name)



# N_GRID = 5
# params_range = np.array([[5, 200], [100, 3000], [0, 0.45]])
# params_offset = (params_range[:, 1] - params_range[:, 0]) / (N_GRID - 1)

# data_i = 0
# for p1 in range(N_GRID):
#     for p2 in range(N_GRID):
#         for p3 in range(N_GRID):
#             params = []
#             params.append(params_range[0][0] + params_offset[0] * p1)
#             params.append(params_range[1][0] + params_offset[1] * p2)
#             params.append(params_range[2][0] + params_offset[2] * p3)
#             print(params)
#             data_name = str(p1) + "_" + str(p2) + "_" + str(p3)
#             simulate_scene(data_i, params, data_name)
#             data_i += 1