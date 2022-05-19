from plb.engine.taichi_env import TaichiEnv
import taichi as ti
import numpy as np
from plb.config import load

import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--out", help="output dir", required=True, type=str)
parser.add_argument("--n", help="number of data", required=False, type=int, default=1)
parser.add_argument("--video", help="write video", action='store_true')
parser.add_argument("--ppos", help="export x positions", action='store_true')
parser.add_argument("--offset", help="number of data offset", required=False, type=int, default=0)
args = parser.parse_args()

out_dir = args.out
os.system('mkdir -p ' + out_dir)

dt = 1. / 60.
dim_shape_state = 14


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

    np.random.seed(data_i) 

    env.initialize()

    scene_info = init_scene_plb(env, args.scene)

    set_parameters(env, params[0], params[1], params[2])

    state = env.get_state()

    gripper_config = sample_gripper_config(args.scene, random=True)
    shape_states_ = calc_shape_states(0 * dt, gripper_config, dim_shape_state, dt, args.scene)

    states_xvfcp = state['state']
    n_frames = 40

    states_xvfcp[0] = np.random.random_sample((5000, 3)) * 0.2 + np.array([0.4, 0.0, 0.4])
    
    for i_grip in range(len(shape_states_)):
        states_xvfcp[4+i_grip][:3] = shape_states_[i_grip][0:3]
        states_xvfcp[4+i_grip][:3] /= 5
        states_xvfcp[4+i_grip][0] += 0.5
        states_xvfcp[4+i_grip][2] += 0.5
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


    for frame in range(1, n_frames):
        print(f'{frame}', end="\r",)

        state = env.get_state()

        # env.simulator.compute_grid_m_kernel(0)
        # gird_m.append(env.simulator.grid_m.to_numpy())

        # ppos.append(env.simulator.x.to_numpy()[0])
        # ppos.append(env.simulator.x.to_numpy())
        
        positions.append(np.concatenate((state['state'][0], np.ones([len(state['state'][0]), 1])), 1))
        sts = np.array(state['state'][4:])
        shape_states.append(np.concatenate([sts[:, :3], sts[:, :3], sts[:, 3:], sts[:, 3:]], axis=1))

        shape_states_ = calc_shape_states(frame * dt, gripper_config, dim_shape_state, dt, args.scene)

        action_after = shape_states_[:, 0:3]
        action_before = shape_states_[:, 3:6]
        action_before /= 5
        action_after /= 5
        action_before[0] += 0.5
        action_before[1] += 0.5
        action_after[0] += 0.5
        action_after[1] += 0.5
        action = action_after - action_before

        env.step((action).flatten() * 100)

        if args.video:
            images.append(env.render('rgb_array'))

        ppos.append(env.simulator.x.to_numpy()[0])

    state = env.get_state()

    # env.simulator.compute_grid_m_kernel(0)
    # gird_m.append(env.simulator.grid_m.to_numpy())

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
            'scene_info': scene_info
            }

    if args.ppos:
       states['x'] = np.array(ppos)

    with open(os.path.join(out_dir, data_name + '.npy'), 'wb') as f:
            np.save(f, states)

    # with open(os.path.join(out_dir, 'gridm_' + data_name + '.npy'), 'wb') as f:
    #         np.save(f, np.array(gird_m))

    if args.video:
        animate(images, os.path.join(out_dir, data_name + '.webm'))


# for data_i in range(args.offset, args.offset + args.n):
#     # YS = 5 + np.random.random()*195
#     # E = 100 + np.random.random()*2900
#     # nu = 0 + np.random.random()*0.45

#     # YS = 5
#     # E = 100 + np.random.random()*2900
#     # nu = 0

#     N_GRID = 5
#     params_range = np.array([[5, 200], [100, 3000], [0, 0.45]])
#     params_offset = (params_range[:, 1] - params_range[:, 0]) / (N_GRID - 1)
#     # YS = params_range[0][0] + params_offset[0] * 3
#     # E = params_range[1][0] + params_offset[1] * 1
#     # nu = params_range[2][0] + params_offset[2] * 3

#     # For finetuning
#     # YS = params_range[0][0] + params_offset[0] * 4
#     # E = params_range[1][0] + params_offset[1] * 1
#     # nu = params_range[2][0] + params_offset[2] * 4

#     # For finetuning
#     YS = params_range[0][0] + params_offset[0] * 1
#     E = params_range[1][0] + params_offset[1] * 4
#     nu = params_range[2][0] + params_offset[2] * 4


#     params = []
#     params.append(YS)
#     params.append(E)
#     params.append(nu)
#     print(data_i, params)
#     data_name = f'{data_i:05d}'
#     simulate_scene(data_i, params, data_name)


benchmark_list = [4012, 4010, 4019, 4022, 4026, 4032, 4024, 4007, 4004, 4003]
materials_list = [[0.5, 3.5, 0.1], [0.5, 0.5, 0.1], [3.5, 3.5, 0.1], [3.5, 0.5, 0.1], [3.5, 0.5, 3.5]]
d_i = 0
for data_i in benchmark_list:
    for mat_i in materials_list:

        N_GRID = 5
        params_range = np.array([[5, 200], [100, 3000], [0, 0.45]])
        params_offset = (params_range[:, 1] - params_range[:, 0]) / (N_GRID - 1)

        YS = params_range[0][0] + params_offset[0] * mat_i[0]
        E = params_range[1][0] + params_offset[1] * mat_i[1]
        nu = params_range[2][0] + params_offset[2] * mat_i[2]


        params = []
        params.append(YS)
        params.append(E)
        params.append(nu)
        print(data_i, params)
        data_name = f'{d_i:05d}'
        simulate_scene(data_i, params, data_name)
        d_i += 1



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