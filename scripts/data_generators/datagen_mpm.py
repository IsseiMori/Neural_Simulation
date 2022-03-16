from plb.engine.taichi_env import TaichiEnv
import taichi as ti
import numpy as np
from plb.config import load

import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene", help="data type", required=True, type=str)
parser.add_argument("--video", help="write video", action='store_true')
args = parser.parse_args()

root_dir = os.environ.get('NSIMROOT')
out_dir = os.path.join(root_dir, "tmp/Finetune/" + args.scene + "/MPM")
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


def simulate_scene(data_i, params):
    
    env.initialize()

    set_parameters(env, params[0], params[1], params[2])

    state = env.get_state()


    # gripper_config = sample_gripper_config("BendTube", random=False)

    d = np.load(os.path.join(root_dir, "tmp/Finetune/" + args.scene + "/FLEX/{:0>4}.npy".format(str(data_i))), allow_pickle=True).item()
    
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

    for frame in range(n_frames-1):
        print(f'{frame}', end="\r",)

        state = env.get_state()
        
        positions.append(np.concatenate((state['state'][0], np.ones([len(state['state'][0]), 1])), 1))
        sts = np.array(state['state'][4:])
        shape_states.append(np.concatenate([sts[:, :3], sts[:, :3], sts[:, 3:], sts[:, 3:]], axis=1))
        

        # env.step(calc_shape_states_dx(frame * dt, gripper_config))
        env.step((d['shape_states'][0][frame+1][:, 0:3] - d['shape_states'][0][frame][:, 0:3]).flatten() * 100)

        if args.video:
            images.append(env.render('rgb_array'))

    state = env.get_state()

    positions.append(np.concatenate((state['state'][0], np.ones([len(state['state'][0]), 1])), 1))

    sts = np.array(state['state'][4:])
    shape_states.append(np.concatenate([sts[:, :3], sts[:, :3], sts[:, 3:], sts[:, 3:]], axis=1))

    states = {
            'positions': np.array([positions]),
            'shape_states': np.array([shape_states]),
            'E': params[0], 
            'YS': params[1], 
            'nu': params[2],
            'scene_info': d['scene_info']
            }

    with open(os.path.join(out_dir, '{:0>4}.npy'.format(str(data_i))), 'wb') as f:
            np.save(f, states)

    if args.video:
        animate(images, os.path.join(out_dir, '{:0>4}.webm'.format(str(data_i))))


for data_i in range(5000):
    YS = 5 + np.random.random()*195
    E = 100 + np.random.random()*2900
    nu = 0 + np.random.random()*0.45
    params = []
    params.append(YS)
    params.append(E)
    params.append(nu)
    print(params)
    simulate_scene(data_i, params)