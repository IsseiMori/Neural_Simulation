import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

import copy

from utils_simulator import *

import json
from glpointrast import perspective, PointRasterizer
from scipy.spatial.transform import Rotation
from pytorch3d.loss import chamfer_distance

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="00000")
parser.add_argument("--data_path", type=str, required=True) # dir to /depth /raw views.json /nn
parser.add_argument("--metadata", type=str, required=True) # relative from data_path
parser.add_argument("--model", type=str, required=True) # relative from data_path
parser.add_argument("--views", type=str, default="views.json") # relative from data_path
parser.add_argument("--opt_steps", type=int, required=True)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=100)
args = parser.parse_args()

pos_dim = 3


class SimulatorRolloutNet(torch.nn.Module):
    def __init__(self, simulator, ct0, ct1, ct2, steps):
        super(SimulatorRolloutNet, self).__init__()
        self.simulator = simulator
        self.ct0 = torch.tensor(ct0).to(device).requires_grad_(True)
        self.ct1 = torch.tensor(ct1).to(device).requires_grad_(True)
        self.ct2 = torch.tensor(ct2).to(device).requires_grad_(True)
        self.steps = steps
        self.simulator.eval()
    
    def custom(self, module):
        def custom_forward(*inputs):
            out = module.predict_positions(
                inputs[0],
                n_particles_per_example=inputs[1]['n_particles_per_example'],
                particle_types=inputs[1]['particle_type'],
                global_context=inputs[1]['step_context'][0]
            )
            return out
        return custom_forward
            
        
    def forward(self, features):

        features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
        features['particle_type'] = torch.tensor(features['particle_type']).to(device)
        
        ct0_expanded = (self.ct0 * 2900 + 100.0).expand(features['step_context'].shape[0],1)
        ct1_expanded = (self.ct1 * 195 + 5.0).expand(features['step_context'].shape[0],1)
        ct2_expanded = (self.ct2 * 0.45).expand(features['step_context'].shape[0],1)
        step_context = torch.cat([ct0_expanded, ct1_expanded, ct2_expanded], axis=1)
        features['step_context'] = step_context

        initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
        ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:INPUT_SEQUENCE_LENGTH+self.steps]

        current_positions = initial_positions
        predictions = []
        for step in range(self.steps):
            next_position = torch.utils.checkpoint.checkpoint(self.simulator.predict_positions, current_positions, features['n_particles_per_example'], features['particle_type'], features['step_context'][0])

            kinematic_mask = (features['particle_type'] == 3).clone().detach().to(device)
            next_position_ground_truth = ground_truth_positions[:, step]
            kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, pos_dim)
            next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
            predictions.append(next_position)
            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

        predictions = torch.stack(predictions) # (time, n_nodes, 2)
        ground_truth_positions = ground_truth_positions.permute(1,0,2)
        loss = (predictions - ground_truth_positions) ** 2
        # loss = torch.sum(loss[1:])
        loss = torch.sum(loss)
        
        return loss, predictions, ground_truth_positions


from PIL import Image as im
def save_depth_image(depth_data, file_name):
    _min = np.amin(depth_data[depth_data != 0])
    _max = np.amax(depth_data[depth_data != 0])
    # print(_min)
    # print(_max)
    _min = -0.7
    _max = -0.4
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)

def save_optimization_rollout(predicted_depths, ground_truth_depths, step, views, output_path, n_steps, p_pos_seq):

    rollout_path = os.path.join(output_path, f'step_{step}')
    os.makedirs(rollout_path, exist_ok=True)

    with open(os.path.join(rollout_path,  'predicted_pos.npy'), 'wb') as f:
        pred = p_pos_seq.cpu().detach().numpy()
        np.save(f, pred)

    for vi, view in enumerate(views):
        os.makedirs(os.path.join(rollout_path, view['view']), exist_ok=True)

        for step_i in range(n_steps):
            save_depth_image(predicted_depths[step_i][vi].to("cpu").detach().numpy(), os.path.join(rollout_path, view['view'], f'predicted_{step_i:05d}.png'))

        for step_i in range(n_steps):
            save_depth_image(ground_truth_depths[vi][step_i].to("cpu").detach().numpy(), os.path.join(rollout_path, view['view'], f'true_{step_i:05d}.png'))


def plot_progress(title, x_label, y_label, x, y, out, target=None):
    plt.plot(x, y)
    if not target == None:
        plt.axhline(y=target, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(out)
    plt.clf()

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


# copy of vispy_utils.py particlify_box()
def particlify_box(center, half_edge, quat):
    
    pos = []

    # initial spacing
    offset_height = 0.02
    offset_width1 = 0.02
    offset_width2 = 0.02
    
    half_width1 = half_edge[0]
    half_height = half_edge[1]
    half_width2 = half_edge[2]

    particle_count_height = math.ceil(half_height * 2 / offset_height)
    particle_count_width1 = math.ceil(half_width1 * 2 / offset_width1)
    particle_count_width2 = math.ceil(half_width2 * 2 / offset_width2)

    offset_height = half_height * 2 / particle_count_height
    offset_width1 = half_width1 * 2 / particle_count_width1
    offset_width2 = half_width2 * 2 / particle_count_width2


    local_bottom_corner_pos = np.array([-half_width1, -half_height, - half_width2])


    for h in range(0, particle_count_height + 1):
        for w in range(0, particle_count_width1):
            pos.append(local_bottom_corner_pos + np.array([offset_width1 * w, offset_height * h, 0]))
        for w in range(0, particle_count_width2):
            pos.append(local_bottom_corner_pos + np.array([half_width1 * 2, offset_height * h, offset_width2 * w]))
        for w in range(0, particle_count_width1):
            pos.append(local_bottom_corner_pos + np.array([half_width1 * 2 - offset_width2 * w, offset_height * h, half_width2 * 2]))
        for w in range(0, particle_count_width2):
            pos.append(local_bottom_corner_pos + np.array([0, offset_height * h, half_width2 * 2 - offset_width2 * w]))

    for r in range(1, particle_count_width1):
        for c in range(1, particle_count_width2):
            pos.append(local_bottom_corner_pos + np.array([offset_width1 * r, half_height * 2, offset_width2 * c]))
            pos.append(local_bottom_corner_pos + np.array([offset_width1 * r, 0, offset_width2 * c]))
        

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
almost same as vispy_utils.py, respos is different
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
                pos_grip = particlify_box(pos, half_edge[i_grip], quat)

                if restpos: pos_grips.append(np.concatenate([pos_grip, pos_grip], axis=1))
                else : pos_grips.append(pos_grip)

            pos_grips = np.vstack(pos_grips)
            pos_grips = pos_grips.reshape(-1, pos_grips.shape[-1])


            pos_grip_iter.append(np.concatenate((positions[r,i], pos_grips), 0))

        pos_all.append(pos_grip_iter)

    pos_all = np.asarray(pos_all, dtype=np.float64)

    return pos_all


def create_features(d):
    features = {}

    features['position'] = add_grips(d['positions'][:,:,:,:3] , d['shape_states'], d['scene_info'])[0].astype(np.float32)[:-1]

    features['key'] = int(0)

    n_particle_plasticine = len(d['positions'][0,0])
    n_particles_total = features['position'][0].shape[0]
    particle_type = np.ones([n_particles_total], dtype=np.int64)
    particle_type[n_particle_plasticine:] += 2
    features['particle_type'] = particle_type

    features['n_particles_per_example'] = np.array([n_particles_total], dtype=np.int32)

    features['step_context'] = np.ones([len(d['positions'][0]), 3], dtype=np.float32)
    features['is_trajectory'] = np.array([True])

    features['position'] = np.swapaxes(features['position'],0,1)
    features['position'] = np.ascontiguousarray(features['position'], dtype=np.float32)

    return features

def solve():

    output_path = os.path.join(args.data_path, "optimizing_nn", args.experiment, args.data_name)
    os.makedirs(output_path, exist_ok=True)

    data_raw = np.load(os.path.join(args.data_path, "raw", args.data_name + ".npy"), allow_pickle=True).item()


    with open(os.path.join(args.data_path, args.metadata), 'rt') as f:
        metadata = json.loads(f.read())
        
    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
    normalization_stats = {
        'acceleration': {
            'mean':torch.FloatTensor(metadata['acc_mean']).to(device), 
            'std':torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + noise_std**2).to(device),
        }, 
        'velocity': {
            'mean':torch.FloatTensor(metadata['vel_mean']).to(device), 
            'std':torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 + noise_std**2).to(device),
        }, 
    }

    if 'context_mean' in metadata:
        normalization_stats['context'] = {
            'mean':torch.FloatTensor(metadata['context_mean']).to(device), 
            'std':torch.FloatTensor(metadata['context_std']).to(device),
        }



    node_extra = len(metadata['context_mean']) if 'context_mean' in metadata else 0

    simulator = Simulator(
        particle_dimension=pos_dim,
        node_in=(INPUT_SEQUENCE_LENGTH+1) * pos_dim + 16 + node_extra, # 7*2 = 14 + 16 = 30 for 2D
        edge_in=pos_dim + 1, # 3 for 2D
        latent_dim=128,
        num_message_passing_steps=10,
        mlp_num_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        num_particle_types=9,
        particle_type_embedding_size=16,
        device=device,
    )

    if device == 'cuda':
        simulator.cuda()

    checkpoint = torch.load(os.path.join(args.data_path, args.model))
    simulator.load_state_dict(checkpoint['model_state_dict'])


    with open(os.path.join(args.data_path, args.views), 'r') as f:
        views = json.load(f)

    ground_truth_depths_list = []
    for view in views:
        depth_file_name = os.path.join(args.data_path, "depth", view['view'], args.data_name, args.data_name+'.npy')
        print('reading ', depth_file_name)
        depth_feature = torch.tensor(np.load(depth_file_name, allow_pickle=True)).to('cuda')
        print(depth_feature.shape)
        ground_truth_depths_list.append(depth_feature[INPUT_SEQUENCE_LENGTH:])


    raster_funcs = []
    for view in views:
        rot = Rotation.from_euler('xyz', view['rotation'], degrees=True).as_matrix()
        mv_mat = np.zeros((4,4))
        mv_mat[:3, :3] = rot
        mv_mat[:3, 3] = np.array(view['translation'])
        mv_mat[3, 3] = 1

        proj = perspective(np.pi / 3, 1, 0.1, 10)
        raster_func = PointRasterizer(128, 128, 0.01, mv_mat, proj)
        raster_funcs.append(raster_func)

    def unproject_torch(proj, proj_inv, depth_image):
        z = proj[2, 2] * depth_image + proj[2, 3]
        w = proj[3, 2] * depth_image
        z_ndc = z / w

        H, W = depth_image.shape
        ndc = torch.stack(
            [
                torch.tensor(x, dtype=depth_image.dtype, device=depth_image.device)
                for x in np.meshgrid(
                    np.arange(0.5, W + 0.4) * 2 / W - 1,
                    np.arange(0.5, H + 0.4)[::-1] * 2 / H - 1,
                )
            ]
            + [z_ndc, torch.ones_like(z_ndc)],
            axis=-1,
        )
        pos = ndc @ proj_inv.T
        return pos[..., :3] / pos[..., [3]]

    proj_matrix = torch.Tensor(np.array([
        [ 1.73205081,  0.,          0.,          0.        ],
        [ 0.,          1.73205081,  0.,          0.        ],
        [ 0.,          0.,         -1.,         -0.1       ],
        [ 0.,          0.,         -1.,          0.        ],
    ])).to('cuda')


    proj_matrix_inv = torch.Tensor(np.array([
        [  0.57735027,   0.,          -0.,           0.        ],
        [  0.,           0.57735027,  -0.,           0.        ],
        [  0.,           0.,          -0.,         -10.        ],
        [  0.,           0.,          -1.,          10.        ],
    ])).to('cuda')


    def forward(model, features_, epoch):
        features = copy.deepcopy(features_)

        loss_pos, predictions, ground_truth_positions = model(features)

        p_pos_seq = predictions[:, :5000]
        # print('p_pos_seq', p_pos_seq.shape)

        predicted_depths = []

        loss_seq = []
        for step in range(n_steps):

            points_predicted = p_pos_seq[step]
            points_predicted = points_predicted - torch.tensor([0.5, 0.1, 0.5]).to('cuda')

            points_predicted = points_predicted.float().contiguous()

            predicted_depths_views = []

            for vi, view in enumerate(views):

                depth_predicted = raster_funcs[vi].apply(points_predicted)

                predicted_depths_views.append(depth_predicted)

                depth_true = ground_truth_depths_list[vi][step]

                # save_depth_image(depth_predicted.to("cpu").detach().numpy(), os.path.join(output_path, "{:0>4}.png".format(str(step))))

                # Set background to a reasonable depth
                depth_predicted[depth_predicted <= -100000] = -100
                depth_true[depth_true==0] = -100

                points_projected_true = unproject_torch(proj_matrix, proj_matrix_inv, depth_true)
                points_projected_pred = unproject_torch(proj_matrix, proj_matrix_inv, depth_predicted)

                points_projected_true = torch.flatten(points_projected_true, start_dim=0, end_dim=1)
                points_projected_pred = torch.flatten(points_projected_pred, start_dim=0, end_dim=1)
                points_projected_true = points_projected_true[None, :]
                points_projected_pred = points_projected_pred[None, :]

                loss_seq.append(chamfer_distance(points_projected_true, points_projected_pred, point_reduction="sum")[0])

            predicted_depths.append(torch.stack(predicted_depths_views))

        loss_seq = torch.stack(loss_seq)
        loss = torch.sum(loss_seq)

        predicted_depths = torch.stack(predicted_depths)
        ground_truth_depths = torch.stack(ground_truth_depths_list)

        if epoch == 0 or epoch % 10 == 0 or epoch == -1 or epoch == -2:
            save_optimization_rollout(predicted_depths, ground_truth_depths, epoch, views, output_path, n_steps, predictions)

        return loss, loss_seq




    steps = []
    ct0_vals = []
    ct1_vals = []
    ct2_vals = []
    ct0_grad_vals = []
    ct1_grad_vals = []
    ct2_grad_vals = []
    loss_vals = []
    loss_seq_vals = []

    if device == 'cuda':
        simulator.cuda()

    eval_loss = []


    n_steps = args.opt_steps

    # YS, E, nu
    m_YS_target = float((data_raw['YS'] - 5)/195)
    m_E_target = float((data_raw['E']-100)/2900)
    m_nu_target = float(data_raw['nu']/0.45)
    print("target material", m_YS_target, " ", m_E_target, " ", m_nu_target)
    target = [m_YS_target, m_E_target, m_nu_target]


    print("Starting system identification...")

    ct0 = 0.5
    ct1 = 0.5
    ct2 = 0.5

    best_material = None
    best_loss = 1e10

    features = create_features(data_raw)

    with torch.no_grad():
        model = SimulatorRolloutNet(simulator, target[0], target[1], target[2], n_steps)
        loss_gt, loss_seq_gt = forward(model, features, -1)
        loss_gt = loss_gt.to("cpu").detach().numpy()
        loss_seq_gt = loss_seq_gt.to("cpu").detach().numpy()


    model = SimulatorRolloutNet(simulator, ct0, ct1, ct2, n_steps)
    optimizer = torch.optim.Adam([model.ct0, model.ct1, model.ct2], lr=args.lr)

    
    for epoch in range(args.num_steps):

        loss, loss_seq = forward(model, features, epoch)


        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        print('iter', epoch)
        print('loss =', loss.to("cpu").detach().numpy())
        print('YS =', model.ct0.detach().to("cpu").numpy(), 'E =', model.ct1.detach().to("cpu").numpy(), 'nu =', model.ct2.detach().to("cpu").numpy())
        print('Gradient YS =', model.ct0.grad.detach().to("cpu").numpy(), 'E =', model.ct1.grad.detach().to("cpu").numpy(), 'nu =', model.ct2.grad.detach().to("cpu").numpy())

        if np.isnan(model.ct0.grad.detach().to("cpu").numpy()) \
            or np.isnan(model.ct1.grad.detach().to("cpu").numpy()) \
            or np.isnan(model.ct2.grad.detach().to("cpu").numpy()):
            exit()

        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            best_material = [model.ct0.detach().to("cpu").numpy(), model.ct1.detach().to("cpu").numpy(), model.ct2.detach().to("cpu").numpy()]

        with torch.no_grad():
            model.ct0.clamp_(0, 1)
            model.ct1.clamp_(0, 1)
            model.ct2.clamp_(0, 1)


        steps.append(epoch)
        ct0_vals.append(model.ct0.detach().to("cpu").numpy().item())
        ct1_vals.append(model.ct1.detach().to("cpu").numpy().item())
        ct2_vals.append(model.ct2.detach().to("cpu").numpy().item())
        ct0_grad_vals.append(model.ct0.grad.detach().to("cpu").numpy().item())
        ct1_grad_vals.append(model.ct1.grad.detach().to("cpu").numpy().item())
        ct2_grad_vals.append(model.ct2.grad.detach().to("cpu").numpy().item())
        loss_vals.append(loss.to("cpu").detach().numpy())
        loss_seq_vals.append(loss_seq.to("cpu").detach().numpy())

        with open(os.path.join(output_path, 'loss'), 'wb') as f:
            loss_info = {
                "ct0": np.array(ct0_vals),
                "ct1": np.array(ct1_vals),
                "ct2": np.array(ct2_vals),
                "ct0_grad": np.array(ct0_grad_vals),
                "ct1_grad": np.array(ct1_grad_vals),
                "ct2_grad": np.array(ct2_grad_vals),
                "loss": np.array(loss_vals),
                "loss_seq": np.array(loss_seq_vals),
                "loss_gt": loss_gt,
                "loss_seq_gt": loss_seq_gt,
            }
            np.save(f, loss_info)

        plot_progress('Optimizing YS value', 'steps', 'YS', np.array(steps), np.array(ct0_vals), os.path.join(output_path, 'ct0.png'), target[0])
        plot_progress('Optimizing E value', 'steps', 'E', np.array(steps), np.array(ct1_vals), os.path.join(output_path, 'ct1.png'), target[1])
        plot_progress('Optimizing nu value', 'steps', 'nu', np.array(steps), np.array(ct2_vals), os.path.join(output_path, 'ct2.png'), target[2])
        plot_progress('YS Gradient', 'steps', 'YS grad', np.array(steps), np.array(ct0_grad_vals), os.path.join(output_path, 'ct0_grad.png'))
        plot_progress('E Gradient', 'steps', 'E grad', np.array(steps), np.array(ct1_grad_vals), os.path.join(output_path, 'ct1_grad.png'))
        plot_progress('nu Gradient', 'steps', 'nu grad', np.array(steps), np.array(ct2_grad_vals), os.path.join(output_path, 'ct2_grad.png'))
        plot_progress('Loss while optimization', 'steps', 'Loss', np.array(steps), np.array(loss_vals), os.path.join(output_path, 'loss.png'), loss_gt)


    with torch.no_grad():
        model = SimulatorRolloutNet(simulator, best_material[0], best_material[1], best_material[2], n_steps)
        loss, loss_seq = forward(model, features, -2)


def main():
    solve()

if __name__ == "__main__":
    main()