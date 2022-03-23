import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from utils_simulator import *
from utils_glpointrast import *
from glpointrast import perspective, PointRasterizer

from pytorch3d.loss import chamfer_distance

import argparse

parser = argparse.ArgumentParser(description="Create scene with material params")

parser.add_argument("--data_path", help="data path", required=True, type=str)
parser.add_argument("--pretrained_model", help="model path", required=True, type=str)
parser.add_argument("--model_path", help="model path", required=True, type=str)
parser.add_argument("--output_path", help="output path", required=False, type=str, default="")
parser.add_argument("--eval_steps", help="eval_steps", required=False, type=int, default=100000)
parser.add_argument("--num_eval_steps", help="num_eval_steps", required=False, type=int, default=1000)
parser.add_argument("--save_steps", help="save_steps", required=False, type=int, default=1000)
parser.add_argument("--rollout_steps", help="rollout_steps", required=False, type=int, default=100000)
parser.add_argument("--num_steps", help="num_steps", required=False, type=int, default=10000000)
parser.add_argument("--dim", help="dimension of the positions", required=False, type=int, default=6)
parser.add_argument("--force_rollout", help="force eval and rollout in the first step", action='store_true')

args = parser.parse_args()


data_path = args.data_path
model_path = args.model_path
output_path = args.output_path
pos_dim = args.dim

os.system('mkdir -p ' + model_path)
os.system('mkdir -p ' + output_path)



INPUT_SEQUENCE_LENGTH = 6
device = 'cuda'
batch_size = 1
noise_std = 6.7e-4
training_steps = args.num_steps # int(3e6)
log_steps = 5
eval_steps = args.eval_steps
num_eval_steps = args.num_eval_steps
save_steps = args.save_steps
rollout_steps = args.rollout_steps
STD_EPSILON = torch.FloatTensor([1e-8]).to(device)



with open(os.path.join(data_path, 'metadata.json'), 'rt') as f:
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


model_view = np.linalg.inv(
    np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, -0.5],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1],
    ]))

proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(128, 128, 0.03, model_view * rotx(90), proj)

def unproject(proj, depth_image):
    z = proj[2, 2] * depth_image + proj[2, 3]
    w = proj[3, 2] * depth_image
    z_ndc = z / w

    H, W = depth_image.shape
    ndc = np.stack(
        np.meshgrid(
            np.arange(0.5, W + 0.4) * 2 / W - 1,
            np.arange(0.5, H + 0.4)[::-1] * 2 / H - 1,
        )
        + [z_ndc, np.ones_like(z_ndc)],
        axis=-1,
    )

    pos = ndc @ np.linalg.inv(proj).T
    return pos[..., :3] / pos[..., [3]]


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
])).to(device)


proj_matrix_inv = torch.Tensor(np.array([
    [  0.57735027,   0.,          -0.,           0.        ],
    [  0.,           0.57735027,  -0.,           0.        ],
    [  0.,           0.,          -0.,         -10.        ],
    [  0.,           0.,          -1.,          10.        ],
])).to(device)



class SimulatorRolloutNet(torch.nn.Module):
    def __init__(self, simulator, steps):
        super(SimulatorRolloutNet, self).__init__()
        self.simulator = simulator
        self.steps = steps
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.simulator.train()
        
    def forward(self, features):


        features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
        features['particle_type'] = torch.tensor(features['particle_type']).to(device)
        features['step_context'] = torch.tensor(features['step_context']).to(device)
        features['depths'] = torch.tensor(features['depths']).to(device)

        initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
        ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:INPUT_SEQUENCE_LENGTH+self.steps]
        ground_truth_depths = features['depths'][INPUT_SEQUENCE_LENGTH:INPUT_SEQUENCE_LENGTH+self.steps]

        features['step_context'][0][0] = 0.5
        features['step_context'][0][1] = 0.00025
        features['step_context'][0][2] = 0.2


        current_positions = initial_positions
        predictions = []
        for step in range(self.steps):
            next_position = torch.utils.checkpoint.checkpoint(self.simulator.predict_positions, current_positions, features['n_particles_per_example'], features['particle_type'], features['step_context'][0], self.dummy_tensor)
            kinematic_mask = (features['particle_type'] == 3).clone().detach().to(device)
            next_position_ground_truth = ground_truth_positions[:, step]
            kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, pos_dim)
            next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
            predictions.append(next_position)
            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

        non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
        num_non_kinematic = non_kinematic_mask.sum()

        predictions = torch.stack(predictions) # (time, n_nodes, 2)
        ground_truth_positions = ground_truth_positions.permute(1,0,2)
        loss_positions = (predictions - ground_truth_positions) ** 2
        loss_positions = loss_positions.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss_positions = torch.where(non_kinematic_mask.bool(), loss_positions, torch.zeros_like(loss_positions))
        loss_positions = torch.sum(loss_positions) / num_non_kinematic

        predicted_depths = []

        loss = 0
        for step, pred in enumerate(predictions):
            
            n_kinetic_particles = len(features['particle_type'][features['particle_type'] == 1])

            points_predicted = pred[:n_kinetic_particles, 3:].float().contiguous() 
            depth_predicted = raster_func.apply(points_predicted)

            predicted_depths.append(depth_predicted)

            depth_true = ground_truth_depths[step]

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

            loss += chamfer_distance(points_projected_true, points_projected_pred)[0]



            ## Masked IOU is not trivial
            # depth_true_intersect = depth_true.clone()
            # depth_predicted_intersec = depth_predicted.clone()
            # depth_true_intersect[depth_true == 0] = 0
            # depth_true_intersect[depth_predicted == 0] = 0
            # depth_predicted_intersec[depth_true == 0] = 0
            # depth_predicted_intersec[depth_predicted == 0] = 0
            # n_pixel_intersect = len(depth_true_intersect[depth_true_intersect != 0])


            # loss_intersection = torch.sqrt((depth_true_intersect - depth_predicted_intersec)**2).mean()


            # depth_overlap = torch.ones_like(depth_true)
            # depth_overlap[depth_true == 0] = 0
            # depth_overlap[depth_predicted == 0] = 0

            # depth_union = torch.zeros_like(depth_true)
            # depth_union[depth_true != 0] = 1
            # depth_union[depth_predicted != 0] = 1
            # loss_iou = depth_overlap.sum() / depth_union.sum()

            # # loss = loss_intersection + (1.0 - loss_iou)
            # loss = loss_intersection

        loss /= self.steps

        predicted_depths = torch.stack(predicted_depths)


        
        return loss, loss_positions, predictions, ground_truth_positions, predicted_depths, ground_truth_depths


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


def save_optimization_rollout(features, predictions, ground_truth_positions, predicted_depths, ground_truth_depths, step, loss):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]

    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().detach().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
        'global_context': features['step_context'].cpu().detach().numpy(),
        'step': step,
        'loss': loss.to("cpu").detach().numpy(),
    }

    rollout_path = os.path.join(output_path, f'train_{step}')
    os.makedirs(rollout_path, exist_ok=True)

    
    output_dict['metadata'] = metadata
    filename = f'rollout_{step}.pkl'
    filename = os.path.join(rollout_path, filename)
    with open(filename, 'wb') as f:
        pickle.dump(output_dict, f)

    for step_i in range(len(predicted_depths)):
        save_depth_image(predicted_depths[step_i].to("cpu").detach().numpy(), os.path.join(rollout_path, f'predicted_{step_i:05d}.png'))

    for step_i in range(len(ground_truth_depths)):
        save_depth_image(ground_truth_depths[step_i].to("cpu").detach().numpy(), os.path.join(rollout_path, f'true_{step_i:05d}.png'))





def train(simulator):

    writer = SummaryWriter(model_path)

    num_particles = 1006

    num_inference_steps = 25
    model = SimulatorRolloutNet(simulator, num_inference_steps)


    checkpoint = torch.load(args.pretrained_model)
    simulator.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = 0


    ds = prepare_data_from_tfds(data_path=data_path, split='train', is_rollout=True)

    is_first_step = True


    lr_init = 1e-4
    lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)


    # Load checkpoint model
    files = glob.glob(os.path.join(model_path, 'model_*.pth'))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(files) > 0:
        print("Loading ", files[0])
        checkpoint = torch.load(files[0])
        simulator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['epoch']
        
        lr_new = lr_init * (lr_decay ** (step/lr_decay_steps))
        for g in optimizer.param_groups:
            g['lr'] = lr_new


    #for step_i in range(training_steps):
    for example_i, (features, labels) in enumerate(ds):
        # Randomly sample 558 particles from the mpm data to match with the pretrained model
        # features_obj = features['position'][:-360]
        # random_sampled_particles = features_obj[np.random.choice(len(features_obj), size=num_particles, replace=False)]
        # features['position'] =  np.concatenate([random_sampled_particles, features['position'][-360:]])

        # features['n_particles_per_example'][0] = num_particles + 360
        # features['particle_type'] = np.concatenate([features['particle_type'][:num_particles], features['particle_type'][-360:]])
        
        for step_i in range(training_steps):


            


            loss, loss_positions, predictions, ground_truth_positions, predicted_depths, ground_truth_depths = model(features)


            if step % rollout_steps == 0 or (args.force_rollout and is_first_step):
                save_optimization_rollout(features, predictions, ground_truth_positions, predicted_depths, ground_truth_depths, step, loss)

            # if step % log_steps == 0:
            if True:
                writer.add_scalar("training_loss", loss, step)
                writer.add_scalar("particle_loss", loss_positions, step)
                writer.add_scalar("lr", lr_new, step)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            lr_new = lr_init * (lr_decay ** (step/lr_decay_steps))
            for g in optimizer.param_groups:
                g['lr'] = lr_new


            step += 1
            print(f'Training step: {step}/{training_steps}. Loss: {loss}.', end="\r",)
            if step >= training_steps == 0:
                break


            if step % save_steps == 0:
                torch.save({
                    'epoch': step,
                    'model_state_dict': simulator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(model_path, 'model_' + str(step) + '.pth'))

            is_first_step = False


train(simulator)