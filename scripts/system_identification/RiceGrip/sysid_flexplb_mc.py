import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils_simulator import *
from glpointrast import perspective, PointRasterizer



root_dir = os.environ.get('NSIMROOT')

data_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/predicted_data")
mpm_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/mpm_data")
model_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/models")
output_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/sysid")
pos_dim = 6


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

checkpoint = torch.load(os.path.join(model_path, 'model_500000.pth'))
simulator.load_state_dict(checkpoint['model_state_dict'])

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
        
        ct0_expanded = (self.ct0 * (0.7 - 0.3) + 0.3).expand(features['step_context'].shape[0],1)
        ct1_expanded = (self.ct1 * (0.0005 - 0.00001) + 0.00001).expand(features['step_context'].shape[0],1)
        ct2_expanded = (self.ct2 * (0.3 - 0.1) + 0.1).expand(features['step_context'].shape[0],1)
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

# infer(simulator, split='test')

OPTIMIZATION_PATH = "rollouts_mpm"
os.makedirs(OPTIMIZATION_PATH, exist_ok=True)

def save_optimization_rollout(features, predictions, ground_truth_positions, step, loss):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]

    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().detach().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
        'global_context': features['step_context'].cpu().detach().numpy(),
        'step': step,
        'loss': loss.to("cpu").detach().numpy(),
        'step_context_true': features['step_context_true'],
    }

    
    output_dict['metadata'] = metadata
    filename = f'rollout_{step}.pkl'
    filename = os.path.join(OPTIMIZATION_PATH, filename)
    with open(filename, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f'{filename}', end="\r",)

steps = []
ct0_vals = []
ct1_vals = []
ct2_vals = []
loss_vals = []

if device == 'cuda':
    simulator.cuda()

ds = prepare_data_from_tfds(data_path=data_path, split='0', is_rollout=True)
ds_mpm = prepare_data_from_tfds(data_path=mpm_path, split='4', is_rollout=True)

eval_loss = []

import math
def sincos(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)

def rotx(a):
    s, c = sincos(a)
    return np.matrix([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])

def roty(a):
    s, c = sincos(a)
    return np.matrix([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])

def rotz(a):
    s, c = sincos(a)
    return np.matrix([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

model_view = np.linalg.inv(
    np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, -0.5],
        [0, 0, 1, 0.5],
        [0, 0, 0, 1],
    ]))

proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(512, 512, 0.03, model_view * rotx(90), proj)


from PIL import Image as im
def save_depth_image(depth_data, file_name):
    # _min = np.amin(depth_data[depth_data != 0])
    # _max = np.amax(depth_data[depth_data != 0])
    _min = -0.5
    _max = -0.2
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)


depth_true_seq = np.load('mpm_depth.npy')



num_inference_steps = 25


print("Starting system identification...")
for example_i, (features, labels) in enumerate(ds):
    for example_i_mpm, (features_mpm, labels_mpm) in enumerate(ds_mpm):

        # ct0 = features['step_context'][0][0]
        # ct1 = 2.7973418e-04 # features['step_context'][0][1]
        # ct2 = features['step_context'][0][2]


        ct0 = 1.0
        ct1 = 0.0
        ct2 = 0.0


        model = SimulatorRolloutNet(simulator, ct0, ct1, ct2, num_inference_steps)
        optimizer = torch.optim.Adam([model.ct0, model.ct1, model.ct2], lr=0.05)
        

        # replace grip position data
        features_mpm_obj = features_mpm['position'][:-360]
        random_sampled_particles = features_mpm_obj[np.random.choice(len(features_mpm_obj), size=558, replace=False)]
        features['position'][:-360, :num_inference_steps+6] = random_sampled_particles[:,:num_inference_steps+6]


        features['position'][-360:, :num_inference_steps+6] = features_mpm['position'][-360:, :num_inference_steps+6]

        features['step_context_true'] = features['step_context']
        
        for epoch in range(50):

            loss_pos, predictions, ground_truth_positions = model(features)

            n_kinetic_particles = len(features['particle_type'][features['particle_type'] == 1])



            points_predicted = predictions[-1,:n_kinetic_particles, 3:].float().contiguous() 
            depth_predicted = raster_func.apply(points_predicted)
            depth_predicted[depth_predicted <= -100000] = 0



            depth_true = depth_true_seq[num_inference_steps]
            depth_true = torch.tensor(depth_true).to(device)

            depth_true_intersect = depth_true.clone()
            depth_predicted_intersec = depth_predicted.clone()
            depth_true_intersect[depth_true == 0] = 0
            depth_true_intersect[depth_predicted == 0] = 0
            depth_predicted_intersec[depth_true == 0] = 0
            depth_predicted_intersec[depth_predicted == 0] = 0
            n_pixel_intersect = len(depth_true_intersect[depth_true_intersect != 0])



            loss_intersection = torch.sqrt((depth_true_intersect - depth_predicted_intersec)**2).mean()


            depth_overlap = torch.ones_like(depth_true)
            depth_overlap[depth_true == 0] = 0
            depth_overlap[depth_predicted == 0] = 0

            depth_union = torch.zeros_like(depth_true)
            depth_union[depth_true != 0] = 1
            depth_union[depth_predicted != 0] = 1
            loss_iou = depth_overlap.sum() / depth_union.sum()

            loss = loss_intersection + (1.0 - loss_iou)


            # loss = ((depth_true - depth_predicted)**2).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            steps.append(epoch)


            ct0 = model.ct0.to("cpu").detach().numpy() * (0.7 - 0.3) + 0.3
            ct1 = model.ct1.to("cpu").detach().numpy() * (0.0005 - 0.00001) + 0.00001
            ct2 = model.ct2.to("cpu").detach().numpy() * (0.3 - 0.1) + 0.1

            ct0_vals.append(ct0)
            ct1_vals.append(ct1)
            ct2_vals.append(ct2)
            loss_vals.append(loss.to("cpu").detach().numpy())


            
            if epoch == 0 or epoch % 10 == 0:
                save_optimization_rollout(features, predictions, ground_truth_positions, epoch, loss)
                save_depth_image(depth_predicted.to("cpu").detach().numpy(), 'images_mpm/depth_predicted' + str(epoch) + '.png')

                output_dict = {
                    'predicted_rollout': predictions.cpu().detach().numpy(),
                    'ground_truth_rollout': np.transpose(features_mpm['position'], (1, 0, 2))[6:num_inference_steps+6],
                    'particle_types': features['particle_type'].cpu().detach().numpy(),
                    'global_context': features['step_context'].cpu().detach().numpy(),
                    'metadata': metadata
                }
                print(predictions.shape)
                filename = 'images_mpm/' + str(epoch) + '.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(output_dict, f)

            exit()

            # print(f'epoch={epoch}, ct0={'{:,5}'.format}, ct1={ct1}, ct2={ct2}, LossInt={loss_intersection}, LossIOU={(1.0 - loss_iou)}')
            print(f'epoch={epoch}, clusterStiffness={ct0:.5}, clusterPlasticThreshold={ct1:.5}, clusterPlasticCreep={ct2:.5}, LossInt={loss_intersection:.5}, LossIOU={(1.0 - loss_iou):.5}, Loss={loss:.5}')




xpoints = np.array(steps)
ypoints = np.array(ct0_vals)

plt.plot(xpoints, ypoints)
plt.title('Optimizing clusterStiffness  value')
plt.xlabel('steps')
plt.ylabel('clusterStiffness')
plt.savefig('images_mpm/ct0.png')
plt.clf()

xpoints = np.array(steps)
ypoints = np.array(ct1_vals)

plt.plot(xpoints, ypoints)
plt.title('Optimizing clusterPlasticThreshold  value')
plt.xlabel('steps')
plt.ylabel('clusterPlasticThreshold')
plt.savefig('images_mpm/ct1.png')
plt.clf()

xpoints = np.array(steps)
ypoints = np.array(ct2_vals)

plt.plot(xpoints, ypoints)
plt.title('Optimizing clusterPlasticCreep   value')
plt.xlabel('steps')
plt.ylabel('clusterPlasticCreep')
plt.savefig('images_mpm/ct2.png')
plt.clf()

xpoints = np.array(steps)
ypoints = np.array(loss_vals)

plt.plot(xpoints, ypoints)
plt.title('Loss while optimization')
plt.xlabel('steps')
plt.ylabel('Loss')
plt.savefig('images_mpm/loss.png')