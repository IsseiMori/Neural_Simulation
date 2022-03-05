import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils_simulator import *
from utils_glpointrast import *
from glpointrast import perspective, PointRasterizer



root_dir = os.environ.get('NSIMROOT')

data_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/finetuned/data") # dataset with mpm particles and rendered depth
model_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/models") # pretrained model
model_finetuned_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/finetuned/models") # save the finetuned model here
rollouts_path = os.path.join(root_dir, "tmp/FLEX_RiceGrip/finetuned/rollouts")
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


class SimulatorRolloutNet(torch.nn.Module):
    def __init__(self, simulator, ct0, ct1, ct2, steps):
        super(SimulatorRolloutNet, self).__init__()
        self.simulator = simulator
        self.steps = steps
        self.simulator.eval()
            
        
    def forward(self, features):

        features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
        features['particle_type'] = torch.tensor(features['particle_type']).to(device)
        features['step_context'] = torch.tensor(features['step_context']).to(device)

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
        loss = torch.sum(loss)
        
        return loss, predictions, ground_truth_positions





model_view = np.linalg.inv(
    np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, -0.5],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1],
    ]))

proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(512, 512, 0.03, model_view * rotx(90), proj)





def train(simulator):

    writer = SummaryWriter(model_finetuned_path)


    num_inference_steps = 25
    model = SimulatorRolloutNet(simulator, num_inference_steps)


    checkpoint = torch.load(os.path.join(model_path, 'model_300000.pth'))
    simulator.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = 0


    ds = prepare_data_from_tfds(data_path=data_path, split='train', is_rollout=True)


    lr_init = 1e-4
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)


    ## Load checkpoint
    # if os.path.exists(os.path.join(model_path, 'model.pth')):
    #     checkpoint = torch.load(os.path.join(model_path, 'model.pth'))
    #     simulator.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     step = checkpoint['epoch']
        
    #     lr_new = lr_init * (lr_decay ** (step/lr_decay_steps))
    #     for g in optimizer.param_groups:
    #         g['lr'] = lr_new


    for example_i, (features, labels) in enumerate(ds):

        exit()

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
