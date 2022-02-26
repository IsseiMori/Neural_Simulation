import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph

from torch.utils.tensorboard import SummaryWriter

from glpointrast import perspective, PointRasterizer

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

INPUT_SEQUENCE_LENGTH = 6
device = 'cuda'
batch_size = 1
noise_std = 6.7e-4
training_steps = int(3e6)
log_steps = 5
eval_steps = 100 #100000
num_eval_steps = 10 # 1000
save_steps = 10 # 1000
rollout_steps = 100 #100000
STD_EPSILON = torch.FloatTensor([1e-8]).to(device)

# data_path = "/imori-fast-vol/tmp/data/RiceGrip/FLEX2/predicted/test_900000"
# mpm_path = "/imori-fast-vol/tmp/data/RiceGrip/plb"
# model_path = "/imori-fast-vol/tmp/models/RiceGrip/FLEX2"
# output_path = "/imori-fast-vol/tmp/rollouts/RiceGrip/FLEX2/sysid"

data_path = "/imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/predicted_data"
mpm_path = "/imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/mpm_data"
model_path = "/imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/models"
output_path = "/imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/sysid"
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

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

class Encoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(*[build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr='add')
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features

class Processor(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Processor, self).__init__(aggr='max')
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in, 
                node_out=node_out,
                edge_in=edge_in, 
                edge_out=edge_out,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])

    def forward(self, x, edge_index, e_features):
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features

class Decoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out)

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)

class EncodeProcessDecode(nn.Module):
    def __init__(
        self, 
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            node_in=node_in, 
            node_out=latent_dim,
            edge_in=edge_in, 
            edge_out=latent_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=latent_dim, 
            node_out=latent_dim,
            edge_in=latent_dim, 
            edge_out=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=latent_dim,
            node_out=node_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        return x

class Simulator(nn.Module):
    def __init__(
        self,
        particle_dimension,
        node_in,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        num_particle_types,
        particle_type_embedding_size,
        device='cuda',
    ):
        super(Simulator, self).__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types

        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=particle_dimension,
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self):
        pass

    def _build_graph_from_raw(self, position_sequence, n_particles_per_example, particle_types, global_context):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence)
        # senders and receivers are integers of shape (E,)
        senders, receivers = self._compute_connectivity(most_recent_position, n_particles_per_example, self._connectivity_radius)
        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        boundaries = torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        distance_to_lower_boundary = (most_recent_position - boundaries[:, 0][None])
        distance_to_upper_boundary = (boundaries[:, 1][None] - most_recent_position)
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)
        
        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            # particle_type_embeddings = particle_type_embeddings.to(device)
            node_features.append(particle_type_embeddings)

        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            epsilon_tensor = STD_EPSILON.expand(context_stats['std'].shape)
            global_context = (global_context - context_stats['mean']) / torch.maximum(context_stats['std'], epsilon_tensor)
            global_context = global_context.repeat(particle_types.shape[0], 1)
            node_features.append(global_context)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        return torch.cat(node_features, dim=-1), torch.stack([senders, receivers]), torch.cat(edge_features, dim=-1)

    def _compute_connectivity(self, node_features, n_particles_per_example, radius, add_self_edges=True):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_example)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats['std']
        ) + acceleration_stats['mean']

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(self, current_positions, n_particles_per_example, particle_types, global_context):
        node_features, edge_index, e_features = self._build_graph_from_raw(current_positions, n_particles_per_example, particle_types, global_context)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions)
        return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_example, particle_types, global_context):
        noisy_position_sequence = position_sequence + position_sequence_noise
        node_features, edge_index, e_features = self._build_graph_from_raw(noisy_position_sequence, n_particles_per_example, particle_types, global_context)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(next_position_adjusted, noisy_position_sequence)
        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats['mean']) / acceleration_stats['std']
        return normalized_acceleration

    def save(self, path='model.pth'):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def prepare_data_from_tfds(data_path, split='train', is_rollout=False, batch_size=1):
    import functools
    import tensorflow.compat.v1 as tf
    import tensorflow_datasets as tfds
    import reading_utils
    import tree
    from tfrecord.torch.dataset import TFRecordDataset
    def prepare_inputs(tensor_dict):
        pos = tensor_dict['position']
        pos = tf.transpose(pos, perm=[1, 0, 2])
        target_position = pos[:, -1]
        tensor_dict['position'] = pos[:, :-1]
        num_particles = tf.shape(pos)[0]
        tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
        if 'step_context' in tensor_dict:
            tensor_dict['step_context'] = tensor_dict['step_context'][-2]
            tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
        return tensor_dict, target_position
    def batch_concat(dataset, batch_size):
        windowed_ds = dataset.window(batch_size)
        initial_state = tree.map_structure(lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),dataset.element_spec)
        def reduce_window(initial_state, ds):
            return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
        return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))
    def prepare_rollout_inputs(context, features):
        out_dict = {**context}
        pos = tf.transpose(features['position'], [1, 0, 2])
        target_position = pos[:, -1]
        out_dict['position'] = pos[:, :-1]
        out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
        if 'step_context' in features:
            out_dict['step_context'] = features['step_context']
        out_dict['is_trajectory'] = tf.constant([True], tf.bool)
        return out_dict, target_position
    
    metadata = _read_metadata(data_path)

    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:    
        split_with_window = functools.partial(
            reading_utils.split_trajectory,
            window_length=6 + 1)
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        ds = ds.repeat()
        ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
    return ds

def eval_single_rollout(simulator, features, num_steps, device):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
    
    non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
    
    current_positions = initial_positions
    predictions = []
    for step in range(num_steps):
        next_position = simulator.predict_positions(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'],
            global_context=features['step_context'][step]
        ) # (n_nodes, 2)
        # Update kinematic particles from prescribed trajectory
        kinematic_mask = (features['particle_type'] == 3).clone().detach().to(device)
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, pos_dim)
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
    predictions = torch.stack(predictions) # (time, n_nodes, 2)
    ground_truth_positions = ground_truth_positions.permute(1,0,2)
    loss = (predictions - ground_truth_positions) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()

    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    
    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
        'global_context': features['step_context'].cpu().numpy(),
    }
    return output_dict, loss

def eval_rollout(ds, simulator, rollout_path, num_steps, num_eval_steps=10, save_results=False, device='cuda'):
    eval_loss = []
    i = 0
    simulator.eval()
    with torch.no_grad():
        for example_i, (features, labels) in enumerate(ds):
            features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device)
            features['step_context'] = torch.tensor(features['step_context']).to(device)
            labels = torch.tensor(labels).to(device)
            example_rollout, loss = eval_single_rollout(simulator, features, num_steps, device)
            example_rollout['metadata'] = metadata
            eval_loss.append(loss)
            if save_results:
                example_rollout['metadata'] = metadata
                filename = f'rollout_{example_i}.pkl'
                filename = os.path.join(rollout_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_rollout, f)
                print(f'{filename}', end="\r",)
            i += 1
            if i >= num_eval_steps:
                break
    simulator.train()
    return torch.stack(eval_loss).mean(0)

def one_step_estimator(simulator, features, labels, device, noise_std=0):
    features['position'] = torch.tensor(features['position']).to(device)
    features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
    features['particle_type'] = torch.tensor(features['particle_type']).to(device)
    features['step_context'] = torch.tensor(features['step_context']).to(device)
    labels = torch.tensor(labels).to(device)

    sampled_noise = get_random_walk_noise_for_position_sequence(features['position'], noise_std_last_step=noise_std).to(device)
    non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    pred, target = simulator.predict_accelerations(
        next_position=labels, 
        position_sequence_noise=sampled_noise, 
        position_sequence=features['position'], 
        n_particles_per_example=features['n_particles_per_example'], 
        particle_types=features['particle_type'],
        global_context=features['step_context']
    )
    loss = (pred - target) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()

    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    
    return loss

def eval_one_step(ds, simulator, device, noise_std):
    
    eval_loss = []
    
    step = 0
    
    simulator.eval()
    with torch.no_grad():
        try:
            for features, labels in ds:
                loss = one_step_estimator(simulator, features, labels, device, noise_std)
                eval_loss.append(loss)
                print(f'Eval step: {step}. Loss: {loss}.', end="\r",)
                step += 1
                
                if step >= num_eval_steps:
                    break
                
        except KeyboardInterrupt:
            pass
    
    simulator.train()
    
    return torch.stack(eval_loss).mean(0)
    
def train(simulator):

    writer = SummaryWriter(model_path)

    lr_init = 1e-4
    lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)

    ds = prepare_data_from_tfds(data_path, split='train', batch_size=batch_size)
    ds_eval = prepare_data_from_tfds(data_path, split='test', batch_size=batch_size)
    # ds_eval = prepare_data_from_tfds(data_path, split='test', is_rollout=True)
    
    step = 0
    
    # if model_path is not None:
    if os.path.exists(os.path.join(model_path, 'model.pth')):
        checkpoint = torch.load(os.path.join(model_path, 'model.pth'))
        simulator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['epoch']
        
        lr_new = lr_init * (lr_decay ** (step/lr_decay_steps))
        for g in optimizer.param_groups:
            g['lr'] = lr_new

    try:
        for features, labels in ds:
            
            loss = one_step_estimator(simulator, features, labels, device, noise_std)

            if step % log_steps == 0:
                writer.add_scalar("training_loss", loss, step)
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

            if step % eval_steps == 0:
                eval_loss = eval_one_step(ds_eval, simulator, device, noise_std)
                writer.add_scalar("eval_loss", eval_loss, step)
            
            if step % rollout_steps == 0:
                
                rollout_path = os.path.join(output_path, f'train_{step}')
                os.makedirs(rollout_path, exist_ok=True)
                
                ds = prepare_data_from_tfds(os.path.join(data_path, 'rollouts'), split='train', is_rollout=True)
                eval_rollout(ds, simulator, rollout_path, num_steps=num_steps, save_results=True, device=device, num_eval_steps=10)
                
                rollout_path = os.path.join(output_path, f'test_{step}')
                os.makedirs(rollout_path, exist_ok=True)
                
                ds = prepare_data_from_tfds(os.path.join(data_path, 'rollouts'), split='test', is_rollout=True)
                eval_rollout(ds, simulator, rollout_path, num_steps=num_steps, save_results=True, device=device, num_eval_steps=10)
            

            if step % save_steps == 0:
                torch.save({
                    'epoch': step,
                    'model_state_dict': simulator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(model_path, 'model.pth'))

    except KeyboardInterrupt:
        pass

    # simulator.save(LOG_DIR+'model.pth')

def infer(simulator, num_eval_steps=10, split='test'):
    # ds = prepare_data_from_tfds(os.path.join(data_path, 'rollouts'), split='test', is_rollout=True)
    ds = prepare_data_from_tfds(data_path, split=split, is_rollout=True)
    eval_rollout(ds, simulator, output_path, num_steps=num_steps, save_results=True, device=device, num_eval_steps=num_eval_steps)

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

ds = prepare_data_from_tfds(data_path=data_path, split='1', is_rollout=True)
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
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]))

proj = perspective(np.pi / 3, 1, 0.1, 10)
raster_func = PointRasterizer(256, 256, 0.01, model_view * rotx(90), proj)


from PIL import Image as im
def save_depth_image(depth_data, file_name):
    # _min = np.amin(depth_data[depth_data > -100000])
    # _max = np.amax(depth_data[depth_data > -100000])
    _min = -0.8986716
    _max = -0.7356762
    # print(np.amax(depth_data[depth_data > -100000]))
    # print(np.amin(depth_data[depth_data > -100000]))
    # print(depth_data[depth_data > -100000].mean())
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)


num_inference_steps = 25


print("Find true position")
for example_i, (features, labels) in enumerate(ds_mpm):

    print(features['position'].shape)


    n_kinetic_particles = len(features['particle_type'][features['particle_type'] == 1])
    points_true = features['position'][:n_kinetic_particles,num_inference_steps, 3:]
    # depth_true = raster_func.apply(torch.tensor(points_true).to(device))


    import open3d
    import trimesh
    import numpy as np
    from tqdm import tqdm


    radius = 0.01  # point radius
    dx = 0.004  # marching cube grid size

    points = points_true
    bbox = np.array([points.min(0) - radius * 1.1, points.max(0) + radius * 1.1])

    dim = np.ceil((bbox[1] - bbox[0]) / dx)
    bbox = np.array([bbox[0], bbox[0] + dim * dx])

    dim = dim.astype(int)

    point_field = np.stack(
        np.meshgrid(
            np.linspace(bbox[0, 0], bbox[1, 0], dim[0]),
            np.linspace(bbox[0, 1], bbox[1, 1], dim[1]),
            np.linspace(bbox[0, 2], bbox[1, 2], dim[2]),
            indexing="ij",
        ),
        -1,
    )

    dist = np.ones(point_field.shape[:3]) * 1e6


    for p in tqdm(points):
        d = np.linalg.norm(point_field - p, axis=-1)
        dist = np.minimum(dist, d)

    dist -= radius

    with open('dist.npy', 'wb') as f:
        np.save(f, dist)


    # save_depth_image(depth_true.to("cpu").detach().numpy(), 'images_mpm/depth_true.png')





# raster_func = PointRasterizer(256, 256, 0.01, model_view * rotx(90), proj)


# print("Starting system identification...")
# for example_i, (features, labels) in enumerate(ds):
#     for example_i_mpm, (features_mpm, labels_mpm) in enumerate(ds_mpm):

#         # ct0 = features['step_context'][0][0]
#         # ct1 = 2.7973418e-04 # features['step_context'][0][1]
#         # ct2 = features['step_context'][0][2]

#         ct0 = 0.5
#         ct1 = 0.5
#         ct2 = 0.5


#         model = SimulatorRolloutNet(simulator, ct0, ct1, ct2, num_inference_steps)
#         optimizer = torch.optim.Adam([model.ct0, model.ct1, model.ct2], lr=0.01)
        
#         print(features_mpm['step_context'][0])

#         print(features['position'].shape)
#         print(features_mpm['position'].shape)

#         # replace grip position data
#         features['position'][-360:, :27] = features_mpm['position'][-360:, :27]

#         features['step_context_true'] = features['step_context']
        
#         for epoch in range(100):

#             loss_pos, predictions, ground_truth_positions = model(features)

#             n_kinetic_particles = len(features['particle_type'][features['particle_type'] == 1])

#             points_predicted = predictions[-1,:n_kinetic_particles, 3:].float().contiguous() 
#             depth_predicted = raster_func.apply(points_predicted)



#             depth_true_intersect = depth_true.clone()
#             depth_predicted_intersec = depth_predicted.clone()
#             depth_true_intersect[depth_true == -100000] = 0
#             depth_true_intersect[depth_predicted == -100000] = 0
#             depth_predicted_intersec[depth_true == -100000] = 0
#             depth_predicted_intersec[depth_predicted == -100000] = 0
#             n_pixel_intersect = len(depth_true_intersect[depth_true_intersect != 0])

#             loss_intersection = ((depth_true_intersect - depth_predicted_intersec)**2).sum() / n_pixel_intersect


#             depth_overlap = torch.ones_like(depth_true)
#             depth_overlap[depth_true == -100000] = 0
#             depth_overlap[depth_predicted == -100000] = 0

#             depth_union = torch.zeros_like(depth_true)
#             depth_union[depth_true != -100000] = 1
#             depth_union[depth_predicted != -100000] = 1
#             loss_iou = depth_overlap.sum() / depth_union.sum()

#             loss = loss_intersection + (1.0 - loss_iou)


#             # loss = ((depth_true - depth_predicted)**2).mean()
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             optimizer.step()

#             steps.append(epoch)


#             ct0 = model.ct0.to("cpu").detach().numpy() * (0.7 - 0.3) + 0.3
#             ct1 = model.ct1.to("cpu").detach().numpy() * (0.0005 - 0.00001) + 0.00001
#             ct2 = model.ct2.to("cpu").detach().numpy() * (0.3 - 0.1) + 0.1

#             ct0_vals.append(ct0)
#             ct1_vals.append(ct1)
#             ct2_vals.append(ct2)
#             loss_vals.append(loss.to("cpu").detach().numpy())


            
#             if epoch == 0 or epoch % 10 == 0:
#                save_optimization_rollout(features, predictions, ground_truth_positions, epoch, loss)
#                save_depth_image(depth_predicted.to("cpu").detach().numpy(), 'images_mpm/depth_predicted' + str(epoch) + '.png')

#             # print(f'epoch={epoch}, ct0={'{:,5}'.format}, ct1={ct1}, ct2={ct2}, LossInt={loss_intersection}, LossIOU={(1.0 - loss_iou)}')
#             print(f'epoch={epoch}, clusterStiffness={ct0:.5}, clusterPlasticThreshold={ct1:.5}, clusterPlasticCreep={ct2:.5}, LossInt={loss_intersection:.5}, LossIOU={(1.0 - loss_iou):.5}, Loss={loss:.5}')




# xpoints = np.array(steps)
# ypoints = np.array(ct0_vals)

# plt.plot(xpoints, ypoints)
# plt.title('Optimizing clusterStiffness  value')
# plt.xlabel('steps')
# plt.ylabel('clusterStiffness')
# plt.savefig('images_mpm/ct0.png')
# plt.clf()

# xpoints = np.array(steps)
# ypoints = np.array(ct1_vals)

# plt.plot(xpoints, ypoints)
# plt.title('Optimizing clusterPlasticThreshold  value')
# plt.xlabel('steps')
# plt.ylabel('clusterPlasticThreshold')
# plt.savefig('images_mpm/ct1.png')
# plt.clf()

# xpoints = np.array(steps)
# ypoints = np.array(ct2_vals)

# plt.plot(xpoints, ypoints)
# plt.title('Optimizing clusterPlasticCreep   value')
# plt.xlabel('steps')
# plt.ylabel('clusterPlasticCreep')
# plt.savefig('images_mpm/ct2.png')
# plt.clf()

# xpoints = np.array(steps)
# ypoints = np.array(loss_vals)

# plt.plot(xpoints, ypoints)
# plt.title('Loss while optimization')
# plt.xlabel('steps')
# plt.ylabel('Loss')
# plt.savefig('images_mpm/loss.png')