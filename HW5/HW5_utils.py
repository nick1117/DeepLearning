# For interactive plotting:
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# utility
import os
import yaml
import zipfile
import pickle
from tqdm import tqdm
from typing import Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as gDataLoader
import torch_geometric.nn as gnn


if torch.cuda.is_available():
    Device = 'cuda'
elif torch.backends.mps.is_available():
    Device = 'mps'
else:
    Device = 'cpu'


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
    
def zip_files(output_filename, *file_paths):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))


# helper function to inspect a tensor:
def print_tensor_info(
        name: str, 
        tensor, # torch.Tensor
        ):
    print(f'{name}')
    print(20*'-')
    if not isinstance(tensor, torch.Tensor):
        print(f'It is {type(tensor).__name__}!')
        print(20*'='+'\n')
        return
    # print name, shhape, dtype, device, require_grad
    print(f'shape: {tensor.shape}')
    print(f'dtype: {tensor.dtype}')
    print(f'device: {tensor.device}')
    print(f'requires_grad: {tensor.requires_grad}')
    print(20*'='+'\n')


#%% Functions to test graph_convolution:

def make_random_edge_index(num_nodes: int) -> np.ndarray:
    """
    creates a numpy array of shape (2, num_edges) where each column is an edge.
    Each node is guaranteed to have at least one incoming edge to avoid isolated nodes.
    No duplicate edges will be created.
    """
    edges = []
    for i in range(num_nodes):
        n_incoming_edges = np.random.randint(1, num_nodes)
        other_nodes = list(set(range(num_nodes))-{i})
        incoming_edges = np.random.choice(other_nodes, size=n_incoming_edges, replace=False)
        edges.extend([(j, i) for j in incoming_edges])
    return np.array(edges, dtype=np.int64).T


@torch.no_grad()
def graph_convolution_torch(
        x: torch.FloatTensor, # shape: (num_nodes, in_channels), dtype: torch.float32
        edge_index: torch.LongTensor, # shape: (2, num_edges), dtype: torch.int64
        edge_weight: torch.FloatTensor, # shape: (num_edges,), dtype: torch.float32
        theta: torch.FloatTensor, # shape: (in_channels, out_channels), dtype: torch.float32
        add_self_loops: bool = True,
        ) -> torch.FloatTensor: # shape: (num_nodes, out_channels), dtype: torch.float32
    
    gcn_layer = gnn.GCNConv(
        in_channels = x.shape[1],
        out_channels = theta.shape[1],
        add_self_loops = add_self_loops,
        bias = False,
        )
    gcn_layer.lin.weight.data = theta.T
    output = gcn_layer(x, edge_index, edge_weight)
    return output


def test_graph_convolution(
        graph_convolution: callable,
        num_tests: int = 10,
        show_failed_edge_index: bool = False,
        ):

    passed = True
    
    for i in range(1, num_tests+1):

        # Create a random graph and theta (weight matrix):
        num_nodes = np.random.randint(5, 11)
        in_channels = np.random.randint(1, 4)
        out_channels = np.random.randint(1, 6)
        add_self_loops = i % 2 == 0
            
        x = np.random.rand(num_nodes, in_channels).astype(np.float32)
        edge_index = make_random_edge_index(num_nodes)
        edge_weight = np.random.rand(edge_index.shape[1]).astype(np.float32)*2
        theta = np.random.rand(in_channels, out_channels).astype(np.float32)

        # Compare your output with the output of the torch_geometric GCNConv layer:
        your_output = graph_convolution(x, edge_index, edge_weight, theta, add_self_loops)
        assert your_output.shape == (num_nodes, out_channels), f"Test {i}: your output shape is wrong"

        torch_output = graph_convolution_torch(
            x = torch.tensor(x, dtype=torch.float32),
            edge_index = torch.tensor(edge_index, dtype=torch.int64),
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32),
            theta = torch.tensor(theta, dtype=torch.float32),
            add_self_loops = add_self_loops,
            )
        try:
            torch.testing.assert_close(
                torch.tensor(your_output, dtype=torch.float32),
                torch_output,
                atol = 1e-6, rtol = 1e-6,
                )
        except AssertionError as e:
            passed = False
            print(f"Test {i} failed:")
            print(e)
            if show_failed_edge_index:
                print(f"edge_index:\n{edge_index}")
            print()
            passed = False
        
    if passed:
        print("All tests passed!")


#%% Helper functions for training and evaluation:

@torch.enable_grad()
def train_epoch(
    model: nn.Module,
    graph_data_loader: gDataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = Device,
    ):

    model.train().to(device)

    for data in graph_data_loader:
        data.to(device)
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(y_pred, data.y)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def eval_epoch(
    model: nn.Module,
    graph_data_loader: gDataLoader,
    loss_fn: nn.Module,
    device: str = Device,
    ):
    # Calculate average loss
    assert loss_fn.reduction == 'mean'
    model.eval().to(device)

    total_loss = 0.
    n = len(graph_data_loader.dataset)
    for data in graph_data_loader:
        data.to(device)
        y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_loss += loss_fn(y_pred, data.y).item()*len(data.y)

    return total_loss/n


def train(
        model: nn.Module,
        train_dataset: Dataset,
        loss_fn: nn.Module = nn.MSELoss(),
        device: str = Device,
        plot_freq: int = 25,

        # train config:
        optimizer_name: str = 'Adam',
        optimizer_config: dict = dict(),
        lr_scheduler_name: Union[str, None] = None,
        lr_scheduler_config: dict = dict(),
        batch_size: int = 32,
        n_epochs: int = 100,
        ):
    
    train_loader = gDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer: optim.Optimizer = optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_config)
    if lr_scheduler_name is not None:
        scheduler: lr_scheduler.LRScheduler = lr_scheduler.__getattribute__(lr_scheduler_name)(optimizer, **lr_scheduler_config)

    tracker = Tracker(n_epochs, plot_freq=plot_freq)
    epoch_pbar = tqdm(range(n_epochs), desc="Epochs", unit="epoch", leave=True)

    for epoch in epoch_pbar:
        
        train_epoch(
            model = model,
            graph_data_loader = train_loader,
            optimizer = optimizer,
            loss_fn = loss_fn,
            device = device,
            )

        train_loss = eval_epoch(
            model = model,
            graph_data_loader = train_loader,
            loss_fn = loss_fn,
            device = device,
            )

        if lr_scheduler_name == 'ReduceLROnPlateau': 
            scheduler.step(train_loss)
        elif lr_scheduler_name is not None:
            scheduler.step()

        tracker.update(train_loss)
        epoch_pbar.set_postfix_str(f"Loss: {train_loss:.4f}")


class Tracker:
    """
    Logs training loss and plots them in real-time in a Jupyter notebook.
    """
    def __init__(
            self, 
            n_epochs: int,
            plot_freq: int = 0, # plot every how many epochs. Pass 0 for no plotting
            ):
        self.train_losses = []
        self.plot = plot_freq is not None
        self.epoch = 0
        self.n_epochs = n_epochs
        self.plot_freq = plot_freq
        if self.plot_freq:
            self.plot_results()
        
        self.keys = ['train_losses', 'epoch', 'n_epochs']

    def plot_results(self):
        self.fig, self.loss_ax = plt.subplots(1, 1, figsize=(16,6))

        # Loss plot:
        self.train_curve, = self.loss_ax.plot(
            range(1, self.epoch+1), 
            self.train_losses,
            )
        self.loss_ax.set_xlim(0, self.n_epochs+1)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Train Loss Learning Curve')
        self.loss_ax.grid(linestyle='--')
        self.loss_text = self.loss_ax.text(1.01, 1.0, '', transform=self.loss_ax.transAxes, va='top', ha='left')

    def update(
            self, 
            train_loss: float,
            ):
        self.train_losses.append(train_loss)
        self.epoch += 1
        if self.plot_freq and self.epoch % self.plot_freq == 0:

            # loss plot:
            self.train_curve.set_data(range(1, self.epoch+1), self.train_losses)
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()
            self.loss_ax.set_ylim(bottom=0.0, top=None)
            self.loss_text.set_text(f'Epoch {self.epoch}\n' + 20*'-' + '\n' + f'Train Loss: {train_loss:.4f}')

            plt.tight_layout()
            self.fig.canvas.draw()
            clear_output(wait=True)
            display(self.fig)

    def save_results(self, path: str):
        # saving losses and accuracies with pickle
        with open(path, 'wb') as file:
            pickle.dump(
                {key: getattr(self, key) for key in self.keys},
                file,
                )

    def load_results(self, path: str):
        # loading losses and accuracies with pickle
        with open(path, 'rb') as file:
            results: dict = pickle.load(file)
            for key, value in results.items():
                setattr(self, key, value)
