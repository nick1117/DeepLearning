# For interactive plotting:
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# utility
import os
import pickle
import zipfile
import yaml
from tqdm import tqdm
from typing import Union

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

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


@torch.enable_grad()
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str = Device,
    pbar: bool = False,
    ):
    if pbar:
        train_pbar = tqdm(
            train_loader,
            desc = 'training',
            unit = 'batch',
            dynamic_ncols = True,
            leave = False,
        )
    else:
        train_pbar = train_loader

    model.train().to(device)

    for x, y in train_pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if pbar:
            train_pbar.set_postfix_str(f'loss: {loss.item():.4f}')


@torch.inference_mode()
def eval_epoch(
    model: nn.Module,
    data_loader: DataLoader, # can be train_loader or val_loader or test_loader
    loss_fn: nn.Module,
    device: str = Device,
    pbar: bool = False,
    ):
    assert loss_fn.reduction in ['mean', 'sum'], 'Invalid reduction method!'
    if pbar:
        val_pbar = tqdm(
            data_loader,
            desc = 'testing',
            unit = 'batch',
            dynamic_ncols = True,
            leave = False,
            )
    else:
        val_pbar = data_loader

    model.eval().to(device)
    
    n = len(data_loader.dataset)
    Loss = 0.
    Accuracy = 0.
    for x, y in val_pbar:
        b = len(x)
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if loss_fn.reduction == 'mean':
            Loss += loss.item()*b
        elif loss_fn.reduction == 'sum':
            Loss += loss.item()

        Accuracy += (y_pred.argmax(dim=-1) == y).sum().item()

    return Loss/n, Accuracy/n


class Tracker:
    """
    Logs training and validation loss and plots them in real-time in a Jupyter notebook.
    """
    def __init__(
            self, 
            n_epochs: int,
            plot_freq: int = 0, # plot every plot_freq epochs. 0 for no plotting
            ):
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.epoch = 0
        self.n_epochs = n_epochs
        self.plot_freq = plot_freq
        if self.plot_freq > 0:
            self.plot_results()
        
        self.keys = ['train_losses', 'test_losses', 'train_accs', 'test_accs', 'epoch', 'n_epochs']

    def plot_results(self):
        self.fig, (self.loss_ax, self.acc_ax) = plt.subplots(1, 2, figsize=(16,6))

        xtickstep = max(1, self.n_epochs//10)
        xticks = list(range(0, self.n_epochs+1, xtickstep))
        if xticks[-1] != self.n_epochs:
            xticks.append(self.n_epochs)

        # Loss plot:
        self.train_curve, = self.loss_ax.plot(
            range(1, self.epoch+1), 
            self.train_losses,
            'o-', 
            label = 'train',
            )
        self.val_curve, = self.loss_ax.plot(
            range(1, self.epoch+1), 
            self.test_losses, 
            'o-',
            label = 'test'
            )
        self.loss_ax.set_xlim(0, self.n_epochs+1)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_xticks(xticks)
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Loss Learning Curve')
        self.loss_ax.legend(loc='upper right')
        self.loss_ax.grid(linestyle='--')
        self.loss_text = self.loss_ax.text(1.01, 1.0, '', transform=self.loss_ax.transAxes, va='top', ha='left')

        # Accuracy plot:
        self.train_acc_curve, = self.acc_ax.plot(
            range(1, self.epoch+1),
            self.train_accs, 
            'o-',
            label = 'train',
            )
        self.val_acc_curve, = self.acc_ax.plot(
            range(1, self.epoch+1), 
            self.test_accs, 
            'o-',
            label = 'test',
            )
        self.acc_ax.set_xlim(0, self.n_epochs+1)
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_xticks(xticks)
        self.acc_ax.set_ylabel('Accuracy')
        self.acc_ax.set_title('Accuracy Learning Curve')
        self.acc_ax.legend(loc='lower right')
        self.acc_ax.grid(linestyle='--')
        self.acc_text = self.acc_ax.text(1.01, 1.0, '', transform=self.acc_ax.transAxes, va='top', ha='left')

    def update(
            self, 
            train_loss: float, 
            test_loss: float, 
            train_acc: float, 
            test_acc: float,
            ):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        self.epoch += 1
        if self.plot_freq and self.epoch % self.plot_freq == 0:

            # loss plot:
            self.train_curve.set_data(range(1, self.epoch+1), self.train_losses)
            self.val_curve.set_data(range(1, self.epoch+1), self.test_losses)
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()
            self.loss_ax.set_ylim(bottom=0.0, top=None)
            self.loss_text.set_text(f'Epoch {self.epoch}\n' + 20*'-' + '\n' + f'Train Loss: {train_loss:.4f}\nTest Loss: {test_loss:.4f}')

            # accuracy plot:
            self.train_acc_curve.set_data(range(1, self.epoch+1), self.train_accs)
            self.val_acc_curve.set_data(range(1, self.epoch+1), self.test_accs)
            self.acc_ax.relim()
            self.acc_ax.autoscale_view()
            self.acc_ax.set_ylim(bottom=None, top=1.0)
            self.acc_text.set_text(f'Epoch {self.epoch}\n' + 20*'-' + '\n' + f'Train Acc: {train_acc:.4f}\nTest Acc: {test_acc:.4f}')

            plt.tight_layout()
            self.fig.canvas.draw()
            clear_output(wait=True)
            display(self.fig)

    def save_results(self, path: str):
        # saving losses and accuracies with pickle
        with open(path, 'wb') as file:
            pickle.dump({key: getattr(self, key) for key in self.keys}, file)

    def load_results(self, path: str):
        # loading losses and accuracies with pickle
        with open(path, 'rb') as file:
            results = pickle.load(file)
            for key, value in results.items():
                setattr(self, key, value)


def train(
    # Model and data
    save_path: str,
    model: nn.Module,
    train_data: Dataset,
    test_data: Dataset,

    # Loss and optimizer
    loss_fn: nn.Module,
    optim_name: str, # from optim
    optim_config: dict = dict(),
    lr_scheduler_name: Union[str, None] = None, # from lr_scheduler
    lr_scheduler_config: dict = dict(),

    # training settings:
    n_epochs: int = 10,
    batch_size: int = 32,
    device: str = Device,

    # progress bar and plotting:
    train_pbar: bool = False,
    val_pbar: bool = False,
    plot_freq: int = 1,
    save_freq: int = 1,
    ):

    if save_freq == 0:
        save_freq = n_epochs
        
    os.makedirs(f'{save_path}/checkpoints', exist_ok=True)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer: optim.Optimizer = optim.__getattribute__(optim_name)(model.parameters(), **optim_config)
    if lr_scheduler_name is not None:
        scheduler: lr_scheduler._LRScheduler = lr_scheduler.__getattribute__(lr_scheduler_name)(optimizer, **lr_scheduler_config)

    epoch_pbar = tqdm(
        range(1, n_epochs+1),
        desc = 'epochs',
        unit = 'epoch',
        dynamic_ncols = True,
        leave = True,
        )

    tracker = Tracker(n_epochs, plot_freq=plot_freq)

    for epoch in epoch_pbar:

        train_epoch(
            model = model,
            train_loader = train_loader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device,
            pbar = train_pbar,
            )

        train_loss, train_acc = eval_epoch(
            model = model,
            data_loader = train_loader,
            loss_fn = loss_fn,
            device = device,
            pbar = val_pbar,
            )

        test_loss, test_acc = eval_epoch(
            model = model,
            data_loader = test_loader,
            loss_fn = loss_fn,
            device = device,
            pbar = val_pbar,
            )

        if lr_scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(train_loss)
        elif lr_scheduler_name is not None:
            scheduler.step()

        tracker.update(train_loss, test_loss, train_acc, test_acc)

        if epoch % save_freq == 0:
            torch.save(model.state_dict(), f'{save_path}/checkpoints/epoch_{epoch}.pt')
            tracker.save_results(f'{save_path}/results.pkl')
