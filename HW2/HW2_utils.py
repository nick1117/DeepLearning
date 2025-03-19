import os
import zipfile
import yaml
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


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


class Learning_Curve_Tracker:
    """
    Logs training and validation loss and plots them in real-time in a Jupyter notebook.
    """
    def __init__(
            self, 
            n_epochs: int,
            plot_freq = None, # update plot once every plot_freq epochs. pass None if you don't want to plot
            # This depedns on how long each epoch takes to run
            # If each epoch takes a small amount of time to train, You may choose to plot less frequently
            # so the frame per second of the plot is not too high.
            ):
        self.train_losses = []
        self.val_losses = []
        self.plot = plot_freq is not None
        self.epoch = 0
        self.n_epochs = n_epochs
        if self.plot:
            self.plot_freq = plot_freq
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.train_curve, = self.ax.plot([], [], label='train loss')
            self.val_curve, = self.ax.plot([], [], label='val loss')
            self.ax.set_xlim(0, self.n_epochs)
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Learning Curve')
            self.ax.legend(loc=(1.01, 0.0))
            self.ax.grid(linestyle='--')
            self.text = self.ax.text(1.01, 1.0, '', transform=self.ax.transAxes, va='top', ha='left')

    def update(self, train_loss, val_loss):
        # updates the first plot
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch += 1
        if self.plot and self.epoch % self.plot_freq == 0:
            self.train_curve.set_data(range(1, self.epoch+1), self.train_losses)
            self.val_curve.set_data(range(1, self.epoch+1), self.val_losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.set_ylim(bottom=0.0, top=None)
            text =  f'Epoch {self.epoch}\n' + 25*'-' + '\n'
            text += f'Train Loss: {self.train_losses[-1]:.4f}\n'
            text += f'   Val Loss: {self.val_losses[-1]:.4f}\n'
            self.text.set_text(text)
            clear_output(wait=True)
            display(self.fig)
            self.fig.canvas.draw()

    def get_losses(self):
        return self.train_losses, self.val_losses
    
    