import os
import yaml
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from IPython.display import display, clear_output


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def zip_files(output_filename: str, file_paths: list):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))


class AirfoilDataset(Dataset):
	'''
	airfoil dataset: no need to modify
	'''
	def __init__(self):
		super().__init__()

		path: str = './data/airfoils.zip'
		if path.endswith('.zip'):
			if not os.path.exists(path):
				raise FileNotFoundError(f"Zip file {path} does not exist.")
			# if it has not been extracted, extract it
			if not os.path.exists(path[:-4]):
				with zipfile.ZipFile(path, 'r') as zip_ref:
					zip_ref.extractall()
			path = './airfoils'
		elif not os.path.exists(path):
			raise FileNotFoundError(f"Directory {path} does not exist.")

		self._X = []	# x coordinates of all airfoils (shared)
		self._Y = []	# y coordinates of all airfoils
		self.names = []	# name of all airfoils
		self.norm_coeff = 0	# normalization coeff to scale y to [-1, 1]
		airfoil_fn = [afn for afn in os.listdir(path) if afn.endswith('.dat')]

		# get x coordinates of all airfoils
		with open(os.path.join(path, airfoil_fn[0]), 'r', encoding="utf8", errors='ignore') as f:
			raw_data = f.readlines()
			for idx in range(len(raw_data)):
				raw_xy = raw_data[idx].split(' ')
				while "" in raw_xy:
					raw_xy.remove("")
				self._X.append(float(raw_xy[0]))
		self._X = np.array(self._X)

		# get y coordinates of each airfoils
		for idx, fn in enumerate(airfoil_fn):
			with open(os.path.join(path, fn), 'r', encoding="utf8", errors='ignore') as f:
				self.names.append(fn[:-10])
				raw_data = f.readlines()
				airfoil = np.empty(self._X.shape[0])
				for i in range(len(raw_data)):
					raw_xy = raw_data[i].split(' ')
					while "" in raw_xy:
						raw_xy.remove("")
					curr_y = float(raw_xy[1])
					airfoil[i] = curr_y
					self.norm_coeff = max(self.norm_coeff, np.abs(curr_y))
				self._Y.append(airfoil)

		self._Y = np.array([airfoil / self.norm_coeff for airfoil in self._Y], dtype=np.float32)

	def get_x(self):
		return self._X

	def get_y(self):
		return self._Y

	def __getitem__(self, idx):
		return self._Y[idx], self.names[idx]
		
	def __len__(self):
		return len(self._Y)


	def plot(self, idx=0):

		self.fig, self.ax = plt.subplots(figsize=(2, 2))
		y, name = self[idx]
		self.line, = self.ax.plot(self.get_x(), y)
		# equate the scale of x and y axis
		self.ax.set_xlim([-0.1, 1.1])
		self.ax.set_ylim([-0.6, 0.6])
		self.ax.set_aspect('equal', 'box')
		self.ax.set_title(name)
		self.ax.set_xlabel('x')
		self.ax.set_ylabel('y')
		self.ax.grid(linestyle='--', alpha=0.6)


def plot_airfoils(x: np.ndarray, ys: np.ndarray):
	"""
	x: np.ndarray of shape (n_points,)
	y: np.ndarray of shape (n_airfoils, n_points)

	Used to plot the generated airfoils.
	"""
	n_airfoils = ys.shape[0]
	n_rows = int(np.ceil(n_airfoils / 4))
	n_cols = min(n_airfoils, 4)
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), sharex=True, sharey=True)
	axs = axs.flatten()
	for i, ax in enumerate(axs):
		if i < n_airfoils:
			ax.plot(x, ys[i])
			ax.set_xlim([-0.1, 1.1])
			ax.set_ylim([-0.6, 0.6])
			ax.set_aspect('equal', 'box')
			ax.grid(linestyle='--', alpha=0.5)
		else:
			ax.axis('off')

	plt.tight_layout()
	plt.show()


class VAE_Tracker:
    """
    Logs and plots different loss terms of a VAE during training.
    """
    def __init__(
            self, 
            n_iters: int,
            plot_freq: int = 0, # plot every plot_freq iterations
            ):
        self.rec_losses = []
        self.prior_losses = []
        self.total_losses = []
        self.plot = plot_freq is not None
        self.iter = 0
        self.n_iters = n_iters
        self.plot_freq = plot_freq
        if self.plot_freq > 0:
            self.plot_results()

    def plot_results(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

        # Loss plot:
        self.rec_loss_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.rec_losses,
			label = 'Rec Loss',
            )
        self.prior_loss_curve, = self.ax2.plot(
            range(1, self.iter+1),
            self.prior_losses,
			label = 'Prior Loss',
            )
        self.total_loss_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.total_losses,
			label = 'Total Loss',
            )
        self.ax1.set_xlim(0, self.n_iters+1)
        self.ax1.set_ylim(0, 0.002)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Reconstruction and Total Loss Learning Curve')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

        self.ax2.set_xlim(0, self.n_iters+1)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Prior Loss Learning Curve')
        self.ax2.grid(linestyle='--')


    def update(
            self, 
            rec_loss: float,
            prior_loss: float,
            total_loss: float,
            ):
        self.rec_losses.append(rec_loss)
        self.prior_losses.append(prior_loss)
        self.total_losses.append(total_loss)
        self.iter += 1
		
        if self.plot_freq > 0 and self.iter % self.plot_freq == 0:

            # loss plot:
            self.rec_loss_curve.set_data(range(1, self.iter+1), self.rec_losses)
            self.prior_loss_curve.set_data(range(1, self.iter+1), self.prior_losses)
            self.total_loss_curve.set_data(range(1, self.iter+1), self.total_losses)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax1.set_ylim(bottom=0.0, top=None)
            self.ax2.relim()
            self.ax2.autoscale_view()
            plt.tight_layout()
            self.fig.canvas.draw()
            clear_output(wait=True)
            display(self.fig)
