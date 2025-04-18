import os
import yaml
import zipfile

import numpy as np

from torch.utils.data import Dataset


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
