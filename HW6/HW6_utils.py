import os
import yaml
import zipfile

import numpy as np

import torch
from torch import nn


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def zip_files(output_filename, file_paths):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))


# %% Testing lstm_cell

def make_random_params(
    input_size: int, 
    hidden_size: int,
    ):
    params = dict()
    for a in 'ifgo':
        params[f'weight_i{a}'] = np.random.randn(input_size, hidden_size).astype(np.float32)
        params[f'bias_i{a}'] = np.random.randn(hidden_size).astype(np.float32)
        params[f'weight_h{a}'] = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        params[f'bias_h{a}'] = np.random.randn(hidden_size).astype(np.float32)

    return params


@torch.no_grad()
def lstm_cell_torch(
    x: np.ndarray, # (batch_size, input_size)
    h_prev: np.ndarray, # (batch_size, hidden_size)
    c_prev: np.ndarray, # (batch_size, hidden_size)
    params: dict,
    ):

    input_size = x.shape[-1]
    hidden_size = h_prev.shape[-1]

    cell = nn.LSTMCell(
        input_size = input_size,
        hidden_size = hidden_size,
        )
    
    cell.weight_ih.data = torch.cat([
        torch.tensor(params[f'weight_i{a}']).T for a in 'ifgo'
        ])

    cell.bias_ih.data = torch.cat([
        torch.tensor(params[f'bias_i{a}']) for a in 'ifgo'
        ])

    cell.weight_hh.data = torch.cat([
        torch.tensor(params[f'weight_h{a}']).T for a in 'ifgo'
        ])

    cell.bias_hh.data = torch.cat([
        torch.tensor(params[f'bias_h{a}']) for a in 'ifgo'
        ])

    return cell(
        torch.tensor(x, dtype=torch.float32),
        (torch.tensor(h_prev, dtype=torch.float32), torch.tensor(c_prev, dtype=torch.float32)),
        )


@ torch.no_grad()
def test_lstm_cell(
        lstm_cell: callable,
        num_tests: int = 10
        ):
    
    passed = True
    output = ''

    for i in range(1, num_tests+1):

        batch_size = np.random.randint(1, 10)
        input_size = np.random.randint(1, 10)
        hidden_size = np.random.randint(1, 10)

        x = np.random.randn(batch_size, input_size).astype(np.float32)
        h_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)
        c_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

        params = make_random_params(input_size, hidden_size)

        # your output
        h, c = lstm_cell(x, h_prev, c_prev, params)

        # The torch module output
        h_torch, c_torch = lstm_cell_torch(x, h_prev, c_prev, params)

        try:
            torch.testing.assert_close(torch.tensor(h), h_torch, rtol=1e-5, atol=1e-5)
        except AssertionError as e:
            passed = False
            output += f'wrong h for test {i}: {e}\n' + 50 * '-' + '\n'
        
        try:
            torch.testing.assert_close(torch.tensor(c), c_torch, rtol=1e-5, atol=1e-5)
        except AssertionError as e:
            passed = False
            output += f'wrong c for test {i}: {e}\n' + 50 * '-' + '\n'

    if passed:
        print('All tests passed!')
    else:
        print(output)
