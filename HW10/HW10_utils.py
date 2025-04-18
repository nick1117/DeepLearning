import os
import yaml
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


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


class Tracker:
    """
    Logs training loss and plots them in real-time in a Jupyter notebook.
    """
    def __init__(
            self, 
            n_epochs: int,
            plot_freq: Union[int, None] = None, # plot every plot_freq epochs
            ):
        self.losses = []
        self.losses_AR = []
        self.plot = plot_freq is not None
        self.epoch = 0
        self.n_epochs = n_epochs
        if self.plot:
            self.plot_freq = plot_freq
            self.plot_results()
        
        self.keys = ['losses', 'losses_AR', 'epoch', 'n_epochs']

    def plot_results(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 4))

        # Loss plot:
        self.loss_curve, = self.ax.plot(
            range(1, self.epoch+1), 
            self.losses,
            '-o',
            label = 'loss'
            )
        
        self.loss_curve_ar, = self.ax.plot(
            range(1, self.epoch+1),
            self.losses_AR,
            '-o',
            label = 'AR loss',
            )
        
        self.ax.set_xlim(0, self.n_epochs+1)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Evaluation Loss')
        self.ax.set_title('Learning Curve')
        self.ax.grid(linestyle='--')
        self.ax.set_xticks(range(1, self.n_epochs+1))
        self.ax.legend()

        self.text = self.ax.text(1.01, 1.0, '', transform=self.ax.transAxes, va='top', ha='left')


    def update(
            self, 
            train_loss: float,
            eval_loss: float,

            ):
        self.losses.append(train_loss)
        self.losses_AR.append(eval_loss)
        self.epoch += 1
        if self.plot and self.epoch % self.plot_freq == 0:

            # loss plot:
            self.loss_curve.set_data(range(1, self.epoch+1), self.losses)
            self.loss_curve_ar.set_data(range(1, self.epoch+1), self.losses_AR)
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.set_ylim(bottom=0.0, top=None)

            self.text.set_text(f'epoch {self.epoch}\n' + 20*'-' + '\n' + f'   loss: {train_loss:.4f}\nAR loss: {eval_loss:.4f}')

            plt.tight_layout()
            self.fig.canvas.draw()
            clear_output(wait=True)
            display(self.fig)


# %% autograder

def test_sdpa(scaled_dot_product_attention: callable, n_tests = 10):

    feedback = ""

    for i in range(1, n_tests+1):

        N, H, S, E, Ev = np.random.randint(1, 10, size=5)
        L = np.random.randint(S, 2*S) # L >= S so causal does not break

        feedback += f"Test {i}: N={N}, H={H}, S={S}, L={L}, E={E}, Ev={Ev}"

        q = np.random.randn(N, H, L, E).astype(np.float32)
        k = np.random.randn(N, H, S, E).astype(np.float32)
        v = np.random.randn(N, H, S, Ev).astype(np.float32)

        attn_mask = None
        is_causal = False

        passed = True

        if i in [5, 6, 7]:
            feedback += ", with mask"
            attn_mask = np.random.randint(0, 2, size=(L, S)) > 0
            attn_mask[np.arange(L), np.random.randint(0, S, size=L)] = True

        elif i in [8, 9, 10]:
            feedback += ", causal"
            is_causal = True

        feedback += " --> "

        out = scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            attn_mask = attn_mask,
            is_causal = is_causal,
        )

        torch_out = F.scaled_dot_product_attention(
            query = torch.as_tensor(q, dtype=torch.float32),
            key = torch.as_tensor(k, dtype=torch.float32),
            value = torch.as_tensor(v, dtype=torch.float32),
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool) if attn_mask is not None else None,
            is_causal = is_causal,
        )

        try:
            torch.testing.assert_close(
                torch.as_tensor(out, dtype=torch.float32), 
                torch_out, 
                atol=1e-3, rtol=1e-3,
                )
            feedback += "PASSED\n"
        except AssertionError as e:
            feedback += f"FAILED\n{e}\n"
            passed = False
        
        feedback += 70*'-' + '\n'

    if not passed:
        feedback += "Some tests FAILED!"
    else:
        feedback += "All tests PASSED!"

    print(feedback)


def test_mha(multi_head_attention: callable, n_tests = 10):

    feedback = ""

    for i in range(1, n_tests+1):

        N, S, kdim, vdim = np.random.randint(1, 10, size=4)
        L = np.random.randint(S, 2*S)
        E = np.random.choice([8, 16, 32])
        H = np.random.choice([1, 2, 4, 8])

        Wq = np.random.randn(E, E).astype(np.float32)
        Wk = np.random.randn(kdim, E).astype(np.float32)
        Wv = np.random.randn(vdim, E).astype(np.float32)
        Wo = np.random.randn(E, E).astype(np.float32)

        feedback += f"Test {i}: N={N}, S={S}, L={L}, E={E}, kdim={kdim}, vdim={vdim}, H={H}"

        query_target = np.random.randn(N, L, E).astype(np.float32)
        key_source = np.random.randn(N, S, kdim).astype(np.float32)
        value_source = np.random.randn(N, S, vdim).astype(np.float32)

        attn_mask = None
        passed = True

        if i in [6, 7, 8 ,9, 10]:
            feedback += ", with mask"
            attn_mask = np.random.randint(0, 2, size=(L, S)) < 1
            attn_mask[np.arange(L), np.random.randint(0, S, size=L)] = False

        feedback += " --> "

        out = multi_head_attention(
            query_target = query_target,
            key_source = key_source,
            value_source = value_source,
            num_heads = H,
            params = dict(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo),
            attn_mask = attn_mask,
            )
        
        mha_module = nn.MultiheadAttention(
            embed_dim = E,
            num_heads = H,
            kdim = kdim,
            vdim = vdim,
            batch_first = True,
            bias = False,
            )
        
        mha_module.q_proj_weight.data = torch.as_tensor(Wq.T, dtype=torch.float32)
        mha_module.k_proj_weight.data = torch.as_tensor(Wk.T, dtype=torch.float32)
        mha_module.v_proj_weight.data = torch.as_tensor(Wv.T, dtype=torch.float32)
        mha_module.out_proj.weight.data = torch.as_tensor(Wo.T, dtype=torch.float32)

        with torch.inference_mode():
            torch_out, _ = mha_module(
                query = torch.as_tensor(query_target, dtype=torch.float32),
                key = torch.as_tensor(key_source, dtype=torch.float32),
                value = torch.as_tensor(value_source, dtype=torch.float32),
                attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool) if attn_mask is not None else None,
                need_weights = False,
                )

        try:
            torch.testing.assert_close(
                torch.as_tensor(out, dtype=torch.float32), 
                torch_out, 
                atol=1e-3, rtol=1e-3,
                )
            feedback += "PASSED\n"
        except AssertionError as e:
            feedback += f"FAILED\n{e}\n"
            passed = False
        feedback += 70*'-' + '\n'

    if not passed:
        feedback += "Some tests FAILED!"
    else:
        feedback += "All tests PASSED!"

    print(feedback)
