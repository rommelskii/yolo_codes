# custom_eaf_modules.py
import torch
import torch.nn as nn

class EinsteinActivationFunction(nn.Module):
    def __init__(self, n_val=64.0, r_val=0.0):
        super().__init__()
        self.n_val = torch.tensor(n_val, dtype=torch.float32)
        if r_val == -1.0:
            raise ValueError("Parameter 'r_val' in EinsteinActivationFunction cannot be -1.0.")
        self.r_val = torch.tensor(r_val, dtype=torch.float32)
        self.factor = (1.0 - self.r_val) / (1.0 + self.r_val)
        self.register_buffer('const_n_val', self.n_val)
        self.register_buffer('const_factor', self.factor)

    def forward(self, x):
        x_div_n = x / self.const_n_val
        tan_x_div_n = torch.tan(x_div_n)
        if torch.isnan(tan_x_div_n).any():
            tan_x_div_n = torch.nan_to_num(tan_x_div_n, nan=0.0, posinf=1e8, neginf=-1e8)
        elif torch.isinf(tan_x_div_n).any():
            tan_x_div_n = torch.clamp(tan_x_div_n, min=-1e8, max=1e8)
        return self.const_n_val * torch.tanh(self.const_factor * tan_x_div_n)

    def __repr__(self):
        return f"EinsteinActivationFunction(n={self.n_val.item()}, r={self.r_val.item()})"

