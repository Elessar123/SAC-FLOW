# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Implementation of Multi-layer Perceptron (MLP).

Residual model is taken from https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py
"""

import torch
from torch import nn
from collections import OrderedDict
import logging


activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
        "SiLU": nn.SiLU(),
    }
)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            # Add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][0]  # Linear layer is first in the last Sequential
            nn.init.constant_(final_linear.bias, out_bias_init)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi-layer perceptron network with residual connections for
    benchmarking the performance of different networks. The residual layers
    are based on the IBC paper implementation, which uses 2 residual layers
    with pre-activation with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        out_bias_init=None,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.bias, out_bias_init)
                    break

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input
    
class GRUNet(nn.Module):
    """
    GRU "no_tanh" 

    GRU Cell
    MLP
    """
    def __init__(
        self,
        dim_list,
        append_dim=0,
        hidden_mlp_dim=128,
        verbose=False,
    ):
        """
        GRU

        Args:
            dim_list (list):  `[input_dim, hidden_dim]`
                             - input_dim:  x 
                             - hidden_dim:  h 
            append_dim (int): 
            hidden_mlp_dim (int): MLP（）
            verbose (bool): 
        """
        super(GRUNet, self).__init__()

        # --- 1.  ---
        if len(dim_list) != 2:
            raise ValueError("GRU, `dim_list`  `[input_dim, hidden_dim]` ")
        
        input_dim = dim_list[0]
        hidden_dim = dim_list[1]
        self.hidden_dim = hidden_dim
        self.append_dim = append_dim

        #  [x, h_prev, append] 
        combined_input_dim = input_dim + hidden_dim + append_dim

        # --- 2.  () ---

        #  (Gate Network):  z
        #  hidden_dim
        self.gate_net = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_mlp_dim),
            nn.Mish(),
            nn.Linear(hidden_mlp_dim, hidden_dim)
        )

        #  (Candidate Network):  h_tilde
        self.candidate_net = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_mlp_dim),
            nn.Mish(),
            nn.Linear(hidden_mlp_dim, hidden_dim)
        )
        
        if verbose:
            print("--- GRUNet (Modular 'no_tanh' variant) ---")
            print(self)
            print("-----------------------------------------")


    def forward(self, x, h_prev, append=None):
        """
        GRU (no_tanh )

        Args:
            x (torch.Tensor):  `(batch, input_dim)`
            h_prev (torch.Tensor):  `(batch, hidden_dim)`
            append (torch.Tensor, optional):  `(batch, append_dim)`

        Returns:
            torch.Tensor:  h_next `(batch, hidden_dim)`
        """
        # --- 1.  ---
        # 
        if append is not None:
            if append.shape[1] != self.append_dim:
                raise ValueError(f" `append`  ({append.shape[1]})  `append_dim` ({self.append_dim}) ")
            net_input = torch.cat([x, h_prev, append], dim=-1)
        else:
            net_input = torch.cat([x, h_prev], dim=-1)

        # --- 2.  ---
        #  z_t = sigmoid(gate_net(x_t, h_{t-1}, ...))
        z = torch.sigmoid(self.gate_net(net_input))

        #  h_tilde_t = candidate_net(x_t, h_{t-1}, ...)
        #  tanh  "no_tanh" 
        h_tilde = self.candidate_net(net_input)
        
        # --- 3.  ---
        # GRU v = z * (h_tilde - h_prev)
        #  v = h_next - h_prev
        # h_next = h_prev + v = h_prev + z * (h_tilde - h_prev) = (1-z)*h_prev + z*h_tilde
        # h_next = (1 - z) * h_prev + z * h_tilde
        h_next = z * (h_tilde - h_prev)
        
        return h_next
  