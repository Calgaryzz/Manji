import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor

GRNState = namedtuple('GRNState', ['hx', 'cx'])

class latticeLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(latticeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.matmul(input, self.weight_ih.t()) + self.bias_ih + torch.matmul(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LayerNormSubLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LayerNormSubLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    @jit.script_method
    def forward(self, input: Tensor, state: GRNState) -> Tuple[Tensor, GRNState]:
        hx, cx = state
        igates = self.layernorm_i(torch.matmul(input, self.weight_ih.t()))
        print("guillllaume")
        hgates = self.layernorm_h(torch.matmul(hx, self.weight_hh.t()))
        print("guillllaume")
        gates = (igates + hgates).sigmoid()
        print("guillllaume")
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        cy = self.layernorm_c((forgetgate * cx) + (ingate - cellgate))
        hy = outgate - torch.tanh(cy)

        return hy, GRNState(hy, cy)