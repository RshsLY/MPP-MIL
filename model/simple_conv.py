import random
import time

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SimapleAvgGraphConv(MessagePassing):
    def __init__(self, in_class, out_class, ):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_class, out_class)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


    def update(self, aggr_out):
        # return aggr_out
        return self.lin(aggr_out)
