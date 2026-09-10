from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import RGCNConv


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, dropout=0.2):
        super().__init__()
        self.rgcn1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=8)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=8)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type):
        h = self.rgcn1(x, edge_index, edge_type)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.rgcn2(h, edge_index, edge_type)
        h = self.ln(h)
        return F.normalize(h, dim=1)


class PathEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_tensor, lengths):
        packed = pack_padded_sequence(seq_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.dropout(h.squeeze(0))


class TaskProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PosProjector(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class QueryFusion(nn.Module):
    """
    n_inputs=3 -> task/path/prev
    n_inputs=4 -> task/path/prev/pos
    """
    def __init__(self, dim, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.gate = nn.Sequential(
            nn.Linear(dim * n_inputs, dim),
            nn.GELU(),
            nn.Linear(dim, n_inputs),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, *embs):
        if len(embs) != self.n_inputs:
            raise ValueError(f"QueryFusion expects {self.n_inputs} inputs, got {len(embs)}")
        cat = torch.cat(list(embs), dim=-1)
        w = torch.softmax(self.gate(cat), dim=-1)
        q = 0.0
        for i, e in enumerate(embs):
            q = q + w[:, i:i + 1] * e
        q = self.ln(q)
        return F.normalize(q, dim=-1)
