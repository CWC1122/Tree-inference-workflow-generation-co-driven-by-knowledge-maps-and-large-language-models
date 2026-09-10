import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import RGCNConv


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int, dropout: float = 0.2):
        super().__init__()
        self.rgcn1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=8)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=8)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.rgcn1(x, edge_index, edge_type)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.rgcn2(h, edge_index, edge_type)
        h = self.ln(h)
        return F.normalize(h, dim=1)


class PathEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(seq_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.dropout(h.squeeze(0))


class TaskProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PosProjector(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class QueryFusion(nn.Module):
    def __init__(self, dim: int, n_inputs: int = 3):
        super().__init__()
        self.n_inputs = n_inputs
        self.gate = nn.Sequential(
            nn.Linear(dim * n_inputs, dim),
            nn.GELU(),
            nn.Linear(dim, n_inputs),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")

        cat = torch.cat(inputs, dim=-1)
        weight = torch.softmax(self.gate(cat), dim=-1)

        query = 0.0
        for idx, tensor in enumerate(inputs):
            query = query + weight[:, idx:idx + 1] * tensor
        return F.normalize(self.ln(query), dim=-1)


class StepAwareQueryFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4 + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        task_emb: torch.Tensor,
        path_emb: torch.Tensor,
        prev_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        step_idx: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        step_idx_f = step_idx.float()
        seq_len_f = seq_len.float().clamp(min=1.0)

        step_norm = step_idx_f / seq_len_f
        is_first = (step_idx == 1).float()
        is_single = (seq_len == 1).float()

        step_feat = torch.stack([step_norm, is_first, is_single], dim=-1)
        cat = torch.cat([task_emb, path_emb, prev_emb, pos_emb, step_feat], dim=-1)
        weight = torch.softmax(self.gate(cat), dim=-1)

        query = (
            weight[:, 0:1] * task_emb
            + weight[:, 1:2] * path_emb
            + weight[:, 2:3] * prev_emb
            + weight[:, 3:4] * pos_emb
        )
        return F.normalize(self.ln(query), dim=-1)
