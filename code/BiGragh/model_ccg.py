import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GraphConv
from torch_geometric.utils import to_dense_batch


class GNNPolicyCCG(nn.Module):
    """Graph encoder that preserves the old global graph embedding and adds graph tokens for CCG."""

    def __init__(self, token_dim=128, output_dim=15):
        super().__init__()

        self.emb_size = emb_size = 32

        hidden_dim1 = 8
        hidden_dim2 = 4
        hidden_dim3 = 4

        cons_nfeats = 6
        edge_nfeats = 1
        var_nfeats = 10

        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_nfeats, 1),
        )
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.ReLU(),
        )
        self.bounds_embedding = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 2),
            nn.ReLU(),
        )

        self.convs_v2c = nn.ModuleList([
            GraphConv((emb_size, emb_size), hidden_dim1),
            GraphConv((hidden_dim1, hidden_dim1), hidden_dim2),
            GraphConv((hidden_dim2, hidden_dim2), hidden_dim3),
        ])
        self.convs_c2v = nn.ModuleList([
            GraphConv((emb_size, emb_size), hidden_dim1),
            GraphConv((hidden_dim1, hidden_dim1), hidden_dim2),
            GraphConv((hidden_dim2, hidden_dim2), hidden_dim3),
        ])

        combined_input_dim = hidden_dim3 + hidden_dim3 + 2
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        # Codex note: graph tokens are projected to the policy hidden size so each candidate can read
        # the shared MILP graph through cross-attention without changing the rest of the architecture.
        self.var_token_projection = nn.Sequential(
            nn.Linear(hidden_dim3, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.cons_token_projection = nn.Sequential(
            nn.Linear(hidden_dim3, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.bounds_token_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

    @staticmethod
    def _ensure_batch(batch_indices, num_items, device):
        if batch_indices is None:
            return torch.zeros(num_items, dtype=torch.long, device=device)
        return batch_indices.to(device)

    def forward_graph_context(self, constraint_features, edge_indices, edge_features,
                              variable_features, bbounds, constraint_batch=None, variable_batch=None):
        variable_features = variable_features.contiguous()
        constraint_features = constraint_features.contiguous()
        edge_features = edge_features.contiguous()
        edge_indices = edge_indices.contiguous()
        bbounds = bbounds.contiguous()

        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features).view(-1)
        bounds_summary = self.bounds_embedding(bbounds)

        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        for conv_v2c, conv_c2v in zip(self.convs_v2c, self.convs_c2v):
            constraint_features_next = F.gelu(
                conv_v2c((variable_features, constraint_features), edge_indices, edge_weight=edge_features)
            )
            variable_features = F.gelu(
                conv_c2v((constraint_features, variable_features), edge_indices_reversed, edge_weight=edge_features)
            )
            constraint_features = constraint_features_next

        if bounds_summary.dim() == 1:
            bounds_summary = bounds_summary.unsqueeze(0)

        var_batch = self._ensure_batch(variable_batch, variable_features.size(0), variable_features.device)
        cons_batch = self._ensure_batch(constraint_batch, constraint_features.size(0), constraint_features.device)

        variable_avg = torch_geometric.nn.global_mean_pool(variable_features, var_batch)
        constraint_avg = torch_geometric.nn.global_mean_pool(constraint_features, cons_batch)
        global_embedding = self.feature_projection(torch.cat((variable_avg, constraint_avg, bounds_summary), dim=1))

        var_tokens, var_mask = to_dense_batch(self.var_token_projection(variable_features), var_batch)
        cons_tokens, cons_mask = to_dense_batch(self.cons_token_projection(constraint_features), cons_batch)

        bounds_tokens = self.bounds_token_projection(bounds_summary).unsqueeze(1)
        bounds_mask = torch.ones(
            (bounds_summary.size(0), 1), dtype=torch.bool, device=bounds_summary.device
        )

        graph_tokens = torch.cat((var_tokens, cons_tokens, bounds_tokens), dim=1)
        graph_valid_mask = torch.cat((var_mask, cons_mask, bounds_mask), dim=1)
        graph_padding_mask = ~graph_valid_mask
        return global_embedding, graph_tokens, graph_padding_mask

    def forward_graph(self, constraint_features, edge_indices, edge_features,
                      variable_features, bbounds, constraint_batch=None, variable_batch=None):
        global_embedding, _, _ = self.forward_graph_context(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            bbounds,
            constraint_batch=constraint_batch,
            variable_batch=variable_batch,
        )
        return global_embedding
