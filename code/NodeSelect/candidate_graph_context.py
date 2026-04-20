import torch
import torch.nn as nn


class CandidateConditionedGraphContext(nn.Module):
    """Candidate-specific graph reader over a shared MILP graph token set."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cand_features, tree_features, graph_tokens, graph_padding_mask=None):
        tree_features_expanded = tree_features.unsqueeze(1).expand(-1, cand_features.size(1), -1)
        # Codex note: each candidate builds its own query from local candidate features and shared tree context.
        query = self.query_projection(torch.cat((cand_features, tree_features_expanded), dim=-1))
        attended_ctx, _ = self.cross_attention(
            query=query,
            key=graph_tokens,
            value=graph_tokens,
            key_padding_mask=graph_padding_mask,
            need_weights=False,
        )
        fused = self.context_fusion(torch.cat((cand_features, tree_features_expanded, attended_ctx), dim=-1))
        return cand_features + fused
