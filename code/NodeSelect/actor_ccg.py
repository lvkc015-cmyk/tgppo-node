import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from BiGragh.model_ccg import GNNPolicyCCG
from NodeSelect.candidate_graph_context import CandidateConditionedGraphContext
from NodeSelect.modules_node import TreeGateBranchingNet


class ActorCCG(nn.Module):
    """Actor network with a candidate-conditioned graph branch."""

    def __init__(self, node_cand_dim, node_dim, mip_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cand_embedding = nn.Sequential(
            nn.LayerNorm(node_cand_dim),
            nn.Linear(node_cand_dim, hidden_dim),
            nn.GELU(),
        )
        self.tree_embedding = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim + 15),
            nn.Linear(node_dim + mip_dim + 15, hidden_dim),
            nn.GELU(),
        )
        self.global_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        encoder_kwargs = {
            "d_model": hidden_dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim * 4,
            "dropout": dropout,
            "activation": "gelu",
            "batch_first": False,
        }
        if "norm_first" in inspect.signature(nn.TransformerEncoderLayer.__init__).parameters:
            encoder_kwargs["norm_first"] = True

        encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = TreeGateBranchingNet(
            branch_size=hidden_dim,
            tree_state_size=hidden_dim,
            dim_reduce_factor=2,
            infimum=1,
            norm='layer',
            depth=2,
            hidden_size=hidden_dim,
        )

        self.gnn_policy = GNNPolicyCCG(token_dim=hidden_dim, output_dim=15)
        self.candidate_graph_context = CandidateConditionedGraphContext(hidden_dim, num_heads, dropout)

    def forward(self, cands_state_mat, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr,
                norm_var, norm_bounds, padding_mask=None, mb_cons_batch=None, mb_var_batch=None):
        graph_embedding, graph_tokens, graph_padding_mask = self.gnn_policy.forward_graph_context(
            norm_cons,
            norm_edge_idx,
            norm_edge_attr,
            norm_var,
            norm_bounds,
            constraint_batch=mb_cons_batch,
            variable_batch=mb_var_batch,
        )

        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)

        batch_size, num_candidates, _ = cands_state_mat.shape

        cand_features = self.cand_embedding(cands_state_mat)
        tree_state = torch.cat([node_state, mip_state, graph_embedding], dim=-1)
        tree_features = self.tree_embedding(tree_state)

        tree_features_expanded = tree_features.unsqueeze(1).expand(-1, num_candidates, -1)
        base_features = self.global_embedding(torch.cat([cand_features, tree_features_expanded], dim=-1))
        # Codex note: preserve the old global branch and add a candidate-conditioned graph branch in parallel.
        ccg_features = self.candidate_graph_context(
            cand_features, tree_features, graph_tokens, graph_padding_mask
        )
        var_features = base_features + ccg_features

        var_features = var_features.transpose(0, 1)
        if padding_mask is not None:
            all_masked = padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                padding_mask = padding_mask.masked_fill(all_masked, False)
            src_key_padding_mask = padding_mask
        else:
            src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

        transformed_features = self.transformer(var_features, src_key_padding_mask=src_key_padding_mask)
        transformed_features = transformed_features.transpose(0, 1)
        action_logits = self.output_layer(transformed_features, tree_features)

        if padding_mask is not None:
            action_logits = action_logits.masked_fill(padding_mask, -1e8)
        return F.softmax(action_logits, dim=-1)
