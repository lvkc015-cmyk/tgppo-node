import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from torch.distributions import Categorical
import numpy as np
from NodeSelect.modules_node import BiMatchingNet, TreeGateBranchingNet

from BiGragh.model import GNNPolicy

class Critic(nn.Module):
    """Critic network for PPO that estimates state values."""

    def __init__(self, node_cand_dim, node_dim, mip_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cand_embedding = nn.Sequential(
            nn.LayerNorm(node_cand_dim),
            nn.Linear(node_cand_dim, hidden_dim),
            nn.GELU()
        )
        self.tree_embedding = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim + 15),
            nn.Linear(node_dim + mip_dim + 15, hidden_dim),
            nn.GELU()
        )

        # 3. 全局-局部 早期融合 (与 Actor 保持一致！)
        self.global_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
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
            encoder_kwargs["norm_first"] = True  # 推荐开启，对训练稳定性更好

        encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # self.tree_refinement = BiMatchingNet(hidden_dim)

        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # self.value_head = TreeGateBranchingNet(
        #     branch_size=hidden_dim,
        #     tree_state_size=hidden_dim,
        #     dim_reduce_factor=2,
        #     infimum=1,
        #     norm='layer',
        #     depth=2,
        #     hidden_size=hidden_dim,
        # )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # ====== 🚨 强化学习核心修复：稳住初始 Value ======
        # 让 Critic 在初始时输出的值接近 0
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0.0)

        self.gnn_policy =  GNNPolicy()


    def forward(self, cands_state_mat, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, padding_mask=None, mb_cons_batch=None, mb_var_batch=None):

        # 1. 获取图特征
        graph_embedding = self.gnn_policy.forward_graph(
                norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, constraint_batch=mb_cons_batch, 
                variable_batch=mb_var_batch
            )

        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)

        batch_size, num_candidates, _ = cands_state_mat.shape

        # 2. 基础特征提取
        cand_features = self.cand_embedding(cands_state_mat)  # [B, L, H]
        tree_state = torch.cat([node_state, mip_state, graph_embedding], dim=-1)
        tree_features = self.tree_embedding(tree_state)       # [B, H]

        # 3. 将全局特征分发给每个节点 (让 Critic 提前拥有全局视野)
        tree_features_expanded = tree_features.unsqueeze(1).expand(-1, num_candidates, -1)
        combined_features = torch.cat([cand_features, tree_features_expanded], dim=-1)
        var_features = self.global_embedding(combined_features) # [B, L, H]

        # 4. 防死区与 Padding Mask 生成
        if padding_mask is not None:
            src_key_padding_mask = padding_mask
            all_masked = src_key_padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                # 安全解包死区
                src_key_padding_mask = src_key_padding_mask.masked_fill(all_masked, False)
        else:
            src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

        # 5. Transformer 交互
        var_features = var_features.transpose(0, 1)  # [L, B, H]
        transformed_features = self.transformer(var_features, src_key_padding_mask=src_key_padding_mask)
        transformed_features = transformed_features.transpose(0, 1)  # [B, L, H]

        # --- 砍掉了 BiMatchingNet，直接进行池化 ---

        # 6. Mean Pooling (只对有效的候选节点求平均)
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, L, 1] 转为 float 用于计算
        masked_sum = (transformed_features * valid_mask).sum(dim=1) # [B, H]
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)          # 防止除以 0
        pooled_features = masked_sum / valid_count                  # [B, H] 代表整批节点的“综合潜力”

        # 7. 聚合最终 Value
        combined_final = torch.cat([pooled_features, tree_features], dim=-1)
        aggregated = self.aggregation(combined_final)
        value = self.value_head(aggregated)

        return value
    
    ##原来的版本
    # def forward(self, cands_state_mat, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, padding_mask=None,mb_cons_batch=None,mb_var_batch=None):

    #     graph_embedding = self.gnn_policy.forward_graph(
    #             norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,constraint_batch=mb_cons_batch, # 只有 learn 阶段有值，推理时为 None
    #             variable_batch=mb_var_batch
    #         )

    #     # 统一维度确保拼接成功
    #     if graph_embedding.dim() == 1:
    #         graph_embedding = graph_embedding.unsqueeze(0)


    #     batch_size, num_candidates, _ = cands_state_mat.shape

    #     cand_features = self.cand_embedding(cands_state_mat)
    #     cand_features = cand_features.transpose(0, 1)  # [L, B, H]

    #     if padding_mask is not None:
    #         src_key_padding_mask = padding_mask
    #         all_masked = src_key_padding_mask.all(dim=-1, keepdim=True)
    #         if all_masked.any():
    #             src_key_padding_mask = src_key_padding_mask.masked_fill(all_masked, False)
    #     else:
    #         src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

    #     transformed_features = self.transformer(cand_features, src_key_padding_mask=src_key_padding_mask)
    #     transformed_features = transformed_features.transpose(0, 1)  # [B, L, H]

    #     tree_state = torch.cat([node_state, mip_state,graph_embedding], dim=-1)
    #     tree_features = self.tree_embedding(tree_state)  # [B, H]

    #     refined_features = self.tree_refinement(tree_features, transformed_features, src_key_padding_mask)

    #     valid_mask = (~src_key_padding_mask).unsqueeze(-1)  # [B, L, 1]
    #     masked_sum = (refined_features * valid_mask).sum(dim=1)
    #     valid_count = valid_mask.sum(dim=1).clamp(min=1)
    #     pooled_features = masked_sum / valid_count

    #     combined = torch.cat([pooled_features, tree_features], dim=-1)
    #     aggregated = self.aggregation(combined)

    #     # value = self.value_head(aggregated, tree_features).unsqueeze(-1)  # [B, 1]
    #     value = self.value_head(aggregated)
    #     return value
