import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from NodeSelect.modules_node import BiMatchingNet, TreeGateBranchingNet

from BiGragh.model import GNNPolicy

class Actor(nn.Module):
    """Actor network for PPO that outputs action probabilities over candidate variables."""

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
        # self.global_embedding = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), # <--- 极其关键的防死区装置
            nn.GELU()                 # <--- 必须有
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,  #随机"丢弃"（暂时移除）网络中的一部分神经元,防止过拟合
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.tree_refinement = BiMatchingNet(hidden_dim)

        self.output_layer = TreeGateBranchingNet(
            branch_size=hidden_dim,
            tree_state_size=hidden_dim,
            dim_reduce_factor=2,
            infimum=1,
            norm='layer',
            depth=2,
            hidden_size=hidden_dim,
        )

        self.gnn_policy =  GNNPolicy()

    def forward(self, cands_state_mat, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,  padding_mask=None, mb_cons_batch=None, mb_var_batch=None):

        # 1. 图特征提取
        graph_embedding = self.gnn_policy.forward_graph(
                norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, constraint_batch=mb_cons_batch, 
                variable_batch=mb_var_batch
            )

        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)

        batch_size, num_candidates, _ = cands_state_mat.shape

        # 2. 基础特征提取
        cand_features = self.cand_embedding(cands_state_mat)

        tree_state = torch.cat([node_state, mip_state, graph_embedding], dim=-1)
        tree_features = self.tree_embedding(tree_state)

        # 3. 全局特征分发与初步融合
        tree_features_expanded = tree_features.unsqueeze(1).expand(-1, num_candidates, -1)
        combined_features = torch.cat([cand_features, tree_features_expanded], dim=-1)
        var_features = self.global_embedding(combined_features)

        var_features = var_features.transpose(0, 1)  # [L, B, H]

        # 🚨 修复 1：把防“死节点”的逻辑提前到这里！
        if padding_mask is not None: 
            # 在进入 Transformer 前就修复全 True 的 Mask，防止产生 NaN
            all_masked = padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                print(f"!!! 预防性修复了 {all_masked.sum().item()} 个全部被屏蔽的死节点 !!!")
                padding_mask = padding_mask.masked_fill(all_masked, False)
            src_key_padding_mask = padding_mask
        else: 
            src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

        # 4. Transformer 交互
        transformed_features = self.transformer(var_features, src_key_padding_mask=src_key_padding_mask)
        transformed_features = transformed_features.transpose(0, 1)  # [B, L, H]

        # 🚨 移除：refined_features = self.tree_refinement(...)
        # 直接把 Transformer 辛勤工作的结果交给输出层打分
        action_logits = self.output_layer(transformed_features, tree_features)  # [B, L]

        # ==========================================
        # ⬇️ 瘦身版 X光诊断探针代码 ⬇️
        # ==========================================
        # import random
        # if random.random() < 0.1:  # 抽样打印，避免刷屏
        #     with torch.no_grad():
        #         print("\n" + "="*50)
        #         print("🚨 X-RAY DIAGNOSTICS: FEATURE SHIFT 🚨")
        #         b_idx = 0  
                
        #         valid_mask = ~padding_mask[b_idx] if padding_mask is not None else torch.ones_like(action_logits[b_idx], dtype=torch.bool)
                
        #         feat_in = var_features.transpose(0, 1)[b_idx][valid_mask] 
        #         feat_trans = transformed_features[b_idx][valid_mask]
                
        #         print(f"【方差监控】(数值稳定性)")
        #         print(f" - Transformer 前方差: {feat_in.std().item():.4f}")
        #         print(f" - Transformer 后方差: {feat_trans.std().item():.4f}")
                
        #         print(f"\n【信息偏移监控】(网络是否真的在干活？)")
        #         cos_sim_trans = torch.nn.functional.cosine_similarity(feat_in, feat_trans, dim=-1).mean().item()
        #         l1_diff_trans = torch.abs(feat_in - feat_trans).mean().item()
        #         print(f" [Transformer 层] 余弦相似度: {cos_sim_trans:.4f} | 平均绝对位移: {l1_diff_trans:.4f}")
                
        #         valid_logits = action_logits[b_idx][valid_mask]
        #         std_logits = valid_logits.std().item() if valid_logits.numel() > 1 else 0.0
        #         print(f"\n【最终输出】Logits 差异度 (Std): {std_logits:.4f}")
        #         print(f" - Logits 截取: {valid_logits[:4].detach().cpu().numpy()}")
        #         print("="*50 + "\n")
        # ==========================================
        # ⬆️ 探针结束 ⬆️
        # ==========================================

        # 🚨 修复 2：使用 -1e8 代替 float('-inf') 保证数值安全
        if padding_mask is not None:
            # -1e8 在经过 softmax 后已经是 0 了，但对计算梯度的稳定性要好一万倍
            action_logits = action_logits.masked_fill(padding_mask, -1e8)

        # 最终输出概率分布给 Agent 进行 Categorical 采样
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


    #老版本          
    # def forward(self, cands_state_mat, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,  padding_mask=None, mb_cons_batch = None, mb_var_batch=None):

        
    #     graph_embedding = self.gnn_policy.forward_graph(
    #             norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, constraint_batch=mb_cons_batch, 
    #             variable_batch=mb_var_batch
    #         )

    #     # 统一维度确保拼接成功
    #     if graph_embedding.dim() == 1:
    #         graph_embedding = graph_embedding.unsqueeze(0)

    #     #num_candidates = L
    #     batch_size, num_candidates, _ = cands_state_mat.shape

    #     # 【修改】：使用改名后的 cand_embedding
    #     cand_features = self.cand_embedding(cands_state_mat)

    #     #[B, L, D_var] → [B, L, H]
    #     # var_features = self.var_embedding(cands_state_mat)
    #     #[B, D_node + D_mip] → [B, H]
    #     tree_state = torch.cat([node_state, mip_state, graph_embedding], dim=-1)
    #     tree_features = self.tree_embedding(tree_state)

    #     #把全局信息“分发”给每个变量 [B, H] → [B, 1, H] → [B, L, H]
    #     tree_features_expanded = tree_features.unsqueeze(1).expand(-1, num_candidates, -1)
    #     #变量自身信息、全局树 / 问题信息拼在一起：[B, L, H] + [B, L, H] → [B, L, 2H]
    #     combined_features = torch.cat([cand_features, tree_features_expanded], dim=-1)
    #     #再压回统一维度：[B, L, 2H] → [B, L, H]
    #     var_features = self.global_embedding(combined_features)

    #     #喂给 Transformer（候选变量之间互相“看”），Transformer 的输入格式要求
    #     #[B, L, H] → [L, B, H]
    #     var_features = var_features.transpose(0, 1)  # [L, B, H]

    #     if padding_mask is not None: #这是 padding，不要参与注意力计算
    #         src_key_padding_mask = padding_mask  # [B, L] True=pad
    #     else: #否则就创建一个全 False 的 mask
    #         src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

    #     #Transformer 编码，它会让每个候选变量根据其他候选变量的特征动态更新自己的表示
    #     transformed_features = self.transformer(var_features, src_key_padding_mask=src_key_padding_mask)
    #     #回到常见格式：[L, B, H] → [B, L, H]
    #     transformed_features = transformed_features.transpose(0, 1)  # [B, L, H]

    #     #树状态再一次精炼，“在看完所有变量后，结合当前树，再修正一次每个变量的重要性”
    #     refined_features = self.tree_refinement(tree_features, transformed_features, src_key_padding_mask)

    #     #输出 logits（未归一化分数）[B, L, H] + [B, H] → [B, L]
    #     action_logits = self.output_layer(refined_features, tree_features)  # [B, L]

    #     # print(f"Logits mean: {action_logits.mean().item():.4f}, std: {action_logits.std().item():.4f}")
    #     # print(f"Logits sample: {action_logits[0, :5].detach().cpu().numpy()}")

    #     # ==========================================
    #     # ⬇️ X光诊断探针代码 ⬇️
    #     # ==========================================
    #     import random
    #     if random.random() < 0.1:  # 抽样打印，避免刷屏
    #         with torch.no_grad():
    #             print("\n" + "="*50)
    #             print("🚨 X-RAY DIAGNOSTICS: FEATURE SHIFT 🚨")
    #             b_idx = 0  # 取 Batch 里的第 0 个图进行解剖
                
    #             # 获取真实变量的掩码（过滤掉 Padding）
    #             valid_mask = ~padding_mask[b_idx] if padding_mask is not None else torch.ones_like(action_logits[b_idx], dtype=torch.bool)
                
    #             # --- 1. 提取有效节点的特征 ---
    #             # 获取进 Transformer 前的特征 (这里借用转置前的 var_features)
    #             feat_in = var_features.transpose(0, 1)[b_idx][valid_mask] 
    #             # 获取出 Transformer 后的特征
    #             feat_trans = transformed_features[b_idx][valid_mask]
    #             # 获取出 Tree Refinement (BiMatchingNet) 后的特征
    #             feat_refine = refined_features[b_idx][valid_mask]
                
    #             # --- 2. 计算方差 (Scale/能量) ---
    #             print(f"【方差监控】(数值稳定性)")
    #             print(f" - Transformer 前方差: {feat_in.std().item():.4f}")
    #             print(f" - Transformer 后方差: {feat_trans.std().item():.4f}")
    #             print(f" - Refinement 后方差:  {feat_refine.std().item():.4f}")
                
    #             # --- 3. 计算信息偏移 (真正发生改变的证据) ---
    #             print(f"\n【信息偏移监控】(网络是否真的在干活？)")
                
    #             # Transformer 层的改变
    #             cos_sim_trans = torch.nn.functional.cosine_similarity(feat_in, feat_trans, dim=-1).mean().item()
    #             l1_diff_trans = torch.abs(feat_in - feat_trans).mean().item()
    #             print(f" [Transformer 层] 余弦相似度: {cos_sim_trans:.4f} | 平均绝对位移: {l1_diff_trans:.4f}")
                
    #             # Tree Refinement (BiMatchingNet) 层的改变
    #             cos_sim_refine = torch.nn.functional.cosine_similarity(feat_trans, feat_refine, dim=-1).mean().item()
    #             l1_diff_refine = torch.abs(feat_trans - feat_refine).mean().item()
    #             print(f" [Refinement 层] 余弦相似度: {cos_sim_refine:.4f} | 平均绝对位移: {l1_diff_refine:.4f}")
                
    #             # --- 4. 检查最终 Logits ---
    #             valid_logits = action_logits[b_idx][valid_mask]
    #             std_logits = valid_logits.std().item() if valid_logits.numel() > 1 else 0.0
    #             print(f"\n【最终输出】Logits 差异度 (Std): {std_logits:.4f}")
    #             print(f" - Logits 截取: {valid_logits[:4].detach().cpu().numpy()}")
    #             print("="*50 + "\n")
    #     # ==========================================
    #     # ⬆️ 探针结束 ⬆️
    #     # ==========================================

    #     #把 padding 位置彻底屏蔽，softmax 后概率 = 0
    #     if padding_mask is not None:

    #         ###########检查#################
    #         # 检查是否存在“死节点”（即没有可选变量的节点）
    #         all_masked = padding_mask.all(dim=-1,keepdim=True)
    #         if all_masked.any():
    #             print(f"!!! 发现 {all_masked.sum().item()} 个样本的所有分支变量都被屏蔽了 !!!")
    #             padding_mask = padding_mask.masked_fill(all_masked, False)
    #             # 如果全被 mask，Softmax 输出会全为 NaN 或 0


    #         action_logits = action_logits.masked_fill(padding_mask, float('-inf'))

    #     action_probs = F.softmax(action_logits, dim=-1)
    #     return action_probs

    # def get_action(self, cands_state_mat, node_state, mip_state, padding_mask=None):
    #     action_probs = self.forward(cands_state_mat, node_state, mip_state, padding_mask)
    #     dist = Categorical(action_probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     return action, log_prob, dist.entropy()