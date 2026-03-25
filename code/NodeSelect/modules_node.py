import torch
import torch.nn as nn
from torch.nn import functional as F
import functools


def get_norm_layer(norm_type='none'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer

# 原方案
#dim_reduce_factor：每一层的降维倍率； infimum：最低特征维度下限
# class TreeGateBranchingNet(nn.Module):
#     def __init__(self, branch_size, tree_state_size, dim_reduce_factor, infimum=32, norm='none', depth=2, hidden_size=128):
#         super().__init__()
#         norm_layer = get_norm_layer(norm)
#         self.branch_size = branch_size
#         #B&B树状态特征维度
#         self.tree_state_size = tree_state_size
#         #每层降维比例
#         self.dim_reduce_factor = dim_reduce_factor
#         #最终最小特征维度
#         self.infimum = infimum
#         #gate网络深度
#         self.depth = depth
#         #隐藏层维度
#         self.hidden_size = hidden_size

#         self.n_layers = 0 #记录需要多少层

#         #自动确定需要多少层才能从 branch_size → infimum， 每一层特征维度都缩小 dim_reduce_factor
#         unit_count = infimum
#         while unit_count < branch_size:
#             unit_count *= dim_reduce_factor
#             self.n_layers += 1
#         self.n_units_dict = {}

#         self.BranchingNet = nn.ModuleList()
#         input_dim = hidden_size
#         for i in range(self.n_layers):
#             output_dim = max(int(input_dim / dim_reduce_factor), 1)
#             self.n_units_dict[i] = input_dim
#             if i < self.n_layers - 1:
#                 layer = [nn.Linear(input_dim, output_dim), norm_layer(output_dim), nn.ReLU(True)]
#             else:
#                 layer = [nn.Linear(input_dim, output_dim)]
#             input_dim = output_dim
#             self.BranchingNet.append(nn.Sequential(*layer))

#         self.GatingNet = nn.Sequential()
#         self.n_attentional_units = sum(self.n_units_dict.values())
#         if depth == 1:
#             self.GatingNet.add_module('gate_linear', nn.Linear(tree_state_size, self.n_attentional_units))
#             self.GatingNet.add_module('gate_sig', nn.Sigmoid())
#         else:
#             self.GatingNet.add_module('gate_linear1', nn.Linear(tree_state_size, hidden_size))
#             self.GatingNet.add_module('gate_relu1', nn.ReLU(True))
#             for i in range(depth - 2):
#                 self.GatingNet.add_module(f'gate_linear{i+2}', nn.Linear(hidden_size, hidden_size))
#                 self.GatingNet.add_module(f'gate_relu{i+2}', nn.ReLU(True))
#             self.GatingNet.add_module('gate_linear_last', nn.Linear(hidden_size, self.n_attentional_units))
#             self.GatingNet.add_module('gate_sig', nn.Sigmoid())

#     # cands_state_mat候选变量特征，tree_state：B&B 树的全局状态向量
#     #用树状态 tree_state 生成一组 gate（注意力权重）
#     #在每一层对候选变量特征 cands_state_mat 做“按维度加权”，
#     #再逐层降维，最终输出每个候选变量的分数。
#     def forward(self, cands_state_mat, tree_state):
#         attn_weights = self.GatingNet(tree_state)
#         start_slice_idx = 0
#         for index, layer in enumerate(self.BranchingNet):
#             end_slice_idx = start_slice_idx + self.n_units_dict[index]
#             attn_slice = attn_weights[:, start_slice_idx:end_slice_idx]
#             if cands_state_mat.dim() == 3:
#                 cands_state_mat = cands_state_mat * attn_slice.unsqueeze(1)
#             else:
#                 cands_state_mat = cands_state_mat * attn_slice
#             cands_state_mat = layer(cands_state_mat)
#             start_slice_idx = end_slice_idx

#         ######## 找错 ############
#         if torch.abs(cands_state_mat).max() > 1e6:
#             print(f"Warning: cands_state_mat values are exploding: {cands_state_mat.max().item()}")


#         if cands_state_mat.size(-1) == 1:
#             return cands_state_mat.squeeze(-1)
#         else:
#             # if 3D -> mean over candidates; if 2D -> mean over features
#             dim = 1 if cands_state_mat.dim() == 3 else -1
#             return cands_state_mat.mean(dim=dim)

# 方案B
class TreeGateBranchingNet(nn.Module):
    def __init__(self, branch_size, tree_state_size, dim_reduce_factor=2, infimum=8, norm='layer', depth=2, hidden_size=128):
        super().__init__()
        # 修复版：精简网络深度，避免小维度 LayerNorm 和指数级梯度消失
        
        # 1. 变量特征提取层 (保持在安全的大维度 hidden_size)
        self.feature_net = nn.Sequential(
            nn.Linear(branch_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2. Tree Gate 生成器 (生成一层的全局注意力权重)
        self.gate_net = nn.Sequential(
            nn.Linear(tree_state_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid() # 限制在 0-1 之间作为门控开关
        )
        
        # 3. 最终打分层 (平滑降维输出标量分数，不加 LayerNorm)
        self.scoring_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, cands_state_mat, tree_state):
        # cands_state_mat: [B, L, H]
        # 1. 提炼变量特征
        cands_feat = self.feature_net(cands_state_mat)
        
        # 2. 生成 Tree Gate: [B, H] -> [B, 1, H]
        gate = self.gate_net(tree_state).unsqueeze(1)
        
        # 3. Tree Gating 核心机制：用当前的树状态去“过滤/激活”候选变量特征
        gated_cands = (cands_feat * gate) +  cands_feat
        # gated_cands = (cands_feat * gate) 
        
        # 4. 独立打分: [B, L, 1] -> [B, L]
        scores = self.scoring_layer(gated_cands).squeeze(-1)
        
        return scores


#双向交叉注意力
#用树状态去“关注”哪些变量重要，同时也让变量去“关注”当前树状态，然后把两者融合成新的变量表示。
class BiMatchingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_feat, cand_feat, padding_mask):
        """
        tree_feat: [B, E]
        var_feat:  [B, L, E]
        padding_mask: [B, L] True for pads; may be None
        """
        B, L, E = cand_feat.shape
        # 不同样本的候选变量数 L 不一样,为了 batch 计算，短的要补 0,padding 的位置不能参与注意力计算
        #True:这个位置是 padding（无效）
        if padding_mask is None:
            padding_mask = torch.zeros((B, L), dtype=torch.bool, device=cand_feat.device)

        #给 tree_feat 在 第 1 个维度 插入一个长度为 1 的维度
        tree_feat = tree_feat.unsqueeze(1)  # [B,1,E]
        
        #var_feat.transpose(1, 2)：var_feat: [B, L, E] → [B, E, L]
        #[B,1,E] × [B,E,L] → [B,1,L]
        #.squeeze(1)：[B,1,L] → [B,L]
        #当前树状态下，每个变量和树的“相似度 / 相关度”
        G_tc = torch.bmm(self.linear1_1(tree_feat), cand_feat.transpose(1, 2)).squeeze(1)  # [B, L]
        #padding 的位置填成 -∞，后面 softmax 后 = 0，padding 变量不会被关注
        G_tc = G_tc.masked_fill(padding_mask, float('-inf'))
        #得到一个 概率分布，树状态认为“哪个变量更重要”
        G_tc = F.softmax(G_tc, dim=1).unsqueeze(1)  # [B,1,L]

        #这次是反过来：每个变量怎么看当前的树状态？
        G_ct = torch.bmm(self.linear1_2(cand_feat), tree_feat.transpose(1, 2)).squeeze(2)  # [B,L]
        G_ct = G_ct.masked_fill(padding_mask, float('-inf'))
        G_ct = F.softmax(G_ct, dim=1).unsqueeze(2)  # [B,L,1]

        #用注意力加权平均所有变量，“在当前树状态下，最重要的变量长什么样”
        E_t = torch.bmm(G_tc, cand_feat)              # [B,1,E]
        #每个变量得到一个“定制版”的树特征
        E_c = torch.bmm(G_ct, tree_feat)             # [B,L,1] x [B,1,E] -> [B,L,E]

        #非线性映射
        S_tc = F.relu(self.linear2_1(E_t))           # [B,1,E]
        S_ct = F.relu(self.linear2_2(E_c))           # [B,L,E]

        #计算门控权重：对 每个变量、每个维度 决定：更信树视角还是变量视角
        attn_weight = torch.sigmoid(self.linear3_1(S_tc) + self.linear3_2(S_ct))  # [B,L,E]
        #门控融合：融合后的、考虑树状态的变量特征
        M_tc = attn_weight * S_tc + (1 - attn_weight) * S_ct

        # ==========================================
        # 🚨 终极修复：把变量自己的特征加回来！(残差连接)
        # 这样网络既拥有了宏观的树视角，又保留了每个变量自己的独立特性
        # ==========================================
        out_feat = cand_feat + M_tc
        # out_feat = M_tc
        return out_feat