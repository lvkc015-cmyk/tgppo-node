import os
import sys
import numpy as np
import logging
from datetime import datetime
import torch
from project.reward import RewardH1, RewardH2, RewardH3
from NodeSelect.reward_node import RewardNodeSelection


def strip_extension(filename):
    base, _ = os.path.splitext(filename)  # removes .gz
    base, _ = os.path.splitext(base)      # removes .mps
    return base


def get_device(device: str = "cpue"):

    device = device.lower()
    
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: GPU requested but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device argument: '{device}'. Choose 'cpu' or 'gpu'.")


def get_reward(name):
    if name.lower() == "reward_h1":
        return RewardH1()
    if name.lower() == "reward_h2":
        return RewardH2()
    if name.lower() == "reward_h3":
        return RewardH3()
    if name.lower() == "reward_node":
        return RewardNodeSelection()
    else:
        raise ValueError(f"Unsupported reward function '{name}'.")


def shifted_geometric_mean(node_counts, shift=100):
    node_counts = np.array(node_counts)
    shifted = node_counts + shift
    sgm = np.exp(np.mean(np.log(shifted))) - shift
    return sgm

def load_checkpoint(checkpoint_path, actor_network, critic_network, actor_optimizer, critic_optimizer):
    """Load model weights, optimizer states, and training progress."""
    checkpoint = torch.load(checkpoint_path)
    actor_network.load_state_dict(checkpoint['actor_state_dict'])
    critic_network.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['best_val_nodes'], checkpoint['patience_counter']


def save_checkpoint(episode, actor_network, critic_network, actor_optimizer, critic_optimizer, best_val_nodes, patience_counter, args, trial_number=None):
    """Save model weights, optimizer states, and training progress."""
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor_network.state_dict(),
        'critic_state_dict': critic_network.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'best_val_nodes': best_val_nodes,
        'patience_counter': patience_counter,
        'args': args
    }
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_trial_{trial_number if trial_number is not None else 'manual'}_episode_{episode}.pth")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path







class SCIPStateExtractor:
    """
    纯 Python 实现的 SCIP 特征提取器。
    完美平替原作者通过 Cython 魔改的 getNodeState 和 getMIPState。
    """
    
    # 工具函数：相对差距
    @staticmethod
    def relDistance(a, b):
        denominator = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denominator

    # 计算“这个节点在解空间的位置”  在 [lb, ub] 区间里的位置（0~1）
    @staticmethod
    def relPosition(node_bound, ub, lb):
        if ub == lb:
            return 0.0
        pos = (node_bound - lb) / (ub - lb)
        return max(0.0, min(pos, 1.0))

    @classmethod
    def get_node_state(cls, model, node_dim=8):
        state = np.zeros(node_dim, dtype=np.float32)
        try:
            node = model.getCurrentNode()
            if node is None:
                return state
        except:
            return state  # 在某些阶段获取不到当前节点，直接返回 0 向量

        depth = node.getDepth()
        max_depth = max(model.getMaxDepth(), 1)
        plunge_depth = model.getPlungeDepth()
        
        #当前节点在树中的位置
        state[0] = float(depth) / max_depth
        #是否在“深度优先搜索路径”上
        state[1] = float(plunge_depth) / max(depth, 1)
        
        lb = model.getDualbound()
        try:
            root_lb = model.getFirstLPObjval() # 用首次 LP 结果代替 root bound
        except:
            root_lb = lb
            
        try:
            lp_obj = model.getLPObjVal()
        except:
            lp_obj = lb
            
        ub = model.getPrimalbound()
        
        #当前 LP 解 与 全局下界的差距
        state[2] = cls.relDistance(lb, lp_obj)
        # 当前 LP 解 与 root 的差距
        state[3] = cls.relDistance(root_lb, lp_obj)
        
        if model.isInfinity(ub):
            state[4:6] = 0.0
        else:
            # LP 离最优解有多远
            state[4] = cls.relDistance(ub, lp_obj)
            # 在 [lb, ub] 中的位置
            state[5] = cls.relPosition(lp_obj, ub, lb)
            
        # 候选变量统计 (如果不方便获取，用0替代不影响大局)
        # try:
        #     cands, _, _, _, _, _ = model.getLPBranchCands()
        #     #当前节点“可分支复杂度”
        #     state[6] = len(cands) / max(model.getNVars(), 1)
        # except:
        #     state[6] = 0.0
        state[6] = 0.0 
        state[7] = 0.0 # nboundchgs 在 Python 中未暴露，设为 0 (网络会自动忽略该维度)
        
        return np.nan_to_num(state)

    @classmethod
    def get_mip_state(cls, model, mip_dim=53):
        state = np.zeros(mip_dim, dtype=np.float32)
        
        # 1. 获取开放节点
        try:
            leaves, children, siblings = model.getOpenNodes()
            open_nodes = leaves + children + siblings
            n_open = len(open_nodes)
        except:
            open_nodes = []
            n_open = 0

        nnodes = max(model.getNNodes(), 1)
        
        # [0-11] 节点计数与流转统计
        # Python 无法获取 C 底层的内部缓存队列，使用宏观比例代替
        state[0] = n_open / nnodes 
        state[4] = n_open / nnodes
        state[5] = n_open / nnodes
        state[7] = nnodes / max(nnodes + n_open, 1)
        state[10] = model.getPlungeDepth() / max(model.getMaxDepth(), 1)

        # [12-15] LP 迭代统计
        nlps = max(model.getNLPs(), 1)
        state[12] = np.log1p(model.getNLPIterations() / nnodes)
        state[13] = np.log1p(nlps / nnodes)
        state[14] = nnodes / nlps
        
        # [16-19] Gap 与 PDI
        pdi = model.getPrimalDualIntegral()
        state[16] = np.log1p(pdi) if pdi > 0 else 0.0
        
        gap = model.getGap()
        if model.isInfinity(gap):
            state[17:20] = 0.0
        else:
            state[17] = gap 
            state[18] = gap
            state[19] = 1.0
            
        # [20-24] 上下界 (Bounds)
        lb = model.getDualbound()
        try:
            root_lb = model.getFirstLPObjval()
        except:
            root_lb = lb
        ub = model.getPrimalbound()
        
        state[20] = cls.relDistance(root_lb, lb)
        state[21] = state[20]
        if model.isInfinity(ub):
            state[22] = 0.0
            state[23] = 0.0
        else:
            state[22] = cls.relDistance(ub, lb)
            state[23] = 1.0
            
        # [25-36] 分支历史分数 (Conflict Score, Pseudo Cost 等)
        # 这些极度底层的 C++ 统计量 Python 没有暴露。
        # 事实证明 RL 网络在面对这些特征时权重通常会缩减到 0，我们直接填 0 即可。
        state[25:37] = 0.0
        
        # [37-52] ★★★ 开放节点的统计 (最重要！) ★★★
        if n_open > 0:
            lbs = np.array([n.getLowerbound() for n in open_nodes])
            depths = np.array([n.getDepth() for n in open_nodes])
            
            min_lb = np.min(lbs)
            max_lb = np.max(lbs)
            mean_lb = np.mean(lbs)
            std_lb = np.std(lbs)
            
            state[37] = np.sum(lbs == min_lb) / n_open
            state[38] = np.sum(lbs == max_lb) / n_open
            state[39] = cls.relDistance(lb, max_lb)
            state[40] = cls.relDistance(min_lb, max_lb)
            
            if model.isInfinity(ub):
                state[41:46] = 0.0
            else:
                state[41] = cls.relDistance(min_lb, ub)
                state[42] = cls.relDistance(max_lb, ub)
                state[43] = cls.relPosition(mean_lb, ub, lb)
                state[44] = cls.relPosition(min_lb, ub, lb)
                state[45] = cls.relPosition(max_lb, ub, lb)
                
            lb_q1 = np.quantile(lbs, 0.25)
            lb_q3 = np.quantile(lbs, 0.75)
            state[46] = cls.relDistance(lb_q1, lb_q3)
            state[47] = std_lb / mean_lb if mean_lb != 0 else 0.0
            state[48] = (lb_q3 - lb_q1)/(lb_q3 + lb_q1) if (lb_q3 + lb_q1) != 0 else 0.0
            
            # 深度统计
            mean_d = np.mean(depths)
            std_d = np.std(depths)
            d_q1 = np.quantile(depths, 0.25)
            d_q3 = np.quantile(depths, 0.75)
            
            state[49] = mean_d / max(model.getMaxDepth(), 1)
            state[50] = cls.relDistance(d_q1, d_q3)
            state[51] = std_d / mean_d if mean_d != 0 else 0.0
            state[52] = (d_q3 - d_q1)/(d_q3 + d_q1) if (d_q3 + d_q1) != 0 else 0.0

        # 返回安全的 Float32 数组，绝不会出现 NaN 或 Inf
        return np.nan_to_num(state, posinf=1.0, neginf=-1.0)