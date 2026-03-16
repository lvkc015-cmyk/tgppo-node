import torch as T
import logging
from typing import List, Tuple, Iterator, Optional, Dict, Any
import numpy as np

from torch_geometric.data import Data, Batch

class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            # 第一行（变量）增加变量节点的偏移
            # 第二行（约束）增加约束节点的偏移
            return T.tensor([[self.x_s.size(0)], [self.x_c.size(0)]], device=value.device)
        return super().__inc__(key, value, *args, **kwargs)


class Memory:
    """Tree-Gate Policy Gradient Branching Memory with import/export APIs.

    Notes
    -----
    - Supports variable-length candidate sets per transition.
    - `export_dict()` outputs plain Python/NumPy objects (CPU) safe for pickling.
    - `import_dict()` reconstructs tensors on this Memory's device.
    - Advantages/returns are optional. If not filled, they are omitted in exports.
    """

    def __init__(self, batch_size, state_dims, device, logger=None):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.batch_size = batch_size
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.cands_states: List[T.Tensor] = []   # (n_cands, var_dim)
        self.mip_states: List[T.Tensor] = []     # (mip_dim,)
        self.node_states: List[T.Tensor] = []    # (node_dim,)
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        # Optional – filled when GAE is computed
        self.advantages: List[float] = []
        self.returns: List[float] = []

        self.var_dim = state_dims["var_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.node_dim = state_dims["node_dim"]

        # --- 新增二分图特征列表 ---
        self.norm_conss: List[T.Tensor] = []      # (n_cons, cons_nfeats)
        self.norm_edge_idxs: List[T.Tensor] = []   # (2, n_edges)
        self.norm_edge_attrs: List[T.Tensor] = []  # (n_edges, edge_nfeats)
        self.norm_vars: List[T.Tensor] = []       # (n_vars, var_nfeats)
        self.norm_boundss: List[T.Tensor] = []     # (1, 2) 或 (2,)

    # ------------------------------------------------------------------
    # Core buffer ops
    # ------------------------------------------------------------------
    def clear(self):
        self.cands_states.clear()
        self.mip_states.clear()
        self.node_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()

        self.norm_conss.clear()
        self.norm_edge_idxs.clear()
        self.norm_edge_attrs.clear()
        self.norm_vars.clear()
        self.norm_boundss.clear()

        self.logger.info("Memory cleared successfully.")

    def _validate_inputs(self, cands_state, mip_state, node_state, action, reward, done, log_prob, value):
        if not isinstance(cands_state, T.Tensor):
            raise TypeError("cands_state must be a torch.Tensor")
        if cands_state.dim() != 2:
            raise ValueError("cands_state must be a 2D tensor (num_candidates x var_dim)")
        if cands_state.size(1) != self.var_dim:
            raise ValueError(f"cands_state must have {self.var_dim} columns, got {cands_state.size(1)}")

        if not isinstance(mip_state, T.Tensor) or mip_state.dim() != 1 or mip_state.size(0) != self.mip_dim:
            raise ValueError("mip_state must be a 1D tensor of correct length")
        if not isinstance(node_state, T.Tensor) or node_state.dim() != 1 or node_state.size(0) != self.node_dim:
            raise ValueError("node_state must be a 1D tensor of correct length")

        if not isinstance(action, int) or action < 0 or action >= cands_state.size(0):
            raise ValueError("Invalid action index")

        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            raise ValueError("reward must be finite number")
        if not isinstance(done, (bool, np.bool_, np.bool8)):
            raise TypeError("done must be a boolean")
        if not isinstance(log_prob, (int, float)) or np.isnan(log_prob) or np.isinf(log_prob):
            raise ValueError("log_prob must be finite number")
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            raise ValueError("value must be finite number")

    def store(self, cands_state: T.Tensor, mip_state: T.Tensor, node_state: T.Tensor,
                norm_cons: T.Tensor, norm_edge_idx: T.Tensor, norm_edge_attr: T.Tensor, 
                norm_var: T.Tensor, norm_bounds: T.Tensor,
              action: int, reward: float, done: bool, log_prob: float, value: float):
        self._validate_inputs(cands_state, mip_state, node_state, action, reward, done, log_prob, value)

        # cands_state = cands_state.detach().to(self.device)
        # mip_state = mip_state.detach().to(self.device)
        # node_state = node_state.detach().to(self.device)

        # 2. 这里的策略：如果是并行采样，建议存入 CPU 内存 (detach().cpu()) 以节省显存
        # 如果是在单卡上且内存充足，可以维持 self.device
        def _prepare(t):
            return t.detach().to(self.device) if isinstance(t, T.Tensor) else T.as_tensor(t, device=self.device)

        # self.cands_states.append(cands_state)
        # self.mip_states.append(mip_state)
        # self.node_states.append(node_state)
        # self.actions.append(int(action))
        # self.rewards.append(float(reward))
        # self.dones.append(bool(done))
        # self.log_probs.append(float(log_prob))
        # self.values.append(float(value))

        # 存储原始 3 个特征
        self.cands_states.append(_prepare(cands_state))
        self.mip_states.append(_prepare(mip_state))
        self.node_states.append(_prepare(node_state))

        # 3. 存储新增的 5 个二分图特征
        self.norm_conss.append(_prepare(norm_cons))
        # 注意：edge_idx 必须是 Long 类型
        self.norm_edge_idxs.append(_prepare(norm_edge_idx).long()) 
        self.norm_edge_attrs.append(_prepare(norm_edge_attr))
        self.norm_vars.append(_prepare(norm_var))
        self.norm_boundss.append(_prepare(norm_bounds))

        # 存储动作和奖励等标量
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.logger.debug(f"Stored transition #{len(self.cands_states)}")

    def set_advantages_returns(self, advantages: List[float], returns: List[float]):
        if len(advantages) != len(self.cands_states) or len(returns) != len(self.cands_states):
            raise ValueError("advantages/returns must match the number of transitions")
        if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
            raise ValueError("advantages contain NaNs/Infs")
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            raise ValueError("returns contain NaNs/Infs")
        self.advantages = [float(a) for a in advantages]
        self.returns = [float(r) for r in returns]

    def __len__(self) -> int:
        return len(self.cands_states)

    def is_empty(self) -> bool:
        return len(self.cands_states) == 0

    # ------------------------------------------------------------------
    # Batching (kept as in your version)
    # ------------------------------------------------------------------

    

    def get_batch_generator(self, batch_size: Optional[int] = None) -> Iterator[Tuple]:
        if self.is_empty():
            raise RuntimeError("Memory is empty. Cannot generate batches.")

        batch_size = batch_size or self.batch_size
        n_states = len(self.cands_states)
        indices = np.arange(n_states)
        np.random.shuffle(indices) # 打乱顺序

        for start_idx in range(0, n_states, batch_size):
            end_idx = min(start_idx + batch_size, n_states)
            batch_indices = indices[start_idx:end_idx]
            
            # 1. 提取基础特征 (原有逻辑)
            batch_cands_states = [self.cands_states[i] for i in batch_indices]
            batch_mip_states = [self.mip_states[i] for i in batch_indices]
            batch_node_states = [self.node_states[i] for i in batch_indices]

            # 2. 【新增】提取二分图特征
            batch_norm_cons = [self.norm_conss[i] for i in batch_indices]
            batch_norm_edge_idx = [self.norm_edge_idxs[i] for i in batch_indices]
            batch_norm_edge_attr = [self.norm_edge_attrs[i] for i in batch_indices]
            batch_norm_vars = [self.norm_vars[i] for i in batch_indices]
            batch_norm_bounds = [self.norm_boundss[i] for i in batch_indices]

            # 3. 处理候选变量 Padding (原有逻辑)
            max_candidates = max(cs.size(0) for cs in batch_cands_states)
            padded_cands_states, cands_masks = [], []
            for cs in batch_cands_states:
                n_cands = cs.size(0)
                padding_len = max_candidates - n_cands

                # 显式将 cs 搬运到 self.device (GPU)
                cs_on_device = cs.to(self.device)

                if padding_len > 0:
                    # 此时两者都在 GPU 上，cat 正常运行
                    padded_cs = T.cat([
                        cs_on_device, 
                        T.zeros(padding_len, self.var_dim, device=self.device)
                    ], dim=0)
                    mask = T.cat([
                        T.zeros(n_cands, dtype=T.bool, device=self.device), 
                        T.ones(padding_len, dtype=T.bool, device=self.device)
                    ], dim=0)
                else:
                    padded_cs = cs_on_device
                    mask = T.zeros(n_cands, dtype=T.bool, device=self.device)
                    
                padded_cands_states.append(padded_cs)
                cands_masks.append(mask)

                # padded_cs = T.cat([cs, T.zeros(padding_len, self.var_dim, device=self.device)], dim=0) if padding_len > 0 else cs
                # mask = T.cat([T.zeros(n_cands, dtype=T.bool, device=self.device), 
                #             T.ones(padding_len, dtype=T.bool, device=self.device)], dim=0) if padding_len > 0 else T.zeros(n_cands, dtype=T.bool, device=self.device)
                # padded_cands_states.append(padded_cs)
                # cands_masks.append(mask)

            # 4. 【核心修改】利用 PyG 工具打包二分图
            data_list = []
            for j in range(len(batch_indices)):
                d = BipartiteData(
                    x_s=batch_norm_vars[j],
                    x_c=batch_norm_cons[j],
                    edge_index=batch_norm_edge_idx[j],
                    edge_attr=batch_norm_edge_attr[j],
                    bounds=batch_norm_bounds[j]
                )
                # 必须手动设置 num_nodes 避免警告
                d.num_nodes = batch_norm_vars[j].size(0) + batch_norm_cons[j].size(0)
                data_list.append(d)
            
            # 【关键】使用 follow_batch，PyG 会自动生成准确的 pyg_batch.x_s_batch
            pyg_batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_c']).to(self.device)

            
           
            
            # data_list = []
            # for j in range(len(batch_indices)):
            #     # 1. 先通过 .size(0) 获取当前图中两类节点的实际数量
            #     num_vars = batch_norm_vars[j].size(0) # 变量节点（Variables）的数量
            #     num_cons = batch_norm_cons[j].size(0) # 约束节点（Constraints）的数量
            #     # 使用 Data 的子类或者手动指定属性
            #     d = Data(
            #         x_s=batch_norm_vars[j],
            #         x_c=batch_norm_cons[j],
            #         edge_index=batch_norm_edge_idx[j],
            #         edge_attr=batch_norm_edge_attr[j],
            #         bounds=batch_norm_bounds[j],
            #         num_nodes = num_vars + num_cons
            #     )
            #     data_list.append(d)
            
            # # 使用 follow_batch 参数强制生成指定的 batch 索引
            # pyg_batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_c']).to(self.device)

            # 5. 封装所有 Tensor 返回
            yield (
                T.stack(padded_cands_states),
                T.stack(cands_masks),
                T.stack(batch_mip_states),
                T.stack(batch_node_states),
                # 以下是二分图专有的 Batch 数据
                pyg_batch.x_c,          # 拼接后的约束特征 [Total_Cons, 4]
                pyg_batch.edge_index,   # 偏移后的边索引 [2, Total_Edges]
                pyg_batch.edge_attr,    # 拼接后的边特征 [Total_Edges, 1]
                pyg_batch.x_s,          # 拼接后的变量特征 [Total_Vars, 6]
                pyg_batch.bounds,       # 拼接后的边界 [B, 2]
                pyg_batch.x_c_batch,    # 约束节点对应的 Batch 索引 (用于池化)
                pyg_batch.x_s_batch,    # 变量节点对应的 Batch 索引 (用于池化)
                # 原有其他标量数据
                T.tensor([self.actions[i] for i in batch_indices], dtype=T.long, device=self.device),
                T.tensor([self.rewards[i] for i in batch_indices], dtype=T.float32, device=self.device),
                T.tensor([self.dones[i] for i in batch_indices], dtype=T.bool, device=self.device),
                T.tensor([self.log_probs[i] for i in batch_indices], dtype=T.float32, device=self.device),
                T.tensor([self.values[i] for i in batch_indices], dtype=T.float32, device=self.device),
                batch_indices,
            )
            
            del pyg_batch, data_list
    # def get_batch_generator(self, batch_size: Optional[int] = None) -> Iterator[Tuple[T.Tensor, ...]]:
    #     if self.is_empty():
    #         raise RuntimeError("Memory is empty. Cannot generate batches.")

    #     batch_size = batch_size or self.batch_size
    #     n_states = len(self.cands_states)
    #     indices = np.arange(n_states)
    #     np.random.shuffle(indices)

    #     for start_idx in range(0, n_states, batch_size):
    #         end_idx = min(start_idx + batch_size, n_states)
    #         batch_indices = indices[start_idx:end_idx]
    #         if len(batch_indices) == 0:
    #             continue

    #         batch_cands_states = [self.cands_states[i] for i in batch_indices]
    #         batch_mip_states = [self.mip_states[i] for i in batch_indices]
    #         batch_node_states = [self.node_states[i] for i in batch_indices]

    #         # 2. 【新增】提取二分图特征
    #         batch_norm_cons = [self.norm_conss[i] for i in batch_indices]
    #         batch_norm_edge_idx = [self.norm_edge_idxs[i] for i in batch_indices]
    #         batch_norm_edge_attr = [self.norm_edge_attrs[i] for i in batch_indices]
    #         batch_norm_vars = [self.norm_vars[i] for i in batch_indices]
    #         batch_norm_bounds = [self.norm_boundss[i] for i in batch_indices]

    #         batch_actions = [self.actions[i] for i in batch_indices]
    #         batch_rewards = [self.rewards[i] for i in batch_indices]
    #         batch_dones = [self.dones[i] for i in batch_indices]
    #         batch_log_probs = [self.log_probs[i] for i in batch_indices]
    #         batch_values = [self.values[i] for i in batch_indices]

    #         max_candidates = max(cs.size(0) for cs in batch_cands_states)
    #         padded_cands_states = []
    #         cands_masks = []
    #         for cs in batch_cands_states:
    #             n_cands = cs.size(0)
    #             if n_cands < max_candidates:
    #                 padding = T.zeros(max_candidates - n_cands, self.var_dim, device=self.device)
    #                 padded_cs = T.cat([cs, padding], dim=0)
    #                 mask = T.cat([
    #                     T.zeros(n_cands, dtype=T.bool, device=self.device),
    #                     T.ones(max_candidates - n_cands, dtype=T.bool, device=self.device),
    #                 ], dim=0)
    #             else:
    #                 padded_cs = cs
    #                 mask = T.zeros(n_cands, dtype=T.bool, device=self.device)
    #             padded_cands_states.append(padded_cs)
    #             cands_masks.append(mask)

    #         batch_cands_states_tensor = T.stack(padded_cands_states)
    #         batch_cands_masks_tensor = T.stack(cands_masks)
    #         batch_mip_states_tensor = T.stack(batch_mip_states)
    #         batch_node_states_tensor = T.stack(batch_node_states)
    #         batch_actions_tensor = T.tensor(batch_actions, dtype=T.long, device=self.device)
    #         batch_rewards_tensor = T.tensor(batch_rewards, dtype=T.float32, device=self.device)
    #         batch_dones_tensor = T.tensor(batch_dones, dtype=T.bool, device=self.device)
    #         batch_log_probs_tensor = T.tensor(batch_log_probs, dtype=T.float32, device=self.device)
    #         batch_values_tensor = T.tensor(batch_values, dtype=T.float32, device=self.device)

    #         # 【核心修改】利用 PyG 工具打包二分图
    #         data_list = []
    #         for j in range(len(batch_indices)):
    #             data_list.append(Data(
    #                 x_s = batch_norm_vars[j],      # 变量节点特征
    #                 x_c = batch_norm_cons[j],      # 约束节点特征
    #                 edge_index = batch_norm_edge_idx[j],
    #                 edge_attr = batch_norm_edge_attr[j],
    #                 bounds = batch_norm_bounds[j]  # 全局边界特征
    #             ))
            
    #         # Batch.from_data_list 会自动处理不同图的节点索引偏移
    #         pyg_batch = Batch.from_data_list(data_list).to(self.device)

    #         yield (
    #             batch_cands_states_tensor,
    #             batch_cands_masks_tensor,
    #             batch_mip_states_tensor,
    #             batch_node_states_tensor,

    #             # 以下是二分图专有的 Batch 数据
    #             pyg_batch.x_c,          # 拼接后的约束特征 [Total_Cons, 4]
    #             pyg_batch.edge_index,   # 偏移后的边索引 [2, Total_Edges]
    #             pyg_batch.edge_attr,    # 拼接后的边特征 [Total_Edges, 1]
    #             pyg_batch.x_s,          # 拼接后的变量特征 [Total_Vars, 6]
    #             pyg_batch.bounds,       # 拼接后的边界 [B, 2]
    #             pyg_batch.x_c_batch,    # 约束节点对应的 Batch 索引 (用于池化)
    #             pyg_batch.x_s_batch,    # 变量节点对应的 Batch 索引 (用于池化)

    #             batch_actions_tensor,
    #             batch_rewards_tensor,
    #             batch_dones_tensor,
    #             batch_log_probs_tensor,
    #             batch_values_tensor,
    #             batch_indices,
    #         )

    def batch(self, batch_size: Optional[int] = None) -> List[Tuple[T.Tensor, ...]]:
        return list(self.get_batch_generator(batch_size))

    # ------------------------------------------------------------------
    # Export / Import APIs
    # ------------------------------------------------------------------
    # 将 Memory（经验回放池）中存储的深度学习张量（Tensors）转换为通用的 NumPy 格式字典。
    def export_dict(self) -> Dict[str, Any]:
        """Export the whole buffer to a CPU/NumPy friendly dict for pickling.
        Variable-length candidate tensors are stored as a list of 2D float32 arrays.
        """
        n = len(self.cands_states)
        if n == 0:
            return {"num_transitions": 0}

        cands_list = [cs.detach().cpu().to(T.float32).numpy() for cs in self.cands_states]
        mip_mat = np.stack([ms.detach().cpu().to(T.float32).numpy() for ms in self.mip_states], axis=0)
        node_mat = np.stack([ns.detach().cpu().to(T.float32).numpy() for ns in self.node_states], axis=0)

        # 2. 【核心修改】转换新增的二分图特征
        # 因为约束和变量的数量在不同节点可能不同，所以用 list 存储，不进行 stack
        norm_cons_list = [t.detach().cpu().to(T.float32).numpy() for t in self.norm_conss]
        norm_var_list = [t.detach().cpu().to(T.float32).numpy() for t in self.norm_vars]
        norm_edge_attr_list = [t.detach().cpu().to(T.float32).numpy() for t in self.norm_edge_attrs]
        # edge_idx 必须存为 int64
        norm_edge_idx_list = [t.detach().cpu().to(T.int64).numpy() for t in self.norm_edge_idxs]
        norm_bounds_list = [t.detach().cpu().to(T.float32).numpy() for t in self.norm_boundss]

        data = {
            "num_transitions": n,
            "var_dim": self.var_dim,
            "mip_dim": self.mip_dim,
            "node_dim": self.node_dim,
            "cands_states": cands_list,              # list of (n_cands_i, var_dim)
            "mip_states": mip_mat,                   # (n, mip_dim)
            "node_states": node_mat,                 # (n, node_dim)

            # --- 新增二分图数据项 ---
            "norm_conss": norm_cons_list,
            "norm_vars": norm_var_list,
            "norm_edge_idxs": norm_edge_idx_list,
            "norm_edge_attrs": norm_edge_attr_list,
            "norm_boundss": norm_bounds_list,
            # ----------------------

            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }
        # Optional
        if len(self.advantages) == n and len(self.returns) == n:
            data["advantages"] = np.asarray(self.advantages, dtype=np.float32)
            data["returns"] = np.asarray(self.returns, dtype=np.float32)
        return data

    def import_dict(self, payload: Dict[str, Any]):
        """Append transitions from an exported dict into this memory (to `self.device`)."""
        if not payload or payload.get("num_transitions", 0) == 0:
            return
        n = int(payload["num_transitions"])  # type: ignore[index]
        cands_list = payload["cands_states"]
        mip_mat = payload["mip_states"]
        node_mat = payload["node_states"]

        # --- 新增二分图特征提取 ---
        cons_list = payload["norm_conss"]
        var_list = payload["norm_vars"]
        edge_idx_list = payload["norm_edge_idxs"]
        edge_attr_list = payload["norm_edge_attrs"]
        bounds_list = payload["norm_boundss"]
        # -----------------------

        actions = payload["actions"]
        rewards = payload["rewards"]
        dones = payload["dones"]
        log_probs = payload["log_probs"]
        values = payload["values"]
        advantages = payload.get("advantages", None)
        returns = payload.get("returns", None)

        # Basic sanity checks
        if len(cands_list) != n:
            raise ValueError("cands_states length mismatch")
        if mip_mat.shape[0] != n or node_mat.shape[0] != n:
            raise ValueError("state matrices length mismatch")
        if actions.shape[0] != n:
            raise ValueError("actions length mismatch")

        for i in range(n):
            cs_np = np.asarray(cands_list[i], dtype=np.float32)
            ms_np = np.asarray(mip_mat[i], dtype=np.float32)
            ns_np = np.asarray(node_mat[i], dtype=np.float32)

            # cs = T.from_numpy(cs_np).to(self.device)
            # ms = T.from_numpy(ms_np).to(self.device)
            # ns = T.from_numpy(ns_np).to(self.device)
            cs = T.from_numpy(cs_np).cpu()
            ms = T.from_numpy(ms_np).cpu()
            ns = T.from_numpy(ns_np).cpu()

            # --- 新增二分图特征转换 ---
            # 注意：edge_idx 需要转为 long (int64)
            # n_cons = T.from_numpy(np.asarray(cons_list[i], dtype=np.float32)).to(self.device)
            # n_var = T.from_numpy(np.asarray(var_list[i], dtype=np.float32)).to(self.device)
            # n_e_idx = T.from_numpy(np.asarray(edge_idx_list[i], dtype=np.int64)).to(self.device)
            # n_e_attr = T.from_numpy(np.asarray(edge_attr_list[i], dtype=np.float32)).to(self.device)
            # n_bnds = T.from_numpy(np.asarray(bounds_list[i], dtype=np.float32)).to(self.device)

            n_cons = T.from_numpy(np.asarray(cons_list[i], dtype=np.float32)).cpu()
            n_var = T.from_numpy(np.asarray(var_list[i], dtype=np.float32)).cpu()
            n_e_idx = T.from_numpy(np.asarray(edge_idx_list[i], dtype=np.int64)).cpu()
            n_e_attr = T.from_numpy(np.asarray(edge_attr_list[i], dtype=np.float32)).cpu()
            n_bnds = T.from_numpy(np.asarray(bounds_list[i], dtype=np.float32)).cpu()
            # -----------------------

            act = int(actions[i])
            rew = float(rewards[i])
            dn = bool(dones[i])
            lp = float(log_probs[i])
            val = float(values[i])

            # 数据验证 (验证二分图特征是否为空或维度异常)
            if n_e_idx.dim() != 2 or n_e_idx.size(0) != 2:
                raise ValueError(f"Imported edge_index at index {i} has wrong shape")

            # Minimal validation
            if cs.dim() != 2 or cs.size(1) != self.var_dim:
                raise ValueError("Imported cands_state has wrong shape")
            if ms.dim() != 1 or ms.size(0) != self.mip_dim:
                raise ValueError("Imported mip_state has wrong shape")
            if ns.dim() != 1 or ns.size(0) != self.node_dim:
                raise ValueError("Imported node_state has wrong shape")
            if act < 0 or act >= cs.size(0):
                # Clamp or raise; raising helps catch bugs early
                raise ValueError("Imported action out of range for its candidate set")

            self.cands_states.append(cs)
            self.mip_states.append(ms)
            self.node_states.append(ns)

            # --- 存入新增的列表 ---
            self.norm_conss.append(n_cons)
            self.norm_vars.append(n_var)
            self.norm_edge_idxs.append(n_e_idx)
            self.norm_edge_attrs.append(n_e_attr)
            self.norm_boundss.append(n_bnds)
            # --------------------

            self.actions.append(act)
            self.rewards.append(rew)
            self.dones.append(dn)
            self.log_probs.append(lp)
            self.values.append(val)

            if advantages is not None and returns is not None:
                # Append lazily; convert to float
                self.advantages.append(float(advantages[i]))
                self.returns.append(float(returns[i]))
        self.logger.debug(f"Successfully imported {n} transitions with bipartite graph features.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_all_data(self) -> Tuple[List[T.Tensor], List[T.Tensor], List[T.Tensor],
                    List[T.Tensor], List[T.Tensor], List[T.Tensor], List[T.Tensor], List[T.Tensor],
                    List[int], List[float], List[bool], List[float], List[float]]:
        if self.is_empty():
            raise RuntimeError("Memory is empty.")
        return (
            self.cands_states.copy(), self.mip_states.copy(), self.node_states.copy(),
            # --- 新增二分图特征 ---
            self.norm_conss.copy(),
            self.norm_edge_idxs.copy(),
            self.norm_edge_attrs.copy(),
            self.norm_vars.copy(),
            self.norm_boundss.copy(),
            # --------------------
            self.actions.copy(), self.rewards.copy(), self.dones.copy(),
            self.log_probs.copy(), self.values.copy(),
        )

    def get_memory_info(self) -> dict:
        if self.is_empty():
            return {
                "num_transitions": 0,
                "memory_empty": True,
                "batch_size": self.batch_size,
                "device": str(self.device),
            }
        cands_sizes = [cs.size(0) for cs in self.cands_states]

        # --- 新增：统计二分图规模 ---
        # 统计约束节点数量
        cons_sizes = [c.size(0) for c in self.norm_conss]
        # 统计边（非零项）的数量
        edge_counts = [ei.size(1) for ei in self.norm_edge_idxs]
        # --------------------------

        has_adv = len(self.advantages) == len(self.cands_states)
        has_ret = len(self.returns) == len(self.cands_states)
        return {
            "num_transitions": len(self.cands_states),
            "memory_empty": False,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "min_candidates": min(cands_sizes),
            "max_candidates": max(cands_sizes),
            "avg_candidates": float(sum(cands_sizes) / len(cands_sizes)),

                # --- 新增：二分图规模统计 ---
            "max_constraints": max(cons_sizes),
            "avg_constraints": float(sum(cons_sizes) / len(cons_sizes)),
            "max_edges": max(edge_counts),
            "avg_edges": float(sum(edge_counts) / len(edge_counts)),
            # --------------------------

            "var_dim": self.var_dim,
            "mip_dim": self.mip_dim,
            "node_dim": self.node_dim,
            "has_advantages": has_adv,
            "has_returns": has_ret,
        }
