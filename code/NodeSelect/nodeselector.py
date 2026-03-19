import pyscipopt as scip
import torch as T
import numpy as np
import traceback
from pyscipopt import Model
from pyscipopt.scip import Nodesel
from NodeSelect.bi_graph import LPFeatureRecorder
from BiGragh.utils import normalize_graph
from project.utils.functions import SCIPStateExtractor


class NodeSelector(Nodesel):
    def __init__(self, model, state_dims, device, agent, reward_func, cutoff, logger, recorder):
        super().__init__()
        self.model = model
        
        # 注意：这里我们依然保留了 var_dim 变量名以便兼容现有的 Agent，
        # 但实际上它现在代表的是“候选结点特征”的维度 (Node Feature Dimension)
        # 1. 候选结点的特征维度（3维）
        self.var_dim = state_dims["var_dim"]
        self.node_cand_dim = state_dims["node_cand_dim"] 
        self.node_dim = state_dims["node_dim"]
        self.mip_dim = state_dims["mip_dim"]
        
        self.device = device
        self.agent = agent
        self.reward_func = reward_func
        self.logger = logger
        self.cutoff = abs(cutoff) if cutoff not in (None, 0) else 1e-6
        self.episode_rewards = []

        self.select_count = 0  # 替换原来的 branch_count
        
        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

        self.recorder = recorder

    def _is_solved(self):
        status = self.model.getStatus()
        return status in ("optimal", "infeasible", "unbounded", "timelimit")

    # =========================================================================
    # 核心改造点 1：回调函数名称从 branchexeclp 改为 nodeselect
    # =========================================================================
    def nodeselect(self):
        try:
            self.select_count += 1

            # =========================================================================
            # 核心改造点 2：获取 Open Nodes 并手动提取特征
            # =========================================================================
            #children:当前节点刚分支出来的子节点; siblings:当前节点的兄弟节点;leaves:其他还没被探索的叶子节点,通常是更“旧”的节点
            leaves, children, siblings = self.model.getOpenNodes()
            # 记录每个结点的原始列表归属，用于构造特征
            raw_candidates = []
            for n in children: raw_candidates.append((n, [1, 0, 0])) # Child
            for n in siblings: raw_candidates.append((n, [0, 1, 0])) # Sibling
            for n in leaves:   raw_candidates.append((n, [0, 0, 1])) # Leaf

            if not raw_candidates:
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}
            # 2. 提取 6 维特征 (基础 3 维 + 类别 3 维)
            node_feats_list = []
            open_nodes = []
            SCIP_INF = 1e15
            for node, type_onehot in raw_candidates:
                lb = node.getLowerbound()
                est = node.getEstimate()
                depth = node.getDepth()
                # 裁剪
                lb = np.clip(lb, -SCIP_INF, SCIP_INF)
                est = np.clip(est, -SCIP_INF, SCIP_INF)
                
                # 合并为一个 6 维向量
                feat_vec = np.array([lb, est, depth] + type_onehot, dtype=np.float32)
                node_feats_list.append(feat_vec)
                open_nodes.append(node)
            # feature=[lb,est,depth,type1​,type2​,type3​] 下届，估计值，深度
            cands_state_mat = np.array(node_feats_list, dtype=np.float32)
            
            # 【新增】：对候选节点的连续特征 (lb, est, depth) 进行局部归一化
            if len(cands_state_mat) > 1:
                # 只取出前 3 列 (lb, est, depth)
                continuous_feats = cands_state_mat[:, :3]
                
                # 计算这批节点在各列上的最小值和最大值
                col_min = np.min(continuous_feats, axis=0, keepdims=True)
                col_max = np.max(continuous_feats, axis=0, keepdims=True)
                col_range = col_max - col_min
                
                # 防止所有节点在某特征上完全一样导致除以 0
                col_range[col_range == 0] = 1.0 
                
                # Min-Max 缩放到 [0, 1] 区间
                normalized_continuous = (continuous_feats - col_min) / col_range
                
                # 重新赋值回矩阵的前 3 列
                cands_state_mat[:, :3] = normalized_continuous
            elif len(cands_state_mat) == 1:
                # 只有一个候选节点时，相对特征全部置为 0（反正是唯一选择）
                cands_state_mat[:, :3] = 0.0

            # =========================================================================
            # 核心改造点 3：截断机制 (截取最有希望的前 N 个结点)
            # =========================================================================
            MAX_SEEDS = 150
            if len(open_nodes) > MAX_SEEDS:
                # 结点选择与变量分支不同：结点的下界 (Lower Bound) 越小，通常越有希望(针对最小化问题)
                # 我们用第 0 列 (即 lb) 进行升序排序，取前 MAX_SEEDS 个最小的
                scores = cands_state_mat[:, 0]

                # 【新增防御】：如果是最大化问题，分数取反，保证 argsort 依然能取到最“大”的
                if self.model.getObjectiveSense() == 'maximize':
                    scores = -scores
                
                # np.argsort 默认从小到大排，取前面最小的
                top_k_indices = np.argsort(scores)[:MAX_SEEDS]
                
                open_nodes = [open_nodes[i] for i in top_k_indices]
                cands_state_mat = cands_state_mat[top_k_indices, :]
            
            # -------------------------------------------------------------------------

            # 节点特征（局部树状态）和 MIP特征（全局状态）保持不变，无需归一化，底层已经归一化
            # node_state = self.model.getNodeState(self.node_dim).astype('float32')
            # mip_state = self.model.getMIPState(self.mip_dim).astype('float32')

            node_state = SCIPStateExtractor.get_node_state(self.model, self.node_dim).astype('float32')
            mip_state = SCIPStateExtractor.get_mip_state(self.model, self.mip_dim).astype('float32')

            # 1. 获取当前节点 (虽然我们即将跳去别的地方，但图结构可以用当前的全局/局部视角提取)
            curr_node = self.model.getCurrentNode()
            #如果当前节点不存在，就用 根节点（root node） 代替
            if curr_node is None:
                leaves, children, siblings = self.model.getOpenNodes()
                open_nodes = leaves + children + siblings
                if open_nodes:
                    # 用队列里的第一个开放节点代替，完美化解 NoneType 危机
                    curr_node = open_nodes[0]

            
            #  提取图结构
            self.recorder.record_sub_milp_graph(self.model, curr_node, task='node_select',k_hops=2)
            
            graph_data = self.recorder.recorded[curr_node.getNumber()]
            
            var_feats = graph_data.var_attributes 
            cons_feats = graph_data.cons_attributes   
            edge_index = graph_data.local_edge_index   
            edge_attr = graph_data.local_edge_attr     
            
            lb_graph, ub_graph = curr_node.getLowerbound(), curr_node.getEstimate()
            if self.model.getObjectiveSense() == 'maximize':
                lb_graph, ub_graph = ub_graph, lb_graph
                
            bounds = T.tensor([[lb_graph, -1 * ub_graph]], device=self.device).float()
            depth = T.tensor([curr_node.getDepth()], device=self.device).float()

            # 归一化局部图数据
            norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, _ = normalize_graph(
                cons_feats, edge_index, edge_attr, 
                var_feats, bounds, depth
            )

            # 转格式 (这里的 cands_state_tensor 现在装载的是候选结点的特征)
            cands_state_tensor = T.tensor(cands_state_mat, dtype=T.float32)
            if cands_state_tensor.dim() == 1:
                n_candidates = len(open_nodes)
                if cands_state_tensor.shape[0] == n_candidates * self.node_cand_dim:
                    cands_state_tensor = cands_state_tensor.view(n_candidates, self.node_cand_dim)
                else:
                    cands_state_tensor = cands_state_tensor.unsqueeze(0)

            cands_state_tensor = cands_state_tensor.to(self.device)
            node_state_tensor = T.from_numpy(node_state).to(self.device)
            mip_state_tensor = T.from_numpy(mip_state).to(self.device)

            # 存储上一步的 transition
            if self.prev_state is not None:
                done = self._is_solved()
                step_reward = self.reward_func.compute(self.model, done)
                try:
                    self.agent.remember(
                        cands_state=self.prev_state['cands_state'],
                        mip_state=self.prev_state['mip_state'],
                        node_state=self.prev_state['node_state'],
                        norm_cons=self.prev_state['norm_cons'],
                        norm_edge_idx=self.prev_state['norm_edge_idx'],
                        norm_edge_attr=self.prev_state['norm_edge_attr'],
                        norm_var=self.prev_state['norm_var'],
                        norm_bounds=self.prev_state['norm_bounds'],
                        action=self.prev_action,
                        reward=float(step_reward) * 0.01,
                        done=bool(done),
                        value=float(self.prev_value),
                        log_prob=float(self.prev_log_prob),
                    )
                except Exception:
                    self.logger.error("Failed to store transition")
                    self.logger.error(traceback.format_exc())
                self.episode_rewards.append(float(step_reward))
                
                if done:
                    return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            padding_mask = None

            # 请求智能体进行决策
            # final_action 会返回在 open_nodes 中的索引
            final_action, value, log_prob, truncated_state, truncated_action = self.agent.choose_action(
                cands_state_tensor.cpu().numpy(),
                mip_state_tensor.cpu().numpy(),
                node_state_tensor.cpu().numpy(),
                norm_cons=norm_cons.cpu().numpy(),          
                norm_edge_idx=norm_edge_idx.cpu().numpy(),  
                norm_edge_attr=norm_edge_attr.cpu().numpy(),
                norm_var=norm_var.cpu().numpy(),            
                norm_bounds=norm_bounds.cpu().numpy(),      
                padding_mask=padding_mask,
                deterministic=False,
            )

            if not (0 <= final_action < len(open_nodes)):
                self.logger.error(f"Invalid final_action selected: {final_action}")
                final_action = 0

            # =========================================================================
            # 核心改造点 4：返回选择的结点给 SCIP
            # =========================================================================
            selected_node = open_nodes[final_action] 

            # 记录本步状态，供下一次迭代保存
            self.prev_state = {
                'cands_state': truncated_state.clone(), 
                'mip_state': mip_state_tensor.clone(),
                'node_state': node_state_tensor.clone(),
                'norm_cons': norm_cons.clone() if T.is_tensor(norm_cons) else norm_cons,
                'norm_edge_idx': norm_edge_idx.clone() if T.is_tensor(norm_edge_idx) else norm_edge_idx,
                'norm_edge_attr': norm_edge_attr.clone() if T.is_tensor(norm_edge_attr) else norm_edge_attr,
                'norm_var': norm_var.clone() if T.is_tensor(norm_var) else norm_var,
                'norm_bounds': norm_bounds.clone() if T.is_tensor(norm_bounds) else norm_bounds,
            }
            self.prev_action = truncated_action 
            self.prev_value = float(value.item() if isinstance(value, np.ndarray) else value)
            self.prev_log_prob = float(log_prob.item() if isinstance(log_prob, np.ndarray) else log_prob)

            # 【新增】：无缝对接新版奖励函数，提取刚才这个动作的潜力与跳跃代价
            norm_lb = cands_state_mat[final_action, 0] #归一化后的下界
            norm_est = cands_state_mat[final_action, 1] #估计值，未来潜力估计
            
            # 从 One-hot 编码 (后3列) 提取跳跃代价
            is_child = cands_state_mat[final_action, 3]    # 顺着搜，代价 0
            is_sibling = cands_state_mat[final_action, 4]  # 兄弟回溯，代价 0.5
            is_leaf = cands_state_mat[final_action, 5]     # 远端乱跳，代价 1.0
            
            switch_penalty = 0.0 * is_child + 0.2 * is_sibling + 0.8 * is_leaf
            
            # 把这两个关键信息传给 Reward 缓存，供下一次 step 的 compute() 结算使用
            self.reward_func.set_action_feedback(norm_lb, norm_est, switch_penalty)

            # Nodesel 需要在字典里返回选中的节点给 SCIP 
            return {"selnode": selected_node}

        except Exception as e:
            self.logger.error(f"Exception in node selection rule: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

    # =========================================================================
    # 因为 Nodesel 没有 branchfree 的原生回调，你需要提供一个给 Environment 在结束时手动调用的方法
    # =========================================================================
    def finalize_episode(self):
        """
        供 Environment 在 Episode 结束 (model.optimize() 运行完毕) 后调用，
        用于把最后一个 terminal transition 存入 PPO 内存。
        """
        if self.prev_state is not None:
            done = True
            terminal_reward = self.reward_func.compute(self.model, done)
            try:
                self.agent.remember(
                    cands_state=self.prev_state['cands_state'],
                    mip_state=self.prev_state['mip_state'],
                    node_state=self.prev_state['node_state'],
                    norm_cons=self.prev_state['norm_cons'],
                    norm_edge_idx=self.prev_state['norm_edge_idx'],
                    norm_edge_attr=self.prev_state['norm_edge_attr'],
                    norm_var=self.prev_state['norm_var'],
                    norm_bounds=self.prev_state['norm_bounds'],
                    action=self.prev_action,
                    reward=float(terminal_reward),
                    done=True,
                    value=float(self.prev_value),
                    log_prob=float(self.prev_log_prob),
                )
            except Exception:
                self.logger.error("Failed to store final transition")
                self.logger.error(traceback.format_exc())
            
            self.episode_rewards.append(float(terminal_reward))
            self.logger.info(
                f"Final transition stored - Reward: {terminal_reward:.4f}, Total episode reward: {sum(self.episode_rewards):.4f}"
            )

        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

    def get_episode_stats(self):
        return {
            'select_count': self.select_count,
            'episode_rewards': self.episode_rewards.copy(),
            'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0.0,
        }