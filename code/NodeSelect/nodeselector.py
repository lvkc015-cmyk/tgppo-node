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
    def __init__(self, model, state_dims, device, agent, reward_func, cutoff, logger, recorder,depth_threshold=-1,use_gating=False, deterministic=False):
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
        # Keep step and terminal rewards on the same scale once they enter PPO memory.
        self.memory_reward_scale = 0.01

        self.select_count = 0  # 替换原来的 branch_count
        
        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

        self.recorder = recorder
        self.depth_threshold = depth_threshold
        self.use_gating = use_gating
        self.deterministic = deterministic

    def _is_solved(self):
        status = self.model.getStatus()
        return status in ("optimal", "infeasible", "unbounded", "timelimit")

    
    def nodeselect(self):
        try:
            self.select_count += 1

            if getattr(self, 'use_gating', False):
                
                # 1. 获取界限更新情况与深度
                num_sols_found = len(self.model.getSols())
                current_depth = self.model.getDepth()

                # 2. 策略一：永久交棒 (极致早停)
                # 触发条件：已找到4个以上可行解 OR 深度超标 OR 步数超标
                if num_sols_found >= 4 or (self.depth_threshold > 0 and current_depth > self.depth_threshold) or self.select_count > 1000:
                    
                    # 动态自废武功！把自己的优先级当场设为 0
                    # 【注意】这里的 "TreeGatePPO_NodeSelector" 必须与你 includeNodesel 时的 name 一致
                    self.model.setIntParam("nodeselection/TreeGatePPO_NodeSelector/stdpriority", 0)
                    
                    # 抛出 SCIP 底层最佳下界节点，安全交接
                    best_node = self.model.getBestboundNode()
                    if best_node is not None:
                        return {"selnode": best_node}
                    else:
                        return {"result": scip.SCIP_RESULT.DIDNOTRUN}

                # 3. 策略二：局部跳帧 (Dynamic Frame Skipping)
                if self.select_count <= 100:
                    skip_interval = 1   # 前 100 步：每步都算
                elif self.select_count <= 500:
                    skip_interval = 5   # 100~500 步：每 5 步算一次
                elif self.select_count <= 1000:
                    skip_interval = 10  # 500 步以后：每 10 步算一次
                else:
                    skip_interval = 100000

                # 如果不满足频率，立刻抛出最佳下界节点交棒
                if self.select_count % skip_interval != 0:
                    best_node = self.model.getBestboundNode()
                    if best_node is not None:
                        return {"selnode": best_node}
                    else:
                        return {"result": scip.SCIP_RESULT.DIDNOTRUN}  

            # =========================================================================
            # 核心改造点 2：获取 Open Nodes 并手动提取特征
            # =========================================================================
            #children:当前节点刚分支出来的子节点; siblings:当前节点的兄弟节点;leaves:其他还没被探索的叶子节点,通常是更“旧”的节点
            leaves, children, siblings = self.model.getOpenNodes()
            raw_nodes = children + siblings + leaves
            # type_labels = [[1, 0, 0]] * len(children) + [[0, 1, 0]] * len(siblings) + [[0, 0, 1]] * len(leaves)

            if not raw_nodes:
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}
            open_nodes = raw_nodes

            # 记录每个结点的原始列表归属，用于构造特征
            # raw_candidates = []
            # for n in children: raw_candidates.append((n, [1, 0, 0])) # Child
            # for n in siblings: raw_candidates.append((n, [0, 1, 0])) # Sibling
            # for n in leaves:   raw_candidates.append((n, [0, 0, 1])) # Leaf

            # if not raw_candidates:
            #     return {"result": scip.SCIP_RESULT.DIDNOTRUN}
            
            # cands_state_mat = getCandidateFeatures(self.model, raw_nodes, type_labels)
            # cands_state_mat = getCandidateFeatures(self.model)
            cands_state_mat = self._extract_candidate_features(raw_nodes, children, siblings, leaves)

            # 2. 提取 6 维特征 (基础 3 维 + 类别 3 维)
            # node_feats_list = []
            # open_nodes = []
            # SCIP_INF = 1e15

            # # 【获取全局的当前最优解 (Primal Bound)】
            # primal_bound = self.model.getPrimalbound()
            # is_maximize = (self.model.getObjectiveSense() == 'maximize')

            # for node, type_onehot in raw_candidates:
            #     lb = node.getLowerbound()
            #     est = node.getEstimate()
            #     depth = node.getDepth()
            #     # 裁剪
            #     lb = np.clip(lb, -SCIP_INF, SCIP_INF)
            #     est = np.clip(est, -SCIP_INF, SCIP_INF)

            #     # ---------------------------------------------------------
            #     # 🔥 新特征 1：Parent-Child Bound Diff (劣化程度)
            #     # 反映顺着这条树枝往下走，下界变差了多少。差值越小，说明这条路越平坦。
            #     # ---------------------------------------------------------
            #     parent = node.getParent()
            #     if parent is not None:
            #         parent_lb = np.clip(parent.getLowerbound(), -SCIP_INF, SCIP_INF)
            #         bound_diff = lb - parent_lb
            #     else:
            #         bound_diff = 0.0 # 根节点没有父节点
                    
            #     # ---------------------------------------------------------
            #     # 🔥 新特征 2：Distance to Incumbent (与当前全局最优解的距离)
            #     # 衡量这个节点离被“剪支”还有多远，或者多有希望能更新全局最优解。
            #     # ---------------------------------------------------------
            #     if primal_bound >= SCIP_INF or primal_bound <= -SCIP_INF:
            #         dist_to_incumbent = 0.0 # 还没找到任何可行解，暂时给 0
            #     else:
            #         if is_maximize:
            #             dist_to_incumbent = lb - primal_bound # 最大化问题：上界(lb在SCIP里统称) - 下界(PB)
            #         else:
            #             dist_to_incumbent = primal_bound - lb # 最小化问题：上界(PB) - 下界(lb)
                
            #     dist_to_incumbent = np.clip(dist_to_incumbent, -SCIP_INF, SCIP_INF)
                
            #     # 合并为一个 6 维向量
            #     feat_vec = np.array([lb, est, depth,bound_diff, dist_to_incumbent] + type_onehot, dtype=np.float32)
            #     node_feats_list.append(feat_vec)
            #     open_nodes.append(node)
            # # feature=[lb,est,depth,type1​,type2​,type3​] 下届，估计值，深度
            # cands_state_mat = np.array(node_feats_list, dtype=np.float32)
            
            # # 【新增】：对候选节点的连续特征 (lb, est, depth) 进行局部归一化
            # if len(cands_state_mat) > 1:
            #     # 只取出前 3 列 (lb, est, depth)
            #     continuous_feats = cands_state_mat[:, :5]
                
            #     # 计算这批节点在各列上的最小值和最大值
            #     col_min = np.min(continuous_feats, axis=0, keepdims=True)
            #     col_max = np.max(continuous_feats, axis=0, keepdims=True)
            #     col_range = col_max - col_min
                
            #     # 防止所有节点在某特征上完全一样导致除以 0
            #     col_range[col_range == 0] = 1.0 
                
            #     # Min-Max 缩放到 [0, 1] 区间
            #     normalized_continuous = (continuous_feats - col_min) / col_range
                
            #     # 重新赋值回矩阵的前 5 列
            #     cands_state_mat[:, :5] = normalized_continuous
            # elif len(cands_state_mat) == 1:
            #     # 只有一个候选节点时，相对特征全部置为 0（反正是唯一选择）
            #     cands_state_mat[:, :5] = 0.0

            # =========================================================================
            # 核心改造点 3：截断机制 (截取最有希望的前 N 个结点)
            # =========================================================================
            # MAX_SEEDS = 150
            
            
            # if len(open_nodes) > MAX_SEEDS:
            #     # 结点选择与变量分支不同：结点的下界 (Lower Bound) 越小，通常越有希望(针对最小化问题)
            #     # 我们用第 0 列 (即 lb) 进行升序排序，取前 MAX_SEEDS 个最小的
            #     scores = cands_state_mat[:, 0]

            #     # 【新增防御】：如果是最大化问题，分数取反，保证 argsort 依然能取到最“大”的
            #     # if self.model.getObjectiveSense() == 'maximize':
            #     #     scores = -scores
                
            #     # np.argsort 默认从小到大排，取前面最小的
            #     top_k_indices = np.argsort(scores)[:MAX_SEEDS]
                
            #     # open_nodes = [open_nodes[i] for i in top_k_indices]
            #     open_nodes = [raw_nodes[i] for i in top_k_indices]
            #     cands_state_mat = cands_state_mat[top_k_indices, :]

            current_max_seeds = 300 if self.use_gating else 150
            if len(open_nodes) > current_max_seeds:
                half_seeds = current_max_seeds // 2

                # 1. 获取所有节点的绝对下界和深度
                # 使用 raw_nodes 直接提取最稳妥，避免受特征矩阵归一化的影响
                lbs = np.array([node.getLowerbound() for node in raw_nodes])
                depths = np.array([node.getDepth() for node in raw_nodes])

                # 【防御机制】：如果是最大化问题，下界越大越好，取反后即可统一使用从小到大排序
                if self.model.getObjectiveSense() == 'maximize':
                    lbs = -lbs

                # 2. 第一轨：选出 LB 最好（最小）的一半，保证理论最优性
                best_lb_indices = np.argsort(lbs)[:half_seeds]

                # 3. 第二轨：选出 深度 最深（最大）的另一半，促进启发式剪枝和下潜
                # 生成布尔掩码，防止把已经被选进 best_lb_indices 的节点重复选入
                mask = np.ones(len(raw_nodes), dtype=bool)
                mask[best_lb_indices] = False
                remaining_indices = np.where(mask)[0]

                # 提取剩余节点的深度，取负号实现降序排序（深度越深排越前）
                remaining_depths = -depths[remaining_indices]
                
                # 计算还需要补齐多少个名额
                needed_seeds = current_max_seeds - len(best_lb_indices)
                
                # 在剩余节点中选出最深的 needed_seeds 个
                deepest_relative_indices = np.argsort(remaining_depths)[:needed_seeds]
                deepest_indices = remaining_indices[deepest_relative_indices]

                # 4. 合并两路人马，得到最终的 Top-K 索引
                top_k_indices = np.concatenate([best_lb_indices, deepest_indices])

                # 5. 更新节点列表和特征矩阵
                open_nodes = [raw_nodes[i] for i in top_k_indices]
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

            # =========================================================================
            # 🚨 测试期异常探针 1：记录提特征前 SCIP 的底层 LP 状态
            # =========================================================================
            # lp_status = self.model.getLPSolstat()
            # =========================================================================

            #  提取图结构
            self.recorder.record_sub_milp_graph(self.model, curr_node, task='node_select',k_hops=2)
            
            graph_data = self.recorder.recorded[curr_node.getNumber()]
            
            var_feats = graph_data.var_attributes 
            cons_feats = graph_data.cons_attributes   
            edge_index = graph_data.local_edge_index   
            edge_attr = graph_data.local_edge_attr     
            
            # =========================================================================
            # 🚨 测试期异常探针 2：拦截并分析图特征提取结果，外加急救包！(PyTorch版)
            # =========================================================================
            # if var_feats.numel() > 0:  # 👈 修复 1：用 numel() 检查 Tensor 是否为空
            #     var_max_val = T.max(T.abs(var_feats)).item() # 👈 修复 2：用 T.max 替代 np.max
                
            #     if var_max_val > 1e7:
            #         print(f"\n" + "!"*55)
            #         print(f"[TESTING PROBE] 捕获到极大特征值: {var_max_val:.2e}")
            #         print(f" -> 当前节点: #{curr_node.getNumber()}, 深度: {curr_node.getDepth()}")
            #         print(f" -> SCIP 当前 LP 求解状态 (LPSolstat): {lp_status}")
                    
            #         # 找出具体是哪一列爆炸了
            #         flat_idx = T.argmax(T.abs(var_feats)).item()
            #         row = flat_idx // var_feats.shape[1]
            #         col = flat_idx % var_feats.shape[1]
            #         col_map = {0:"sol", 1:"lb", 2:"ub", 5:"obj", 9:"is_sol_inf"}
            #         print(f" -> 爆炸位置: 行 {row}, 列 {col} ({col_map.get(col, '未知')})")
            #         print(f" -> 爆炸特征向量:\n {var_feats[row]}")
                    
            #         # 【急救修复机制】
            #         if lp_status != 'optimal' and col == 0:
            #             print(" -> [急救] 诊断：LP 未达到 optimal (可能由于截断或错误)。产生垃圾 sol 值。")
            #             print(" -> [急救] 动作：已强制将所有变量的 sol (第0列) 归零！")
            #             var_feats[:, 0] = 0.0
            #         else:
            #             print(" -> [急救] 动作：未知原因极大值，已强行把特征裁剪到 [-1e6, 1e6]！")
            #             var_feats = T.clamp(var_feats, min=-1e6, max=1e6) # 👈 修复 3：用 T.clamp 替代 np.clip
            #         print("!"*55 + "\n")
            # =========================================================================

            lb_graph, ub_graph = curr_node.getLowerbound(), curr_node.getEstimate()
            if self.model.getObjectiveSense() == 'maximize':
                lb_graph, ub_graph = ub_graph, lb_graph
                
            bounds = T.tensor([[lb_graph, -1 * ub_graph]], device=self.device).float()
            # =========================================================
            # 1. 物理截断：将绝对值限制在 1e7 以内，防止转换 float32 时产生 1e17 的畸形数值
            bounds = T.clamp(bounds, min=-1e7, max=1e7)
            
            # 2. 对数压缩：与你在 bi_graph.py 中处理变量特征的方法保持一致
            #    将较大的数值软化（例如 100000 -> 约 11.5），防止方差爆炸
            bounds = T.sign(bounds) * T.log1p(T.abs(bounds))

            depth = T.tensor([curr_node.getDepth()], device=self.device).float()

            

            # 归一化局部图数据
            norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, _ = normalize_graph(
                cons_feats, edge_index, edge_attr, 
                var_feats, bounds, depth, bound_normalizor=15.0
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
                        reward=float(step_reward) * self.memory_reward_scale,
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

            # 🚨 插入以下调试探针 🚨
            # print(f"\n[探针] 正在进行第 {self.model.getNNodes()} 个节点的选择...")
            # print(f"[探针] 当前 SCIP 给出的候选节点池大小: {len(open_nodes)}")
           
            
            # # 提取候选节点的数字 ID 看看
            # node_numbers = [n.getNumber() for n in open_nodes]
            # print(f"[探针] 候选节点的数字 ID: {node_numbers}")
            
            # # 验证是否有已经被截断 (cutoff) 甚至是 None 的废节点
            # node_status = [n.getLowerbound() for n in open_nodes]
            # print(f"[探针] 候选节点的松弛下界 (检查是否有 inf): {node_status}")

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
                deterministic=self.deterministic,
            )

            if not (0 <= final_action < len(open_nodes)):
                self.logger.error(f"Invalid final_action selected: {final_action}")
                final_action = 0

            # =========================================================================
            # 核心改造点 4：返回选择的结点给 SCIP
            # =========================================================================
            selected_node = open_nodes[final_action] 
            # print(f"[探针] 模型做出的选择 (Action index): {final_action.item()}")
            # print(f"[探针] 准备向 SCIP 提交选中的节点编号: {selected_node.getNumber()}\n")

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

            # Codex note: 第 0/1 列现在是候选池内的相对质量代价，范围保持在 [0, 1]，
            # 0 代表当前候选池里最好，1 代表当前候选池里最差，便于奖励函数继续复用。
            norm_lb = cands_state_mat[final_action, 0]
            norm_est = cands_state_mat[final_action, 1]
            
            # Codex note: type one-hot 仍然放在最后 3 维，便于下游逻辑稳定复用。
            is_child = cands_state_mat[final_action, 9]    # 顺着搜，代价 0
            is_sibling = cands_state_mat[final_action, 10]  # 兄弟回溯，代价 0.5
            is_leaf = cands_state_mat[final_action, 11]     # 远端乱跳，代价 1.0
            
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
                    reward=float(terminal_reward) * self.memory_reward_scale,
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
    
    def _extract_candidate_features(self, raw_nodes, children, siblings, leaves):
        """
        纯 Python + Numpy 的安全极速特征提取器
        提取 12 维 candidate-conditioned 特征，彻底避免 C 语言指针越界 (Segfault)
        """
        # Codex note: 这里保留 12 维而不是扩维，目的是不改 actor/critic 输入尺寸，
        # 但把候选特征从“绝对值”改成“相对当前候选池的位置与质量”。
        feature_dim = 12
        n_cands = len(raw_nodes)
        if n_cands == 0:
            return np.zeros((0, feature_dim), dtype=np.float32)

        SCIP_INF = self.model.infinity()
        global_pb = self.model.getPrimalbound()
        is_maximize = (self.model.getObjectiveSense() == 'maximize')
        eps = 1e-6

        # Codex note: 候选特征需要稳定缩放，否则不同实例上的数值尺度会直接污染策略网络。
        def _safe_minmax(values):
            if values.size == 0:
                return np.zeros(0, dtype=np.float32)
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            vrange = vmax - vmin
            if vrange <= eps:
                return np.zeros(values.shape, dtype=np.float32)
            return ((values - vmin) / vrange).astype(np.float32)

        def _safe_rank(values, ascending=True):
            n_vals = values.shape[0]
            if n_vals <= 1:
                return np.zeros(values.shape, dtype=np.float32)
            order = np.argsort(values if ascending else -values, kind='mergesort')
            ranks = np.empty(n_vals, dtype=np.float32)
            ranks[order] = np.linspace(0.0, 1.0, num=n_vals, dtype=np.float32)
            return ranks

        def _safe_center_tanh(values):
            if values.size == 0:
                return np.zeros(0, dtype=np.float32)
            scale = max(float(np.std(values)), eps)
            centered = (values - float(np.mean(values))) / scale
            return np.tanh(centered).astype(np.float32)

        has_pb = False
        if is_maximize:
            if global_pb > -SCIP_INF:
                has_pb = True
        else:
            if global_pb < SCIP_INF:
                has_pb = True

        lb_arr = np.zeros(n_cands, dtype=np.float32)
        est_arr = np.zeros(n_cands, dtype=np.float32)
        depth_arr = np.zeros(n_cands, dtype=np.float32)
        parent_lb_arr = np.zeros(n_cands, dtype=np.float32)

        for i, node in enumerate(raw_nodes):
            lb_arr[i] = node.getLowerbound()
            est_arr[i] = node.getEstimate()
            depth_arr[i] = node.getDepth()

            parent = node.getParent()
            if parent is not None:
                parent_lb_arr[i] = parent.getLowerbound()
            else:
                parent_lb_arr[i] = lb_arr[i]

        lb_arr = np.clip(lb_arr, -SCIP_INF, SCIP_INF)
        est_arr = np.clip(est_arr, -SCIP_INF, SCIP_INF)
        parent_lb_arr = np.clip(parent_lb_arr, -SCIP_INF, SCIP_INF)

        # Codex note: 统一把目标方向转换成“越小越好”的 cost 语义，
        # 这样最小化/最大化问题都能复用同一套候选排序特征。
        score_lb_arr = -lb_arr if is_maximize else lb_arr
        score_est_arr = -est_arr if is_maximize else est_arr
        score_parent_lb_arr = -parent_lb_arr if is_maximize else parent_lb_arr

        best_lb_score = np.min(score_lb_arr)
        best_est_score = np.min(score_est_arr)
        lb_gap_arr = score_lb_arr - best_lb_score
        est_gap_arr = score_est_arr - best_est_score
        parent_delta_arr = score_lb_arr - score_parent_lb_arr

        cands_state_mat = np.zeros((n_cands, feature_dim), dtype=np.float32)

        # 0-1: 当前候选相对池内最优候选的质量代价，保持 [0, 1] 便于奖励复用。
        cands_state_mat[:, 0] = _safe_minmax(lb_gap_arr)
        cands_state_mat[:, 1] = _safe_minmax(est_gap_arr)

        # 2-3: 当前候选在 open nodes 里的相对名次，0 为最好，1 为最差。
        cands_state_mat[:, 2] = _safe_rank(score_lb_arr, ascending=True)
        cands_state_mat[:, 3] = _safe_rank(score_est_arr, ascending=True)

        # 4-5: 深度相对位置，两种编码同时给网络，兼顾排序和离群程度。
        cands_state_mat[:, 4] = _safe_rank(depth_arr, ascending=True)
        cands_state_mat[:, 5] = _safe_center_tanh(depth_arr)

        # 6-7: 相对父节点的 bound 变化，既保留排序，也保留方向信息。
        cands_state_mat[:, 6] = _safe_rank(parent_delta_arr, ascending=True)
        cands_state_mat[:, 7] = _safe_center_tanh(parent_delta_arr)

        # 8: 相对 incumbent 的改进空间，在候选池内做 min-max，越大代表离 incumbent 留出的空间越多。
        if has_pb:
            incumbent_score = -global_pb if is_maximize else global_pb
            incumbent_slack = np.maximum(incumbent_score - score_lb_arr, 0.0)
            cands_state_mat[:, 8] = _safe_minmax(incumbent_slack)

        # 9-11: 保留候选类型 one-hot，继续供策略和奖励使用。
        n_child = len(children)
        n_sib = len(siblings)
        if n_child > 0:
            cands_state_mat[0:n_child, 9] = 1.0
        if n_sib > 0:
            cands_state_mat[n_child:n_child + n_sib, 10] = 1.0
        if len(leaves) > 0:
            cands_state_mat[n_child + n_sib:, 11] = 1.0

        return cands_state_mat
