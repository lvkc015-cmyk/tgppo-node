import pyscipopt as scip
import torch as T
import numpy as np
import traceback

from BiGragh.recorders import LPFeatureRecorder
from BiGragh.utils import normalize_graph


class Brancher(scip.Branchrule):
    def __init__(self, model, state_dims, device, agent, reward_func, cutoff, logger,recorder):
        super().__init__()
        self.model = model
        self.var_dim = state_dims["var_dim"]
        self.node_dim = state_dims["node_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.device = device
        self.agent = agent
        self.reward_func = reward_func
        self.logger = logger
        self.cutoff = abs(cutoff) if cutoff not in (None, 0) else 1e-6
        self.episode_rewards = []

        self.branch_count = 0
        self.branchexec_count = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

        self.recorder = recorder

    def _is_solved(self):
        status = self.model.getStatus()
        return status in ("optimal", "infeasible", "unbounded", "timelimit")

    def branchexeclp(self, allowaddcons):
        try:
            self.branchexec_count += 1

            #提取特征：cands：可分支变量列表；cands_pos：变量在 SCIP 内部的位置；cands_state_mat：候选变量特征矩阵
            cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branch_count)
            if not cands or cands_state_mat is None:
                self.logger.error("No candidates available for branching.")
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            
            # ----------------【种子截断】--------------------------
            MAX_SEEDS = 400
            if len(cands) > MAX_SEEDS:
                # 使用第 0 列 (lpcandsfrac) 排序，选出分数最大的前 400 个
                scores = cands_state_mat[:, 0]
                # 选取前 400 个索引 (这里用 np.argsort 因为 cands_state_mat 是 numpy 数组)
                top_k_indices = np.argsort(scores)[-MAX_SEEDS:]
                
                # 同步截断所有相关列表
                cands = [cands[i] for i in top_k_indices]
                cands_state_mat = cands_state_mat[top_k_indices, :]
                cands_pos = [cands_pos[i] for i in top_k_indices]
            #--------------------------------------------------------------------

            # 节点特征（node_dim=8维）
            node_state = self.model.getNodeState(self.node_dim).astype('float32')
            # MIP特征（mip_dim=53维）
            mip_state = self.model.getMIPState(self.mip_dim).astype('float32')

            ############################### 新加入 #################################
            # 1. 获取当前正在处理的节点
            curr_node = self.model.getCurrentNode()

            # 手动设置你需要的 K 值，例如 2
            K_HOPS = 2
            # 2. 更新并记录当前节点的二分图状态 触发局部图提取
            self.recorder.record_sub_milp_graph(self.model, curr_node, cands=cands,k_hops=K_HOPS)
            
            # 3. 提取图数据 根据当前节点的唯一编号，从 Recorder 的字典里把刚才提取好的“局部图对象”拿出来。
            graph_data = self.recorder.recorded[curr_node.getNumber()]
            
            var_feats = graph_data.var_attributes # 变量特征 [n_vars, 6]
            cons_feats = graph_data.cons_attributes   # 这是我们在 Recorder 中新存的
            edge_index = graph_data.local_edge_index   # 已经是局部映射后的索引 [0, 1000]
            edge_attr = graph_data.local_edge_attr     # 对应的边系数
            lb, ub = curr_node.getLowerbound(), curr_node.getEstimate()
            if self.model.getObjectiveSense() == 'maximize':
                lb, ub = ub, lb
                
            bounds = T.tensor([[lb, -1 * ub]], device=self.device).float()
            depth = T.tensor([curr_node.getDepth()], device=self.device).float()

            # ================== 变量取值验证审计 ==================
            # print(f"\n" + "!"*20 + " 数据流验证 " + "!"*20)
            # data_to_check = {
            #     "var_feats": var_feats,
            #     "cons_feats": cons_feats,
            #     "edge_index": edge_index,
            #     "edge_attr": edge_attr,
            #     "bounds": bounds,
            #     "depth": depth
            # }

            # for name, tensor in data_to_check.items():
            #     if tensor is None:
            #         print(f"CRITICAL: {name} 是 None!")
            #     elif isinstance(tensor, T.Tensor):
            #         # 打印形状，重点看第一维是不是 0
            #         print(f"CHECK: {name:12} | Shape: {list(tensor.shape)} | Device: {tensor.device}")
            #         if tensor.numel() == 0:
            #             print(f"   --> WARNING: {name} 元素数量为 0 (空张量)!")
            #     else:
            #         print(f"CHECK: {name:12} | Value: {tensor} (非 Tensor 类型)")

            # # 针对你之前的 Size(400) vs Size(0) 报错，做一个强制断言提示
            # if var_feats.shape[0] > 0 and cons_feats.shape[0] == 0:
            #     print(f"MATCH FOUND: 发现【有变量但没约束】的情况！这就是导致 GNN 崩溃的原因。")
            # print("!"*50 + "\n")
            # ====================================================


            # 6. 归一化 (直接传入局部图数据)
            norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, _ = normalize_graph(
                cons_feats, edge_index, edge_attr, 
                var_feats, bounds, depth
            )
            #  # 6. 归一化 (调用项目提供的 utils.py)
            # norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, _ = normalize_graph(
            #     cons_feats.clone(), edge_index.clone(), edge_attr.clone(), 
            #     var_feats.clone(), bounds.clone(), depth.clone()
            # )

            ####################################################################


            # 转格式
            if isinstance(cands_state_mat, np.ndarray):
                cands_state_tensor = T.from_numpy(cands_state_mat.astype(np.float32))
            else:
                cands_state_tensor = T.tensor(cands_state_mat, dtype=T.float32)

            # 确保最终的形状为[num_candidates, var_dim]
            if cands_state_tensor.dim() == 1:
                n_candidates = len(cands)
                if cands_state_tensor.shape[0] == n_candidates * self.var_dim:
                    cands_state_tensor = cands_state_tensor.view(n_candidates, self.var_dim)
                else:
                    cands_state_tensor = cands_state_tensor.unsqueeze(0)

            cands_state_tensor = cands_state_tensor.to(self.device)
            node_state_tensor = T.from_numpy(node_state).to(self.device)
            mip_state_tensor = T.from_numpy(mip_state).to(self.device)

            

            #存储“上一步”的 transition （s,a,r,s）
            if self.prev_state is not None:
                done = self._is_solved()
                #计算 reward
                step_reward = self.reward_func.compute(self.model, done)
                # 存入 PPO memory
                try:
                    self.agent.remember(
                        cands_state=self.prev_state['cands_state'],
                        mip_state=self.prev_state['mip_state'],
                        node_state=self.prev_state['node_state'],

                        norm_cons = self.prev_state['norm_cons'],
                        norm_edge_idx = self.prev_state['norm_edge_idx'],
                        norm_edge_attr = self.prev_state['norm_edge_attr'],
                        norm_var = self.prev_state['norm_var'],
                        norm_bounds = self.prev_state['norm_bounds'],

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
                self.logger.info(
                    f"Stored transition for branch {self.branch_count - 1} - "
                    f"Action: {self.prev_action}, Reward: {step_reward:.4f}, Done: {done}"
                )

                #如果 episode 已结束, 让 SCIP 自己收尾
                if done:
                    self.logger.info(f"Episode completed with total reward: {sum(self.episode_rewards):.4f}")
                    return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            padding_mask = None  # no padding at selection time (exact candidate set)

            #当前状态 → 选 action
            # action, value, log_prob = self.agent.choose_action(
            #     cands_state_tensor.cpu().numpy(),
            #     mip_state_tensor.cpu().numpy(),
            #     node_state_tensor.cpu().numpy(),
            #     padding_mask=padding_mask,
            #     deterministic=False,  #训练阶段,保留探索
            # )

            # 调用 choose_action，接收新增的 truncated_action
            #final_action:原始候选变量集合 cands 里的索引,只是一个索引
            #truncated_action:被截断后的候选集合里的索引
            #truncated_state:截断后的变量列表
            #truncated_action:选中的动作在truncated_state的下标
            final_action, value, log_prob, truncated_state, truncated_action = self.agent.choose_action(
                cands_state_tensor.cpu().numpy(),
                mip_state_tensor.cpu().numpy(),
                node_state_tensor.cpu().numpy(),

                norm_cons=norm_cons.cpu().numpy(),          # 形状 [n_cons, 4]
                norm_edge_idx=norm_edge_idx.cpu().numpy(),  # 形状 [2, n_edges]
                norm_edge_attr=norm_edge_attr.cpu().numpy(),# 形状 [n_edges, 1]
                norm_var=norm_var.cpu().numpy(),            # 形状 [n_vars, 6]
                norm_bounds=norm_bounds.cpu().numpy(),      # 形状 [1, 2]
                
                padding_mask=padding_mask,
                deterministic=False,
            )
            
            # if isinstance(action, np.ndarray):
            #     action = int(action.item() if action.size == 1 else action[0])
            # else:
            #     action = int(action)

            # if not (0 <= action < len(cands)):
            #     self.logger.error(f"Invalid action selected: {action} for {len(cands)} candidates")
            #     action = 0
            #     self.logger.warning("Using fallback action: 0")

            if not (0 <= final_action < len(cands)):
                self.logger.error(f"Invalid final_action selected: {final_action}")
                final_action = 0

            #执行 SCIP 分支,这是唯一真正改变 B&B 搜索树结构的地方
            # selected_var = cands[action]
            # self.model.branchVar(selected_var)

            # 1. 执行分支使用原始索引 (final_action)
            selected_var = cands[final_action]  #选择的变量
            self.model.branchVar(selected_var) #强制求解器对指定的变量进行分支（Branching）操作


            self.branch_count += 1

            # self.prev_state = {
            #     'cands_state': cands_state_tensor.clone(),
            #     'mip_state': mip_state_tensor.clone(),
            #     'node_state': node_state_tensor.clone(),
            # }
            # self.prev_action = action

            self.prev_state = {
                'cands_state': truncated_state.clone(), # 存入已经截断后的 500 维特征
                'mip_state': mip_state_tensor.clone(),
                'node_state': node_state_tensor.clone(),

                # 存入归一化后的 numpy 或 tensor 格式（建议 numpy 保持与前三个一致）
                'norm_cons': norm_cons.clone() if T.is_tensor(norm_cons) else norm_cons,
                'norm_edge_idx': norm_edge_idx.clone() if T.is_tensor(norm_edge_idx) else norm_edge_idx,
                'norm_edge_attr': norm_edge_attr.clone() if T.is_tensor(norm_edge_attr) else norm_edge_attr,
                'norm_var': norm_var.clone() if T.is_tensor(norm_var) else norm_var,
                'norm_bounds': norm_bounds.clone() if T.is_tensor(norm_bounds) else norm_bounds,

            }
            self.prev_action = truncated_action # 存入 0-499 之间的相对索引


            self.prev_value = float(value.item() if isinstance(value, np.ndarray) else value)
            self.prev_log_prob = float(log_prob.item() if isinstance(log_prob, np.ndarray) else log_prob)

            result = scip.SCIP_RESULT.BRANCHED

            # Optional validation
            try:
                if result == scip.SCIP_RESULT.BRANCHED:
                    children = self.model.getChildren()
                    if children:
                        branch_infos = children[0].getBranchInfos()
                        if len(branch_infos) > 1:
                            chosen_variable = branch_infos[1]
                            assert chosen_variable is not None, "Chosen variable is None"
                            assert chosen_variable.isInLP(), "Chosen variable is not in LP"
            except Exception:
                # Do not break on validation failure — logging only
                self.logger.debug("Validation check failed or unavailable.")

            return {"result": result}
        except Exception as e:
            self.logger.error(f"Exception in branching rule: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

    # 把最后一次 branching 决策对应的 transition（终止状态）存进 PPO 的 memory 里
    def branchfree(self):
        # Called when branchrule is freed; ensure we store final transition once
        if self.prev_state is not None:
            done = True
            terminal_reward = self.reward_func.compute(self.model, done)
            try:
                self.agent.remember(
                    cands_state=self.prev_state['cands_state'],
                    mip_state=self.prev_state['mip_state'],
                    node_state=self.prev_state['node_state'],
                    # --- 新增二分图特征 ---
                    norm_cons=self.prev_state['norm_cons'],
                    norm_edge_idx=self.prev_state['norm_edge_idx'],
                    norm_edge_attr=self.prev_state['norm_edge_attr'],
                    norm_var=self.prev_state['norm_var'],
                    norm_bounds=self.prev_state['norm_bounds'],
                    # --------------------
                    action=self.prev_action,
                    reward=float(terminal_reward) ,
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
            'branch_count': self.branch_count,
            'branchexec_count': self.branchexec_count,
            'episode_rewards': self.episode_rewards.copy(),
            'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0.0,
        }