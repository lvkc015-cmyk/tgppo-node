import torch
from NodeSelect.memory import Memory
import torch.nn.functional as F
from torch.distributions import Categorical
import traceback
import logging
import pandas as pd
import os
import gc

def save_metrics_to_csv(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([metrics])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


class Agent:
    """
    Policy Gradient Branching Agent (PPO).
    """

    def __init__(self, actor_network, actor_optimizer, critic_network, critic_optimizer,
                 policy_clip, entropy_weight, gamma, gae_lambda, batch_size, n_epochs,
                 device, state_dims, logger=None):
        self.actor_network = actor_network
        self.actor_optimizer = actor_optimizer
        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        self.policy_clip = policy_clip
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.state_dims = state_dims
        self.logger = logger or logging.getLogger(__name__)

        self.memory = Memory(self.batch_size, self.state_dims, self.device, self.logger)

        # Move networks to device
        self.actor_network
        self.critic_network

        # Training statistics
        self.training_step = 0
        self.update_counter = 0

    
    def remember(self, cands_state, mip_state, node_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,action, reward, done, value, log_prob):
        try:
            self.memory.store(
                cands_state=cands_state.detach().cpu(),
                mip_state=mip_state.detach().cpu(),
                node_state=node_state.detach().cpu(),

                # --- 新增二分图存储 ---
                norm_cons=norm_cons.detach().cpu(),
                norm_edge_idx=norm_edge_idx.detach().cpu(),
                norm_edge_attr=norm_edge_attr.detach().cpu(),
                norm_var=norm_var.detach().cpu(),
                norm_bounds=norm_bounds.detach().cpu(),

                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )
        except Exception:
            self.logger.error("Error in storing transition in memory")
            self.logger.error(traceback.format_exc())


    # 变量截取的选择函数; deterministic: 是否贪心选动作（评估用）
    def choose_action(self, cands_state, mip_state, node_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds, padding_mask=None, deterministic=False):
        try:
            # Ensure inputs are tensors on correct device
            cands_state = torch.as_tensor(cands_state, dtype=torch.float32, device=self.device)
            mip_state = torch.as_tensor(mip_state, dtype=torch.float32, device=self.device)
            node_state = torch.as_tensor(node_state, dtype=torch.float32, device=self.device)

            norm_cons = torch.as_tensor(norm_cons, dtype=torch.float32, device=self.device)
            norm_edge_idx = torch.as_tensor(norm_edge_idx, dtype=torch.long, device=self.device) # 索引通常用 long
            norm_edge_attr = torch.as_tensor(norm_edge_attr, dtype=torch.float32, device=self.device)
            norm_var = torch.as_tensor(norm_var, dtype=torch.float32, device=self.device)
            norm_bounds = torch.as_tensor(norm_bounds, dtype=torch.float32, device=self.device)

            # if norm_bounds.dim() == 1:
            #     norm_bounds = norm_bounds.unsqueeze(0)

            if cands_state.dim() == 2:
                cands_state = cands_state.unsqueeze(0)


            #如果是一维
            if mip_state.dim() == 1:
                #在第 0 个位置增加一个维度
                mip_state = mip_state.unsqueeze(0)
            if node_state.dim() == 1:
                node_state = node_state.unsqueeze(0)


            self.actor_network.eval()
            self.critic_network.eval()

            with torch.no_grad():
                #Actor & Critic 前向（核心）
                action_probs = self.actor_network(cands_state, node_state, mip_state,norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,padding_mask,mb_cons_batch = None,mb_var_batch=None)
                value = self.critic_network(cands_state, node_state, mip_state, norm_cons, norm_edge_idx, norm_edge_attr, norm_var, norm_bounds,padding_mask)
                value = value.squeeze(-1)  # [batch]

                if deterministic:
                    #直接选择概率分布中最大的那个位置
                    final_action = action_probs.argmax(dim=-1)
                else:
                    if torch.isnan(action_probs).any():
                        print("!!! Found NaN in action_probs !!!")
                        print(f"norm_var min/max: {norm_var.min()}, {norm_var.max()}")
                        print(f"norm_cons min/max: {norm_cons.min()}, {norm_cons.max()}")
                        print(f"edge_index size: {norm_edge_idx.shape}")
                    #根据概率分布进行随机采样
                    dist = Categorical(action_probs)
                    #得到的是一个整数索引,这个索引就是 AI 在当前截断后的变量池（500个变量）中选中的那个位置
                    final_action = dist.sample() 
                log_prob = torch.log(action_probs.gather(1, final_action.unsqueeze(-1)).clamp_min(1e-12))
               
            final_action_np = final_action.cpu().numpy()
            final_value = value.cpu().numpy()
            final_log_prob = log_prob.cpu().numpy()

            action_idx = final_action_np[0] if final_action_np.shape[0] == 1 else final_action_np

            if final_action_np.shape[0] == 1:
                return action_idx, final_value[0], final_log_prob[0],cands_state.squeeze(0), action_idx
            return action_idx, final_value, final_log_prob,cands_state.squeeze(0), action_idx
        except Exception as e:    
            self.logger.error(f"Error in choose_action: {e}")
            # self.logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def _compute_gae(self, rewards, values, dones, bootstrap_value):
        """Compute GAE with scalar bootstrap value for the state after the last step.
        Args:
            rewards: [T]
            values:  [T]
            dones:   [T] float tensor (0.0 or 1.0)
            bootstrap_value: scalar tensor for V(s_{T}) if episode not done, else 0
        Returns:
            advantages [T], returns [T]
        """
        T_ = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        next_value = bootstrap_value
        gae = torch.zeros((), device=rewards.device)
        for t in reversed(range(T_)):
            nonterminal = 1.0 - dones[t]
            nv = next_value if t == T_ - 1 else values[t + 1]
            nv = nv * (1.0 - dones[t + 1]) if t < T_ - 1 else nv
            delta = rewards[t] + self.gamma * nv * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        return advantages, returns


    def learn(self):
        
        try:
            self.logger.debug("Started Learning...")
            if self.memory.is_empty():
                self.logger.debug("Memory is empty, cannot train")
                return {}
            # 防止样本数太少导致梯度噪声爆炸
            if len(self.memory) < self.batch_size:
                self.logger.debug(f"Not enough samples in memory for training. Have {len(self.memory)}, need {self.batch_size}")
                return {}

            #从 memory 取出所有完整轨迹；这里是 on-policy、整轨迹 PPO，没有 replay 偏差
            (cands_states_list, mip_states_list, node_states_list,
            norm_conss_list,      # 新增
            norm_edge_idxs_list,   # 新增
            norm_edge_attrs_list,  # 新增
            norm_vars_list,       # 新增
            norm_boundss_list,
            actions_list, rewards_list, dones_list, log_probs_list, values_list) = self.memory.get_all_data()

            print(f"DEBUG: 真正用于训练的 Step 数量是: {len(actions_list)}")
            
            actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
            old_values = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=self.device)

            ###加入了 .to(self.device)
            mip_states = torch.stack(mip_states_list).to(self.device)
            node_states = torch.stack(node_states_list).to(self.device)

            # Build padded last state to compute bootstrap
            #B&B 的 action space 是可变大小的，对最后一个 state：padding 到 max_candidates
            #统计全部轨迹中的最大候选变量数
            max_candidates = max(cs.size(0) for cs in cands_states_list)
            #取轨迹最后一个状态的候选集合
            last_cs = cands_states_list[-1].to(self.device)
            if last_cs.size(0) < max_candidates:
                pad = torch.zeros(max_candidates - last_cs.size(0), last_cs.size(1), device=self.device)
                last_cs_padded = torch.cat([last_cs, pad], dim=0).unsqueeze(0)
                last_mask = torch.cat([
                    torch.zeros(last_cs.size(0), dtype=torch.bool, device=self.device),
                    torch.ones(max_candidates - last_cs.size(0), dtype=torch.bool, device=self.device),
                ], dim=0).unsqueeze(0)
            else:
                last_cs_padded = last_cs.unsqueeze(0)
                last_mask = torch.zeros(last_cs.size(0), dtype=torch.bool, device=self.device).unsqueeze(0)

            # 获取最后一个状态的二分图特征
            last_norm_cons = self.memory.norm_conss[-1].to(self.device)
            last_norm_edge_idx = self.memory.norm_edge_idxs[-1].to(self.device)
            last_norm_edge_attr = self.memory.norm_edge_attrs[-1].to(self.device)
            last_norm_var = self.memory.norm_vars[-1].to(self.device)
            last_norm_bounds = self.memory.norm_boundss[-1].to(self.device)

            #如果episode 结束, bootstrap_value=0
            if dones[-1] > 0.5:
                bootstrap_value = torch.tensor(0.0, device=self.device)
            #否则 → 用 critic 预测,预测轨迹尽头那个“未完待续”状态的预期收益（Bootstrap Value）
            else:
                with torch.no_grad():
                    # 2. 统一使用切片获取最后一行，保持 (1, D) 形状
                    last_node = node_states[-1:] # (1, hidden_dim)
                    last_mip = mip_states[-1:]   # (1, hidden_dim)

                    # 3. 调用网络
                    v_out = self.critic_network(
                        last_cs_padded,   # (1, N, D)
                        last_node,        # (1, D)
                        last_mip,         # (1, D)
                        last_norm_cons, 
                        last_norm_edge_idx, 
                        last_norm_edge_attr, 
                        last_norm_var, 
                        last_norm_bounds,
                        last_mask         # (1, N)
                    )
                    
                    # 4. 安全提取标量
                    bootstrap_value = v_out.view(-1)[0]

            #计算 GAE
            advantages, returns = self._compute_gae(rewards, old_values, dones, bootstrap_value)
            #这个动作是否比平均预期更好
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # === 诊断块 1：检查全局优势信号 ===
            print(f"--- DIAGNOSIS: GAE Stats ---")
            print(f"Advantage - Mean: {advantages.mean().item():.6f}, Std: {advantages.std().item():.6f}")
            print(f"Returns - Mean: {returns.mean().item():.6f}, Max: {returns.max().item():.6f}")
            # =================================

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            num_updates = 0

            self.actor_network.train()
            self.critic_network.train()

            
            for epoch in range(self.n_epochs):
                
                iter =0
                #循环次数（打印次数） = 数据总数 / Batch Size
                self.logger.debug(f"Epoch {epoch+1}/{self.n_epochs}")
                for batch in self.memory.get_batch_generator(self.batch_size):
                    (mb_cands_states, mb_cands_masks, mb_mip_states, mb_node_states,
                    mb_norm_cons, mb_norm_edge_idx, mb_norm_edge_attr, mb_norm_var, mb_norm_bounds,
                     mb_cons_batch, mb_var_batch,
                     mb_actions, _mb_rewards, _mb_dones, mb_old_log_probs, mb_old_values, batch_indices) = batch


                    iter = iter +1
                    self.logger.info(f"******iter********** : {iter}")
                    torch.cuda.empty_cache()
                    
                    mb_cands_states = mb_cands_states.to(self.device)
                    mb_cands_masks = mb_cands_masks.to(self.device)
                    mb_mip_states = mb_mip_states.to(self.device)
                    mb_node_states = mb_node_states.to(self.device)
                    mb_actions = mb_actions.to(self.device)
                    mb_old_log_probs = mb_old_log_probs.to(self.device)
                    mb_old_values = mb_old_values.to(self.device)

                    # 确保所有二分图数据在 GPU 上 (已经在 generator 处理过，这里做双重确认或处理)
                    mb_norm_cons = mb_norm_cons.to(self.device)
                    mb_norm_edge_idx = mb_norm_edge_idx.to(self.device)
                    mb_norm_edge_attr = mb_norm_edge_attr.to(self.device)
                    mb_norm_var = mb_norm_var.to(self.device)
                    mb_norm_bounds = mb_norm_bounds.to(self.device)
                    # 注意：GNN Policy 内部需要这两个 batch 索引来进行全局平均池化
                    mb_cons_batch = mb_cons_batch.to(self.device)
                    mb_var_batch = mb_var_batch.to(self.device)


                    mb_advantages = advantages[batch_indices]
                    mb_returns = returns[batch_indices]
                    mb_padding_masks = mb_cands_masks

                    

                                        
                    # action_probs = self.actor_network(mb_cands_states, mb_node_states, mb_mip_states,mb_norm_cons, mb_norm_edge_idx, mb_norm_edge_attr, mb_norm_var, mb_norm_bounds, mb_padding_masks,mb_cons_batch,mb_var_batch)
                    # 将 610 行改为：
                    action_probs = self.actor_network(
                        cands_state_mat=mb_cands_states, 
                        node_state=mb_node_states, 
                        mip_state=mb_mip_states,
                        norm_cons=mb_norm_cons, 
                        norm_edge_idx=mb_norm_edge_idx, 
                        norm_edge_attr=mb_norm_edge_attr, 
                        norm_var=mb_norm_var, 
                        norm_bounds=mb_norm_bounds, 
                        padding_mask=mb_padding_masks,
                        mb_cons_batch=mb_cons_batch, 
                        mb_var_batch=mb_var_batch
                    )
                    # ==========================================
                    # ⬇️ 插入：概率饱和度验证代码 ⬇️
                    # ==========================================
                    with torch.no_grad():
                        _max_p = action_probs.max().item()
                        _min_p = action_probs.min().item()
                        
                        if iter % 5 == 0:  # 限制打印频率，每 5 个 Batch 看一次
                            print(f"DEBUG [Probs] Batch {iter} | Max Prob: {_max_p:.6f} | Min Prob: {_min_p:.6f}")
                            if _max_p > 0.99:
                                print("🚨 警告：出现极端概率！模型处于极度自信（饱和）状态，必然导致梯度消失！")
                    # ==========================================
                    # ⬆️ 插入结束 ⬆️
                    # ==========================================

                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * mb_advantages
                    # === 诊断块 2：检查 Ratio 和裁剪 ===
                    with torch.no_grad():
                        ratios = torch.exp(new_log_probs - mb_old_log_probs)
                        clipped_mask = (ratios > (1 + self.policy_clip)) | (ratios < (1 - self.policy_clip))
                        num_clipped = clipped_mask.sum().item()
                        print(f"DEBUG - Batch Ratio Mean: {ratios.mean().item():.6f}, Clipped: {num_clipped}/{mb_actions.size(0)}")
                    # =================================
                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss -= self.entropy_weight * entropy.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # === 诊断块 3：检查 Actor 梯度 ===
                    # 注意：clip_grad_norm_ 会返回裁剪前的总梯度范数
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
                    if iter % 5 == 0: # 没必要每步都打，每10个 batch 打一次
                        print(f"DEBUG - Actor Grad Norm (Original): {actor_grad_norm.item():.8f}")
                        print(f"DEBUG - Batch {iter} | Actor Loss: {actor_loss.item():.8f}")
                    # =================================
                    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)

                    before_sum = sum(p.sum().item() for p in self.actor_network.parameters())
                    self.actor_optimizer.step()

                    after_sum = sum(p.sum().item() for p in self.actor_network.parameters())
                    if iter % 5 == 0:
                        print(f"DEBUG - Batch {iter} | Weight Sum Diff: {abs(after_sum - before_sum):.8f}")

                  

                    values = self.critic_network(mb_cands_states, mb_node_states, mb_mip_states,mb_norm_cons, mb_norm_edge_idx, mb_norm_edge_attr, mb_norm_var, mb_norm_bounds, mb_padding_masks,mb_cons_batch,mb_var_batch)
                    values = values.view(-1)
                    mb_returns = mb_returns.view(-1)
                    critic_loss = F.mse_loss(values, mb_returns)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
                    self.critic_optimizer.step()

                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

                    approx_kl = 0.0
                    if num_updates > 0:
                        with torch.no_grad():
                            last_action_probs = self.actor_network(mb_cands_states, mb_node_states, mb_mip_states, mb_norm_cons, mb_norm_edge_idx, 
                                                                mb_norm_edge_attr, mb_norm_var, mb_norm_bounds,mb_padding_masks,mb_cons_batch=mb_cons_batch,  mb_var_batch=mb_var_batch)
                            last_dist = Categorical(last_action_probs)
                            last_log_probs = last_dist.log_prob(mb_actions)
                            # approx_kl = (mb_old_log_probs - last_log_probs).mean().item()
                            log_ratio = last_log_probs - mb_old_log_probs
                            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()


                    # 1. 及时删除损失和概率张量
                    del action_probs, dist, new_log_probs, ratio, actor_loss, critic_loss, values
                    del mb_norm_cons, mb_norm_edge_idx, mb_norm_edge_attr, mb_norm_var, mb_norm_bounds
                    del mb_cons_batch, mb_var_batch
                    del mb_cands_states, mb_cands_masks, mb_mip_states, mb_node_states
                    del surr1, surr2, entropy, mb_advantages, mb_returns
                    # 释放当前 mini-batch 产生的显存碎片
                    torch.cuda.empty_cache()

            # Clear memory after updates
           
            self.update_counter += 1

            advantage_std = advantages.std().item()

            # approx_kl = 0.0
            # if num_updates > 0:
            #     with torch.no_grad():
            #         last_action_probs = self.actor_network(mb_cands_states, mb_node_states, mb_mip_states, mb_padding_masks)
            #         last_dist = Categorical(last_action_probs)
            #         last_log_probs = last_dist.log_prob(mb_actions)
            #         approx_kl = (mb_old_log_probs - last_log_probs).mean().item()

            episode_return = rewards.sum().item()
            print(f"4444444444444444444444")
            metrics = {
                'update_step': self.update_counter,
                'actor_loss': total_actor_loss / max(num_updates, 1),
                'critic_loss': total_critic_loss / max(num_updates, 1),
                'entropy': total_entropy / max(num_updates, 1),
                'advantage_std': advantage_std,
                'approx_kl_divergence': approx_kl,
                'episode_return': episode_return,
                'num_updates': num_updates,
            }

            # 1. 显式删除占用巨大的张量对象
            # del actions, rewards, dones, old_values, old_log_probs, advantages, returns
            # del mb_cands_states, mb_cands_masks, mb_mip_states, mb_node_states, action_probs
            # del actions, rewards, dones, old_values, old_log_probs, advantages, returns

            # 2. 强制回收 Python 垃圾
            self.memory.clear()
            
            gc.collect()

            # 3. 释放 PyTorch 预留但未使用的显存
            
            torch.cuda.empty_cache()

            self.logger.info(
                f"PPO Update {self.update_counter} | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"KL Div: {metrics['approx_kl_divergence']:.6f} | "
                f"Return: {metrics['episode_return']:.2f}"
            )

            # save_metrics_to_csv(metrics, filepath="bi_output/training_metrics.csv")
            return metrics
        except Exception as e:
            self.logger.error(f"Error in learn: {e}")
            self.logger.error(traceback.format_exc())
            raise


    def save_models(self, path):
        try:
            torch.save({
                'actor_state_dict': self.actor_network.state_dict(),
                'critic_state_dict': self.critic_network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'update_counter': self.update_counter,
            }, f"{path}_checkpoint.pth")
            self.logger.info(f"Models saved to {path}_checkpoint.pth")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            self.logger.error(traceback.format_exc())

    def load_models(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.update_counter = checkpoint.get('update_counter', 0)
            self.logger.info(f"Models loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.logger.error(traceback.format_exc())