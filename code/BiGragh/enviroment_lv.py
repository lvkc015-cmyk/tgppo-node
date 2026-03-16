import pyscipopt as scip
from project.utils import init_params
from BiGragh.bracher_lv import Brancher
import numpy as np
import torch as T
import traceback

from BiGragh.recorders import LPFeatureRecorder

class Environment:
    def __init__(self, device, agent, state_dims, scip_limits, scip_params, scip_seed, reward_func, logger):
        self.device = device
        self.agent = agent
        self.var_dim = state_dims["var_dim"]
        self.node_dim = state_dims["node_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.state_dims = state_dims
        self.scip_limits = scip_limits
        self.scip_params = scip_params
        self.scip_seed = scip_seed
        self.reward_func = reward_func
        self.logger = logger

        self.model = None
        self.brancher = None

        self.episode_count = 0
        self.total_branches = 0
        self.total_nodes = 0
        self.cutoff = None

        self.baseline_nodes = None
        self.baseline_gap = None
        self.baseline_integral = None
        self.baseline_status = None

        self.recorder = None

    #求解完成判断
    def _is_solved(self):
        status = self.model.getStatus()
        return status in ["optimal", "infeasible", "unbounded", "timelimit"]

    # 一次 episode 的初始化
    def reset(self, instance, cutoff=None, baseline_nodes=None, baseline_gap=None,
              baseline_integral=None, baseline_status=None):
        try:
            #释放上一次 SCIP
            if self.model is not None:
                try:
                    self.model.freeProb()
                    self.logger.info("Previous SCIP model freed successfully")
                except Exception as e:
                    self.logger.error(f"Failed to free previous SCIP model: {e}")
                self.model = None
                gc.collect()

            #创建新 SCIP Model
            self.model = scip.Model()
            init_params(self.model, self.scip_limits, self.scip_params)

            #设置随机化（用于泛化）
            self.model.setBoolParam('randomization/permutevars', True)
            self.model.setIntParam('randomization/permutationseed', int(self.scip_seed))

            #读取 MILP 实例
            self.model.readProblem(instance)
            self.cutoff = cutoff
            self.baseline_nodes = baseline_nodes
            self.baseline_gap = baseline_gap
            self.baseline_integral = baseline_integral
            self.baseline_status = baseline_status

            # ensure reward function reset is robust (handles None)
            bn = float(baseline_nodes) if baseline_nodes is not None else 1.0
            bg = float(baseline_gap) if baseline_gap is not None else 0.0
            bp = float(baseline_integral) if baseline_integral is not None else 1.0
            bs = baseline_status if baseline_status is not None else 'timelimit'

            #奖励函数参数设置
            # H4
            # self.reward_func.reset(self.model, time_limit=3600)
           

            self.reward_func.reset(baseline_nodes=bn, baseline_gap=bg, baseline_pdi=bp,
                                   solver_status=bs, time_limit=900, logger=self.logger)
            # self.reward_func.reset(baseline_nodes=bn, baseline_gap=bg, baseline_pdi=bp,
            # solver_status=bs, time_limit=3600, logger=None)
                                   

            if self.scip_params.get('cutoff', False) and self.cutoff is not None:
                self.model.setObjlimit(float(self.cutoff))

            self.logger.info(f"Environment reset for instance: {instance}")
            return True
        except Exception as e:
            self.logger.error(f"Error in environment reset: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def run_episode(self):
        try:
            self.episode_count += 1

            self.recorder = LPFeatureRecorder(self.model, self.device)
            
            # 提取特征
            self.brancher = Brancher(
                model=self.model,
                state_dims=self.state_dims,
                device=self.device,
                agent=self.agent,
                reward_func=self.reward_func,
                cutoff=self.cutoff,
                logger=self.logger,
                recorder=self.recorder,  # <-- 新增：传递引用
            )

            self.model.includeBranchrule(
                branchrule=self.brancher,
                name="TreeGatePPO_Brancher",
                desc="Tree-Gate PPO Training Branching Rule",
                priority=999999,
                maxdepth=-1,
                maxbounddist=1,
            )

            self.logger.info(f"Starting episode {self.episode_count}")
            self.model.optimize() # -->调用 brancher.branchexeclp()

            # Ensure final transition stored
            self.logger.info("Calling branchfree")
            self.brancher.branchfree()

            status = self.model.getStatus()
            done = self._is_solved()

            episode_stats = self.brancher.get_episode_stats()
            episode_reward = episode_stats['total_reward']
            reward_list = episode_stats['episode_rewards']

            gap_val = 0.0 if done else self.model.getGap()

            # self.logger.info(
            #     f"Episode {self.episode_count} completed - "
            #     f"Status: {status}, "
            #     f"Branches: {self.brancher.branch_count}, "
            #     f"Nodes: {self.model.getNNodes()}, "
            #     f"Total Reward: {episode_reward:.4f}, "
            #     f"Reward Count: {len(reward_list)}, "
            #     f"Gap: {gap_val:.4%}"
            # )

            self.brancher.episode_rewards = []

            info = {
                "status": status,
                "objective": self.model.getObjVal() if done and status == "optimal" else None,
                "branch_count": self.brancher.branch_count,
                "gap": gap_val,
                "primal_bound": self.model.getPrimalbound(),
                "dual_bound": self.model.getDualbound(),
                "primalDualIntegral": self.model.getPrimalDualIntegral(),
                "scip_solve_time": self.model.getSolvingTime(),
                "max_depth": self.model.getMaxDepth(),
                "nnodes": self.model.getNNodes(),
                "episode": self.episode_count,
                "total_reward": episode_reward,
            }

            self.total_branches += self.brancher.branch_count
            self.total_nodes += info['nnodes']

            self.model.freeProb()
            self.recorder.clear()
            self.model = None
            self.brancher = None
            self.recorder = None # 彻底断开引用

            return done, info, episode_reward
        except Exception as e:
            self.logger.error(f"Error in run_episode: {e}")
            self.logger.error(traceback.format_exc())
            if self.model is not None:
                try:
                    self.model.freeProb()
                except Exception:
                    pass
                self.model = None
            self.brancher = None
            raise

    def get_memory_size(self):
        return len(self.agent.memory) if self.agent is not None else 0

    def should_update(self, min_memory_size=None):
        if self.agent is None:
            return False
        min_size = min_memory_size or self.agent.batch_size
        return len(self.agent.memory) >= min_size

    def update_agent(self):
        if self.should_update():
            return self.agent.learn()
        return None

    def get_stats(self):
        return {
            'episode_count': self.episode_count,
            'total_branches': self.total_branches,
            'total_nodes': self.total_nodes,
            'avg_branches_per_episode': self.total_branches / max(self.episode_count, 1),
            'avg_nodes_per_episode': self.total_nodes / max(self.episode_count, 1),
            'memory_size': self.get_memory_size(),
        }

