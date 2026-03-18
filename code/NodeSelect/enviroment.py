import pyscipopt as scip
import gc
import traceback
from project.utils import init_params
from project.utils.functions import SCIPStateExtractor

# 【注意这里的导入路径变成了你的新文件夹】
from NodeSelect.nodeselector import NodeSelector 
from NodeSelect.bi_graph import LPFeatureRecorder

class Environment:
    def __init__(self, device, agent, state_dims, scip_limits, scip_params, scip_seed, reward_func, logger):
        self.device = device
        self.agent = agent
        self.var_dim = state_dims["var_dim"]
        self.node_dim = state_dims["node_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.node_cand_dim = state_dims["node_cand_dim"] # 新增：候选节点特征维度
        self.state_dims = state_dims
        self.scip_limits = scip_limits
        self.scip_params = scip_params
        self.scip_seed = scip_seed
        self.reward_func = reward_func
        self.logger = logger

        self.model = None
        self.node_selector = None  # 原本是 self.brancher

        self.episode_count = 0
        self.total_selects = 0  # 原本是 self.total_branches
        self.total_nodes = 0
        self.cutoff = None

        self.baseline_nodes = None
        self.baseline_gap = None
        self.baseline_integral = None
        self.baseline_status = None

        self.recorder = None

    def _is_solved(self):
        status = self.model.getStatus()
        return status in ["optimal", "infeasible", "unbounded", "timelimit"]

    def reset(self, instance, cutoff=None, baseline_nodes=None, baseline_gap=None,
              baseline_integral=None, baseline_status=None):
        try:
            if self.model is not None:
                try:
                    self.model.freeProb()
                    self.logger.info("Previous SCIP model freed successfully")
                except Exception as e:
                    self.logger.error(f"Failed to free previous SCIP model: {e}")
                self.model = None
                gc.collect()

            self.model = scip.Model()
            init_params(self.model, self.scip_limits, self.scip_params)

            self.model.setBoolParam('randomization/permutevars', True)
            self.model.setIntParam('randomization/permutationseed', int(self.scip_seed))

            self.model.readProblem(instance)
            self.cutoff = cutoff
            self.baseline_nodes = baseline_nodes
            self.baseline_gap = baseline_gap
            self.baseline_integral = baseline_integral
            self.baseline_status = baseline_status

            bn = float(baseline_nodes) if baseline_nodes is not None else 1.0
            bg = float(baseline_gap) if baseline_gap is not None else 0.0
            bp = float(baseline_integral) if baseline_integral is not None else 1.0
            bs = baseline_status if baseline_status is not None else 'timelimit'

            self.reward_func.reset(baseline_nodes=bn, baseline_gap=bg, baseline_pdi=bp,
                                   solver_status=bs, time_limit=900, logger=self.logger)

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
            
            # 1. 实例化新写的 NodeSelector
            self.node_selector = NodeSelector(
                model=self.model,
                state_dims=self.state_dims,
                device=self.device,
                agent=self.agent,
                reward_func=self.reward_func,
                cutoff=self.cutoff,
                logger=self.logger,
                recorder=self.recorder,
            )

            # 2. 挂载到 SCIP，注意使用 includeNodesel
            self.model.includeNodesel(
                nodesel=self.node_selector,
                name="TreeGatePPO_NodeSelector",
                desc="Tree-Gate PPO Training Node Selection Rule",
                stdpriority=999999,
                memsavepriority=999999
            )

            self.logger.info(f"Starting episode {self.episode_count} with Node Selection")
            
            # 3. 触发求解
            self.model.optimize() 

            # 4. 求解结束后，手动调用收尾方法记录最后一条 Transition
            self.logger.info("Calling node selector finalize_episode")
            self.node_selector.finalize_episode()

            status = self.model.getStatus()
            done = self._is_solved()

            episode_stats = self.node_selector.get_episode_stats()
            episode_reward = episode_stats['total_reward']
            gap_val = 0.0 if done else self.model.getGap()

            info = {
                "status": status,
                "objective": self.model.getObjVal() if done and status == "optimal" else None,
                "select_count": self.node_selector.select_count,
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

            self.total_selects += self.node_selector.select_count
            self.total_nodes += info['nnodes']

            # 清理资源
            self.model.freeProb()
            self.recorder.clear()
            self.model = None
            self.node_selector = None
            self.recorder = None

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
            self.node_selector = None
            raise