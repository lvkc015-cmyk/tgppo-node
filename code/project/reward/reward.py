import math
import numpy as np


def _safe_tanh(x, s=1.0):
    return float(np.tanh(s * x))


def _ratio(num, den, cap=None):
    den = max(float(den), 1e-12)
    val = float(num) / den
    if cap is not None:
        return min(val, cap)
    return val


class RewardH1:
    """Baseline-normalized node efficiency + terminal bonus.

    Simple, scale-robust: penalizes nodes per step normalized by baseline nodes
    and bonuses at terminal based on speedup vs. baseline and problem status.
    """

    def __init__(self, logger=None, alpha=1.0, bonus_cap=3.0):
        self.logger = logger
        self.alpha = alpha
        self.bonus_cap = bonus_cap
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        dn = max(nodes - self.prev_nodes, 0)
        self.prev_nodes = nodes

        # Step penalty normalized by baseline nodes; bounded by tanh
        step_penalty = - _safe_tanh(dn / (self.B * 0.02 + 1.0), s=self.alpha)  # ~2% of baseline gives ~0.76 penalty
        if not done:
            if self.logger:
                self.logger.info(f"H1 step: nodes={nodes}, dn={dn}, penalty={step_penalty:.4f}")
            return step_penalty

        # Terminal bonus
        status = model.getStatus()
        gap = float(model.getGap())
        pdi = float(model.getPrimalDualIntegral())

        speedup = _ratio(self.B, max(nodes, 1), cap=self.bonus_cap)  # >1 if better than baseline
        if status == "optimal":
            bonus = 1.0 + 2.0 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.5 + 1.5 * speedup
        elif status == "timelimit":
            # progress vs baseline
            gap_gain = _safe_tanh((self.baseline_gap - gap))
            pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
            bonus = 0.2 * speedup + 0.6 * gap_gain + 0.2 * pdi_gain
        else:
            bonus = 0.2 * speedup

        if self.logger:
            self.logger.info(f"H1 terminal: status={status}, nodes={nodes}, speedup={speedup:.3f}, bonus={bonus:.4f}")
        return float(bonus)


class RewardH2:
    """Log-scaled node efficiency + progress shaping.

    Adds: (1) log scaling to further damp huge baselines, (2) pace term comparing
    current nodes to a power-law expected curve, (3) gap/PDI improvement shaping.
    """

    def __init__(self, logger=None, scale=1.5, pace_power=0.7, w_nodes=0.5, w_pace=0.2, w_gap=0.2, w_pdi=0.1):
        self.logger = logger
        self.scale = scale
        self.pace_power = pace_power
        self.w_nodes = w_nodes
        self.w_pace = w_pace
        self.w_gap = w_gap
        self.w_pdi = w_pdi
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0

    def _node_efficiency(self, nodes):
        # 1 - log(nodes+1)/log(B+1) in [-inf,1]; map via tanh
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    def _pace_term(self, nodes, t):
        expected = self.B * (t ** self.pace_power)
        # positive when under expected nodes
        pace = (expected - nodes) / (expected + 1.0)
        return _safe_tanh(pace, s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        t = float(model.getSolvingTime()) / self.time_limit
        t = min(max(t, 0.0), 1.0)

        gap = float(model.getGap())
        if math.isinf(gap):
            gap = 1e6
        pdi = float(model.getPrimalDualIntegral())

        # Components
        r_nodes = self._node_efficiency(nodes)
        r_pace = self._pace_term(nodes, t)
        gap_impr = _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9)) if self.prev_gap < float('inf') else 0.0
        pdi_impr = _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi) if self.prev_pdi > 0 else 0.0

        self.prev_nodes = nodes
        self.prev_gap = gap
        self.prev_pdi = pdi

        if not done:
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * gap_impr + self.w_pdi * pdi_impr)
            reward = float(np.clip(reward, -1.0, 1.0))
            if self.logger:
                self.logger.info(f"H2 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, gap_impr={gap_impr:.3f}, pdi_impr={pdi_impr:.3f}, R={reward:.3f}")
            return reward

        # Terminal bonus: blend speedup with final quality
        status = model.getStatus()
        # 衡量搜索的节点相对baseline 用的节点数的比例, 用更少节点 → speedup > 1 → 好
        speedup = _ratio(self.B, max(nodes, 1), cap=4.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)

        if status == "optimal":
            bonus = 1.0 + 2.5 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.7 + 2.0 * speedup
        else:  # timelimit or other
            bonus = 0.4 * speedup + 0.4 * gap_gain + 0.2 * pdi_gain
        if self.logger:
            self.logger.info(f"H2 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(bonus)


class RewardH3:
    """Adaptive difficulty-aware reward.

    Uses a smooth mapping from baseline difficulty (log baseline nodes) to
    weights over components (nodes efficiency, progress, gap, PDI, time pace).
    Encourages anytime improvement on hard instances while still rewarding
    beating baseline on easy ones.
    """

    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale = scale
        self.reset(1.0, 0.0, 1.0, "timelimit", 900.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=900.0, logger=None):
        # 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 900.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # Difficulty weight schedule via sigmoid over log baseline nodes
        # easy -> emphasize speedup; hard -> emphasize gap/PDI progress
        # map logB in [log(1), log(1e6)] roughly to [0,1]

        #归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)
        #权重 w_* 随难度调整
        #简单问题 (d≈0) → w_nodes 大，强调搜索效率。
        self.w_nodes = 0.55 * (1 - d) + 0.25 * d    # 0.55 -> 0.25
        #难问题 (d≈1) → w_gap 和 w_pdi 大，强调解质量
        self.w_gap = 0.10 * (1 - d) + 0.30 * d      # 0.10 -> 0.30
        self.w_pdi = 0.05 * (1 - d) + 0.20 * d      # 0.05 -> 0.20
        self.w_progress = 0.15 * (1 - d) + 0.15 * d # 0.15 -> 0.15
        self.w_pace = 0.15 * (1 - d) + 0.10 * d     # 0.15 -> 0.10
        #最后归一化使五个权重之和为 1
        s = self.w_nodes + self.w_gap + self.w_pdi + self.w_progress + self.w_pace
        self.w_nodes /= s; self.w_gap /= s; self.w_pdi /= s; self.w_progress /= s; self.w_pace /= s

    #节点效率奖励，节点数越少，eff 越大 → 奖励越高。
    def _node_eff(self, nodes):
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    #时间/进度奖励，难问题曲线更平滑，不强调早期加速；鼓励早期快找到解。
    def _pace(self, nodes, t):
        # piecewise expected curve: early aggressive, then conservative
        # exponent varies with difficulty; harder -> smaller exponent
        exponent = 0.9 - 0.5 * min(max(self.logB / math.log1p(1e6), 0.0), 1.0)
        expected = self.B * (t ** exponent)
        return _safe_tanh((expected - nodes) / (expected + 1.0), s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes()) #当前 B&B 已扩展的节点数,越小越好（搜索更高效）
        #已用时间 / 时间上限 [0,1],表示进度条
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        gap = float(model.getGap()) #当前最优解与下界之间的差距百分比
        if math.isinf(gap):
            gap = 1e6

        #pdi越小越好,  原对偶间隙随时间变化的曲线下方围成的面积
        #举例：两个求解器可能都在 100 秒时找到了最优解。但求解器 A 在第 10 秒就找到了一个非常接近最优的解，而求解器 B 直到第 90 秒才找到。
        #求解器 A 的 PDI 更小，说明它在实际应用中更优秀
        pdi = float(model.getPrimalDualIntegral())
        #当前最好可行解(上界)
        pb = model.getPrimalbound()
        #当前松弛下界
        db = model.getDualbound()

        r_nodes = self._node_eff(nodes) #节点效率奖励，节点数越少，r_nodes 越大
        r_pace = self._pace(nodes, tfrac) #求解速度奖励,相同时间探索更多节点 → 奖励更高
        r_progress = 0.0   #进展奖励

        # gap improvement Gap 改善奖励
        if self.prev_gap < float('inf'):
            #计算当前步与上一步相比，原对偶间隙（Gap）缩小的比例。
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=self.scale)
        # bound improvements  这是为了在 Gap 还没有整体大幅变动时，给智能体更细粒度的信号
        if self.prev_pb is not None and pb < self.prev_pb: #原问题上界（Primal Bound）下降
            r_progress += _safe_tanh((self.prev_pb - pb) / (abs(self.prev_pb) + 1e-9), s=self.scale)
        if self.prev_db is not None and db > self.prev_db: # 对偶下界（Dual Bound）上升
            r_progress += _safe_tanh((db - self.prev_db) / (abs(self.prev_db) + 1e-9), s=self.scale)
        # pdi decrease PDI 减小奖励
        if self.prev_pdi > 0: 
            r_progress += _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi, s=self.scale)

        # update prevs
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done: # 非终止时的奖励计算
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * ( - _safe_tanh(gap / (self.baseline_gap + 1e-9), s=0.5) ) +
                      self.w_pdi * _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi) +
                      self.w_progress * r_progress)
            reward = float(np.clip(reward, -1.0, 1.0))
            # if self.logger:
            #     self.logger.info(f"H3 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, r_prog={r_progress:.3f}, R={reward:.3f}")
            return reward

        # Terminal: blend speedup and quality with difficulty-aware weights
        status = model.getStatus() #求解状态
        ## 衡量搜索的节点相对baseline 用的节点数的比例, 用更少节点 → speedup > 1 → 好
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        if status == "optimal": #求解结束
            bonus = 1.0 + 3.0 * speedup
        elif status in ("infeasible", "unbounded"): #没找到可行解
            bonus = 0.8 + 2.0 * speedup
        elif status == "timelimit":  #达到截止时间
            bonus = 0.5 * speedup + 0.3 * gap_gain + 0.2 * pdi_gain
        else: #其他情况
            bonus = 0.3 * speedup
        # if self.logger:
        #     self.logger.info(f"H3 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(bonus)



#------------------------  工业进化版 ------------------------------------
class RewardH4:
    """
    Industrial Adaptive Reward (Zero-Baseline).
    
    1. 移除 baseline_nodes, baseline_gap, baseline_pdi。
    2. 使用算例的静态规模 (NZ) 预估难度权重。
    3. 使用根节点初始状态 (First Gap) 作为改善基准。
    4. 引入动态时间衰减，平衡早期发现解与后期收敛。
    """

    def __init__(self, logger=None, scale=1.0):
        self.logger = logger
        self.scale = scale
        # 初始默认重置
        self._internal_reset(1000, 1.0, 3600.0)

    def reset(self, model, time_limit=3600.0):
        """
        直接从 SCIP 模型中提取初始状态进行重置。
        无需任何外部 pickle 字典。
        """
        # 1. 获取静态规模：非零元数量 (NZ) 所有约束矩阵里“非零系数”的总个数
        nz = model.getNNonzeros()
        
        # 2. 获取初始解状态 (根节点 Gap)
        # 如果根节点还没解，SCIP 会返回 inf，我们处理为 1.0
        first_gap = model.getGap()
        if math.isinf(first_gap) or first_gap > 1e6:
            first_gap = 1.0
            
        self._internal_reset(nz, first_gap, time_limit)

    def _internal_reset(self, nz, first_gap, time_limit):
        # 1. 难度指数 d：基于 NZ 规模自感知
        # 映射 NZ [1e3, 1e7] -> d [0, 1]
        self.d = min(max((math.log10(nz + 1) - 3.0) / 4.0, 0.0), 1.0)
        self.nz = nz
        
        # 2. 初始状态记录
        self.first_gap = max(first_gap, 1e-6)
        self.time_limit = max(time_limit, 1.0)
        
        # 3. 权重分配 (逻辑继承 H3，但 d 是自感知的)
        self.w_nodes = 0.55 * (1 - self.d) + 0.25 * self.d
        self.w_gap   = 0.10 * (1 - self.d) + 0.30 * self.d
        self.w_pdi   = 0.05 * (1 - self.d) + 0.20 * self.d
        self.w_progress = 0.15
        self.w_pace     = 0.15 * (1 - self.d) + 0.10 * self.d
        
        # 归一化权重
        s = self.w_nodes + self.w_gap + self.w_pdi + self.w_progress + self.w_pace
        self.w_nodes /= s; self.w_gap /= s; self.w_pdi /= s
        self.w_progress /= s; self.w_pace /= s

        # 4. 实时状态追踪
        self.prev_nodes = 0
        self.prev_gap = self.first_gap
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

    def _node_penalty(self, nodes, tfrac):
      
        # log10(100)=2;   nz_scale最小为2
        # nz:小规模 100 到 5,000 之间; 中规模:10,000 到 200,000;  大规模: 可能达到 1,00,0000（百万级）
        nz_scale = math.log10(max(self.nz, 100)) # 2-6之间
        # 基础容量
        base_capacity = 500 * (nz_scale - 1) # 500-2500之间
        
        # 当前预算:随着时间进度 tfrac 的增加（从 0 到 1），预算会线性放大。到结束时，预算会变成基础容量的 11 倍
        current_budget = base_capacity * (1.0 + 10.0 * tfrac)
        
        #用当前的节点数除以预算。如果比例大于 1.0，说明节点数已经超标  excess_ratio越小越好
        excess_ratio = nodes / max(current_budget, 1.0)
        
        # 问题越难:d=1,stiffness=o.25 ;  问题越简单:d=0,stiffness=0.5
        stiffness = 0.5 * (1.0 - self.d * 0.5) 
        penalty = - _safe_tanh(excess_ratio * stiffness, s=0.5)
        
        return penalty

    def compute(self, model, done):

        nodes = int(model.getNNodes()) #节点总数
        #已用时间 / 时间上限 [0,1],表示进度条
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        gap = float(model.getGap())
        if math.isinf(gap): gap = 1.0
        
        pdi = float(model.getPrimalDualIntegral())
        pb = model.getPrimalbound()
        db = model.getDualbound()

        # 1. 各项组件计算
        r_nodes = self._node_penalty(nodes, tfrac)
        
        # Progress: 奖励相对于自身初始 Gap 的下降量
        r_progress = 0.0
        if self.prev_gap < float('inf'):
            # 相对于起始 Gap 的边际改善
            r_progress += _safe_tanh((self.prev_gap - gap) / self.first_gap, s=self.scale)
        
        # Bound 改善（细粒度信号）
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += 0.5 * _safe_tanh((self.prev_pb - pb) / (abs(self.prev_pb) + 1e-9))
        if self.prev_db is not None and db > self.prev_db:
            r_progress += 0.5 * _safe_tanh((db - self.prev_db) / (abs(self.prev_db) + 1e-9))

        # 2. 组合奖励 (Step Reward)
        if not done:
            # 使用 self.first_gap 替代 baseline_gap
            reward = (
                self.w_nodes * r_nodes +
                self.w_gap   * (-_safe_tanh(gap / self.first_gap, s=0.5)) +
                self.w_pdi   * (-_safe_tanh(pdi / (self.first_gap * self.time_limit), s=0.5)) +
                self.w_progress * r_progress
            )
            reward = float(np.clip(reward, -1.0, 1.0))
            return reward

        # 3. 终止奖励 (Terminal Bonus)
        # 工业版 Terminal 不再看 speedup，而是看绝对质量和时间节省
        status = model.getStatus()
        time_saved_ratio = 1.0 - tfrac
        gap_reduced_ratio = max(0, (self.first_gap - gap) / self.first_gap)
        
        if status == "optimal":
            bonus = 2.0 + 2.0 * time_saved_ratio
        elif status == "timelimit": # 不合理
            bonus = 1.0 * gap_reduced_ratio + 0.5 * time_saved_ratio
        else:
            bonus = 0.5 * gap_reduced_ratio
            
        return float(bonus)