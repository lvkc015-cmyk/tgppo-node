import math
import numpy as np

#值域严格在 -1 和 1 之间
def _safe_tanh(x, s=1.0):
    return math.tanh(s * x)

# 计算两个数的比值
def _ratio(num, den, cap=None):
    den = max(float(den), 1e-12)
    val = float(num) / den
    if cap is not None:
        return min(val, cap)
    return val


def _clip_unit_interval(x):
    return min(max(float(x), 0.0), 1.0)


def _terminal_bonus(status, speedup, pdi_gain, gap):
    gap_penalty = _safe_tanh(gap, s=0.5)

    if status == "optimal":
        return 1.0 + 3.0 * speedup + 1.0 * pdi_gain
    if status in ("infeasible", "unbounded"):
        return 0.8 + 2.0 * speedup + 1.0 * pdi_gain
    if status in ("timelimit", "nodelimit"):
        return 0.1 * speedup + 0.25 * pdi_gain - 0.5 * gap_penalty
    return -0.25 * gap_penalty

# ### 适合setcover的奖励函数
class RewardNodeSelection:
    """
    公式: R = w1 * r_progress + w2 * r_promise - w3 * r_switch
    """
    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale_range = (1.5, 5.0)  # 简单问题用 1.5，极难问题放大到 5.0
        self.depth_boost = 2.0 # 随搜索深度额外放大的倍数上限
        self.current_scale = scale

        # 引入绝对生存税，只要走一步就必须扣分
        self.base_step_penalty = -0.05
        
        # --- 核心权重分配 (初始默认值，会在 reset 中被覆盖) ---
        self.w1 = 1.0  # 进步权重 (最重要，一旦突破给予重奖)
        self.w2 = 0.2  # 潜力权重 (稠密奖励，引导 AI 每一步的选择)
        self.w3 = 0.3  # 跳跃惩罚 (引擎保护器，防止频繁冷启动)

        self.alpha = 2.0  # PB 突破的倍率
        self.beta = 1.5   # DB 突破的倍率
        self.gamma = 0.5  # 估计值 (Estimate) 相对于下界的权重

        # --- 新增：停滞监测参数 ---
        self.stagnation_counter = 0  # 连续无进展计数器
        #self.base_time_penalty = -0.01  # 基础每步惩罚（极小，防止抵消正常奖励）
        self.penalty_step = 0.02  # 停滞累积因子
        self.penalty_floor = -0.5      # 惩罚下限：无论停滞多久，单步惩罚不低于此值

        self.floor_range = (-0.4, -0.1)  # 简单问题扣分狠，难题扣分轻
        self.step_range = (0.01, 0.002)  # 简单问题惩罚累积快，难题累积慢
        
        self.reset(1.0, 0.0, 1.0, "timelimit", 300.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=300.0, logger=None):
        ## 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        
        # 【修复】：必须在这里计算 logB，否则下面计算 d 的时候会报错
        self.logB = math.log1p(self.B)
        
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)

        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 300.0), 1.0)
        self.logger = logger or self.logger
        
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # 用于暂存上一步选择的节点质量和跳跃代价
        self.last_r_promise = 0.0
        self.last_r_switch = 0.0

        self.stagnation_counter = 0 # 重置计数器

        ##归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)

        # 难题更需要“显微镜”去放大微小的进展
        #self.current_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 2. 初始 Scale (根据问题难度锚定起点)
        # 难题起点更高 (e.g., d=1 -> scale=4.0; d=0 -> scale=1.5)
        self.initial_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 难题 (d=1): floor=-0.1, step=0.002 (温柔)
        # 简单 (d=0): floor=-0.4, step=0.01 (严厉)
        self.penalty_floor = self.floor_range[0] * (1 - d) + self.floor_range[1] * d
        self.penalty_step = self.step_range[0] * (1 - d) + self.step_range[1] * d
        
        # ==========================================
        # 动态权重引擎 (Dynamic Weights)
        # ==========================================
        # w1 (全局进步): 永远是北极星，锚定在 1.0 附近，越难的问题越依赖它
        self.w1 = 0.8 * (1 - d) + 1.2 * d   # 0.8 -> 1.2
        
        # w2 (稠密引导): 简单问题大胆贪心，难问题防陷阱
        self.w2 = 0.4 * (1 - d) + 0.1 * d   # 0.4 -> 0.1
        
        # w3 (跳跃惩罚): 简单问题跳跃免费，难问题跳跃要命
        self.w3 = 0.1 * (1 - d) + 0.6 * d   # 0.1 -> 0.6

        s = self.w1 + self.w2 + self.w3
        self.w1 /= s
        self.w2 /= s
        self.w3 /= s

        # ======== 探针 1 ========
        # if self.logger:
        #     print(f"\n[探针1-Reset] 实例基线节点 B: {self.B:.1f} | 算出的难度 d: {d:.3f}")
        #     print(f"[探针1-Reset] 归一化权重 -> w1(进步): {self.w1:.3f}, w2(潜力): {self.w2:.3f}, w3(跳跃): {self.w3:.3f}")
        # ========================


    
    def set_action_feedback(self, norm_lb, norm_est, switch_penalty):
        norm_lb = _clip_unit_interval(norm_lb)
        norm_est = _clip_unit_interval(norm_est)

        # 最完美的结点给出 0 分，越差的结点扣分越多。绝对不给正分！
        promise_penalty = (norm_lb + self.gamma * norm_est) / (1.0 + self.gamma) 
        
        # 加个负号，让它变成惩罚项
        self.last_r_promise = -promise_penalty  # 范围在 [-1.0, 0.0]
        self.last_r_switch = switch_penalty


    def compute(self, model, done):

        # nodes_ratio 越大，说明搜得越深。利用 log 让增长在前期快，后期稳。
        current_nodes = max(float(model.getNNodes()), 1.0)
        # phi 会从 0 慢慢增长，在达到基线节点数 B 时约为 1.0
        
        phi = min(math.log1p(current_nodes) / max(math.log1p(self.B), 2.3), 1.5) # 2.3 约等于 log(10)
        
        # --- 实时动态 Scale ---
        # 随搜索深度线性增加 scale，确保后期微小进展也能被“显微镜”放大
        # 最终 scale = 初始 scale * (1 + 深度加成因子)
        active_scale = self.initial_scale * (1.0 + phi * 0.5) 
    
        gap = float(model.getGap())
        if math.isinf(gap): gap = 1e6

        #PDI 越小，说明它在实际应用中越优秀
        pdi = float(model.getPrimalDualIntegral())

        ##当前最好可行解(上界)
        pb = model.getPrimalbound()
        ##当前松弛下界
        db = model.getDualbound()

        r_progress = 0.0 #进展奖励

        has_real_progress = False

        # 1. 基础进步：Gap 缩小
        if self.prev_gap < float('inf') and gap < self.prev_gap:
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 核心突破：找到了更好的整数解 (PB 下降)
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += self.alpha * _safe_tanh((self.prev_pb - pb) / max(abs(self.prev_pb), 1e-9), s=active_scale)
            has_real_progress = True
        # 3. 核心突破：证明了子树，提升了全局下界 (DB 上升)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += self.beta * _safe_tanh((db - self.prev_db) / max(abs(self.prev_db), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 检测是否有实质性突破
        
        
        # 2. 停滞惩罚逻辑 (Stagnation Penalty)
        if has_real_progress:
            self.stagnation_counter = 0
            stagnation_penalty = 0.0
        else:
            self.stagnation_counter += 1
            # 计算线性累积惩罚：随着步数增加，扣分变多
            raw_penalty = - (self.stagnation_counter * self.penalty_step)
            # 使用 max 取两者中的较大值（因为是负数，所以起到了截断 Floor 的作用）
            stagnation_penalty = max(raw_penalty, self.penalty_floor)


        # 更新历史记录
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            # 只要没解完，每走一步底薪就是负的！
            base_step_penalty = -0.05
            # 代入你的宏伟公式
            reward = base_step_penalty + (self.w1 * r_progress) + (self.w2 * self.last_r_promise) - (self.w3 * self.last_r_switch)+ \
                        stagnation_penalty
            
            # ======== 探针 2 ========
            # print(f"[探针2-Step] 步数: {current_nodes} | 总奖励: {reward:.4f} (截断前)")
            # print(f" ┣━ 基础底薪: {base_step_penalty:.3f}")
            # print(f" ┣━ w1 * 进步(r_progress): {self.w1 * r_progress:.3f}")
            # print(f" ┣━ w2 * 潜力(last_r_promise): {self.w2 * self.last_r_promise:.3f}")
            # print(f" ┣━ -w3 * 跳跃(last_r_switch): {-self.w3 * self.last_r_switch:.3f}")
            # print(f" ┗━ 停滞惩罚: {stagnation_penalty:.3f}")
            # ========================

            return float(np.clip(reward, -5.0, 5.0))

        # 终止状态（Terminal）：保持一定的加速奖励
        nodes = int(model.getNNodes())
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        status = model.getStatus()
        
        bonus = _terminal_bonus(status, speedup, pdi_gain, gap)

        # ======== 探针 3 ========
        # print(f"\n[探针3-Done] Episode 结束！状态 (Status): {status}")
        # print(f"[探针3-Done] 探索节点数: {nodes} | Speedup: {speedup:.3f} | PDI Gain: {pdi_gain:.3f}")
        # print(f"[探针3-Done] 最终发放 Bonus: {bonus:.4f}\n")
        # ========================
            
        return float(bonus)


### 适合cauction的奖励函数,test2
class RewardNodeSelection2:
    """
    公式: R = w1 * r_progress + w2 * r_promise - w3 * r_switch
    """
    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale_range = (1.5, 5.0)  # 简单问题用 1.5，极难问题放大到 5.0
        self.depth_boost = 2.0 # 随搜索深度额外放大的倍数上限
        self.current_scale = scale

        # 引入绝对生存税，只要走一步就必须扣分
        self.base_step_penalty = -0.01
        
        # --- 核心权重分配 (初始默认值，会在 reset 中被覆盖) ---
        self.w1 = 1.0  # 进步权重 (最重要，一旦突破给予重奖)
        self.w2 = 0.2  # 潜力权重 (稠密奖励，引导 AI 每一步的选择)
        self.w3 = 0.3  # 跳跃惩罚 (引擎保护器，防止频繁冷启动)

        self.alpha = 2.0  # PB 突破的倍率
        self.beta = 1.5   # DB 突破的倍率
        self.gamma = 0.5  # 估计值 (Estimate) 相对于下界的权重

        # --- 新增：停滞监测参数 ---
        self.stagnation_counter = 0  # 连续无进展计数器
        #self.base_time_penalty = -0.01  # 基础每步惩罚（极小，防止抵消正常奖励）
        self.penalty_step = 0.02  # 停滞累积因子
        self.penalty_floor = -0.5      # 惩罚下限：无论停滞多久，单步惩罚不低于此值

        self.floor_range = (-0.4, -0.1)  # 简单问题扣分狠，难题扣分轻
        self.step_range = (0.01, 0.002)  # 简单问题惩罚累积快，难题累积慢
        
        self.reset(1.0, 0.0, 1.0, "timelimit", 300.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=300.0, logger=None):
        ## 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        
        # 【修复】：必须在这里计算 logB，否则下面计算 d 的时候会报错
        self.logB = math.log1p(self.B)
        
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)

        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 300.0), 1.0)
        self.logger = logger or self.logger
        
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # 用于暂存上一步选择的节点质量和跳跃代价
        self.last_r_promise = 0.0
        self.last_r_switch = 0.0

        self.stagnation_counter = 0 # 重置计数器

        ##归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)

        # 难题更需要“显微镜”去放大微小的进展
        #self.current_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 2. 初始 Scale (根据问题难度锚定起点)
        # 难题起点更高 (e.g., d=1 -> scale=4.0; d=0 -> scale=1.5)
        self.initial_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 难题 (d=1): floor=-0.1, step=0.002 (温柔)
        # 简单 (d=0): floor=-0.4, step=0.01 (严厉)
        self.penalty_floor = self.floor_range[0] * (1 - d) + self.floor_range[1] * d
        self.penalty_step = self.step_range[0] * (1 - d) + self.step_range[1] * d
        
        # ==========================================
        # 动态权重引擎 (Dynamic Weights)
        # ==========================================
        # w1 (全局进步): 永远是北极星，锚定在 1.0 附近，越难的问题越依赖它
        self.w1 = 0.8 * (1 - d) + 1.2 * d   # 0.8 -> 1.2
        
        # w2 (稠密引导): 简单问题大胆贪心，难问题防陷阱
        self.w2 = 0.3 * (1 - d) + 0.1 * d   # 0.3 -> 0.1
        
        # w3 (跳跃惩罚): 简单问题跳跃免费，难问题跳跃要命
        self.w3 = 0.1 * (1 - d) + 0.3 * d   # 0.1 -> 0.3

        s = self.w1 + self.w2 + self.w3
        self.w1 /= s
        self.w2 /= s
        self.w3 /= s

        # ======== 探针 1 ========
        # if self.logger:
        #     print(f"\n[探针1-Reset] 实例基线节点 B: {self.B:.1f} | 算出的难度 d: {d:.3f}")
        #     print(f"[探针1-Reset] 归一化权重 -> w1(进步): {self.w1:.3f}, w2(潜力): {self.w2:.3f}, w3(跳跃): {self.w3:.3f}")
        # ========================


    
    def set_action_feedback(self, norm_lb, norm_est, switch_penalty):
        norm_lb = _clip_unit_interval(norm_lb)
        norm_est = _clip_unit_interval(norm_est)

        # 最完美的结点给出 0 分，越差的结点扣分越多。绝对不给正分！
        promise_penalty = (norm_lb + self.gamma * norm_est) / (1.0 + self.gamma) 
        
        # 加个负号，让它变成惩罚项
        self.last_r_promise = -promise_penalty  # 范围在 [-1.0, 0.0]
        self.last_r_switch = switch_penalty


    def compute(self, model, done):

        # nodes_ratio 越大，说明搜得越深。利用 log 让增长在前期快，后期稳。
        current_nodes = max(float(model.getNNodes()), 1.0)
        # phi 会从 0 慢慢增长，在达到基线节点数 B 时约为 1.0
        
        phi = min(math.log1p(current_nodes) / max(math.log1p(self.B), 2.3), 1.5) # 2.3 约等于 log(10)
        
        # --- 实时动态 Scale ---
        # 随搜索深度线性增加 scale，确保后期微小进展也能被“显微镜”放大
        # 最终 scale = 初始 scale * (1 + 深度加成因子)
        active_scale = self.initial_scale * (1.0 + phi * 0.5) 
    
        gap = float(model.getGap())
        if math.isinf(gap): gap = 1e6

        #PDI 越小，说明它在实际应用中越优秀
        pdi = float(model.getPrimalDualIntegral())

        ##当前最好可行解(上界)
        pb = model.getPrimalbound()
        ##当前松弛下界
        db = model.getDualbound()

        r_progress = 0.0 #进展奖励

        has_real_progress = False

        # 1. 基础进步：Gap 缩小
        if self.prev_gap < float('inf') and gap < self.prev_gap:
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 核心突破：找到了更好的整数解 (PB 下降)
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += self.alpha * _safe_tanh((self.prev_pb - pb) / max(abs(self.prev_pb), 1e-9), s=active_scale)
            has_real_progress = True
        # 3. 核心突破：证明了子树，提升了全局下界 (DB 上升)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += self.beta * _safe_tanh((db - self.prev_db) / max(abs(self.prev_db), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 检测是否有实质性突破
        
        
        # 2. 停滞惩罚逻辑 (Stagnation Penalty)
        if has_real_progress:
            self.stagnation_counter = 0
            stagnation_penalty = 0.0
        else:
            self.stagnation_counter += 1
            # 计算线性累积惩罚：随着步数增加，扣分变多
            raw_penalty = - (self.stagnation_counter * self.penalty_step)
            # 使用 max 取两者中的较大值（因为是负数，所以起到了截断 Floor 的作用）
            stagnation_penalty = max(raw_penalty, self.penalty_floor)


        # 更新历史记录
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            # 只要没解完，每走一步底薪就是负的！
            base_step_penalty = -0.01
            # 代入你的宏伟公式
            reward = base_step_penalty + (self.w1 * r_progress) + (self.w2 * self.last_r_promise) - (self.w3 * self.last_r_switch)+ \
                        stagnation_penalty
            
            # ======== 探针 2 ========
            # print(f"[探针2-Step] 步数: {current_nodes} | 总奖励: {reward:.4f} (截断前)")
            # print(f" ┣━ 基础底薪: {base_step_penalty:.3f}")
            # print(f" ┣━ w1 * 进步(r_progress): {self.w1 * r_progress:.3f}")
            # print(f" ┣━ w2 * 潜力(last_r_promise): {self.w2 * self.last_r_promise:.3f}")
            # print(f" ┣━ -w3 * 跳跃(last_r_switch): {-self.w3 * self.last_r_switch:.3f}")
            # print(f" ┗━ 停滞惩罚: {stagnation_penalty:.3f}")
            # ========================

            return float(np.clip(reward, -5.0, 5.0))
        
            

        # 终止状态（Terminal）：保持一定的加速奖励
        nodes = int(model.getNNodes())
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        status = model.getStatus()
        
        bonus = _terminal_bonus(status, speedup, pdi_gain, gap)

        # ======== 探针 3 ========
        # print(f"\n[探针3-Done] Episode 结束！状态 (Status): {status}")
        # print(f"[探针3-Done] 探索节点数: {nodes} | Speedup: {speedup:.3f} | PDI Gain: {pdi_gain:.3f}")
        # print(f"[探针3-Done] 最终发放 Bonus: {bonus:.4f}\n")
        # ========================
            
        return float(bonus)


#适合cauction的奖励函数,对应 test3
class RewardNodeSelection3:
    """
    公式: R = w1 * r_progress + w2 * r_promise - w3 * r_switch
    """
    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale_range = (1.5, 5.0)  # 简单问题用 1.5，极难问题放大到 5.0
        self.depth_boost = 2.0 # 随搜索深度额外放大的倍数上限
        self.current_scale = scale

        # 引入绝对生存税，只要走一步就必须扣分
        self.base_step_penalty = -0.05
        
        # --- 核心权重分配 (初始默认值，会在 reset 中被覆盖) ---
        self.w1 = 1.0  # 进步权重 (最重要，一旦突破给予重奖)
        self.w2 = 0.2  # 潜力权重 (稠密奖励，引导 AI 每一步的选择)
        self.w3 = 0.3  # 跳跃惩罚 (引擎保护器，防止频繁冷启动)

        self.alpha = 2.0  # PB 突破的倍率
        self.beta = 1.5   # DB 突破的倍率
        self.gamma = 0.5  # 估计值 (Estimate) 相对于下界的权重

        # --- 新增：停滞监测参数 ---
        self.stagnation_counter = 0  # 连续无进展计数器
        #self.base_time_penalty = -0.01  # 基础每步惩罚（极小，防止抵消正常奖励）
        self.penalty_step = 0.02  # 停滞累积因子
        self.penalty_floor = -0.5      # 惩罚下限：无论停滞多久，单步惩罚不低于此值

        self.floor_range = (-0.4, -0.1)  # 简单问题扣分狠，难题扣分轻
        self.step_range = (0.01, 0.002)  # 简单问题惩罚累积快，难题累积慢
        
        self.reset(1.0, 0.0, 1.0, "timelimit", 400.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=400.0, logger=None):
        ## 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        
        # 【修复】：必须在这里计算 logB，否则下面计算 d 的时候会报错
        self.logB = math.log1p(self.B)
        
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)

        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 400.0), 1.0)
        self.logger = logger or self.logger
        
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # 用于暂存上一步选择的节点质量和跳跃代价
        self.last_r_promise = 0.0
        self.last_r_switch = 0.0

        self.stagnation_counter = 0 # 重置计数器

        ##归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)

        # 难题更需要“显微镜”去放大微小的进展
        #self.current_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 2. 初始 Scale (根据问题难度锚定起点)
        # 难题起点更高 (e.g., d=1 -> scale=4.0; d=0 -> scale=1.5)
        self.initial_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 难题 (d=1): floor=-0.1, step=0.002 (温柔)
        # 简单 (d=0): floor=-0.4, step=0.01 (严厉)
        self.penalty_floor = self.floor_range[0] * (1 - d) + self.floor_range[1] * d
        self.penalty_step = self.step_range[0] * (1 - d) + self.step_range[1] * d
        
        # ==========================================
        # 动态权重引擎 (Dynamic Weights)
        # ==========================================
        # w1 (全局进步): 永远是北极星，锚定在 1.0 附近，越难的问题越依赖它
        self.w1 = 0.8 * (1 - d) + 1.2 * d   # 0.8 -> 1.2
        
        # # 让难题的 w2 更大，简单问题反而不需要那么多引导。
        self.w2 = 0.1 * (1 - d) + 0.5 * d   # 0.1 -> 0.3
        
        # w3 (跳跃惩罚): 简单问题跳跃免费，难问题跳跃要命
        self.w3 = 0.1 * (1 - d) + 0.3 * d   # 0.3 -> 0.1

        s = self.w1 + self.w2 + self.w3
        self.w1 /= s
        self.w2 /= s
        self.w3 /= s

        # ======== 探针 1 ========
        # if self.logger:
        #     print(f"\n[探针1-Reset] 实例基线节点 B: {self.B:.1f} | 算出的难度 d: {d:.3f}")
        #     print(f"[探针1-Reset] 归一化权重 -> w1(进步): {self.w1:.3f}, w2(潜力): {self.w2:.3f}, w3(跳跃): {self.w3:.3f}")
        # ========================


    
    def set_action_feedback(self, norm_lb, norm_est, switch_penalty):
        norm_lb = _clip_unit_interval(norm_lb)
        norm_est = _clip_unit_interval(norm_est)

        # 最完美的结点给出 0 分，越差的结点扣分越多。绝对不给正分！
        promise_penalty = (norm_lb + self.gamma * norm_est) / (1.0 + self.gamma) 
        
        # 加个负号，让它变成惩罚项
        # self.last_r_promise = -promise_penalty  # 范围在 [-1.0, 0.0]
        self.last_r_promise = - _safe_tanh(promise_penalty, s=5.0)
        self.last_r_switch = switch_penalty


    def compute(self, model, done):

        # nodes_ratio 越大，说明搜得越深。利用 log 让增长在前期快，后期稳。
        current_nodes = max(float(model.getNNodes()), 1.0)
        # phi 会从 0 慢慢增长，在达到基线节点数 B 时约为 1.0
        
        phi = min(math.log1p(current_nodes) / max(math.log1p(self.B), 2.3), 1.5) # 2.3 约等于 log(10)
        
        # --- 实时动态 Scale ---
        # 随搜索深度线性增加 scale，确保后期微小进展也能被“显微镜”放大
        # 最终 scale = 初始 scale * (1 + 深度加成因子)
        active_scale = self.initial_scale * (1.0 + phi * 0.5) 
    
        gap = float(model.getGap())
        if math.isinf(gap): gap = 1e6

        #PDI 越小，说明它在实际应用中越优秀
        pdi = float(model.getPrimalDualIntegral())

        ##当前最好可行解(上界)
        pb = model.getPrimalbound()
        ##当前松弛下界
        db = model.getDualbound()

        r_progress = 0.0 #进展奖励

        has_real_progress = False

        # 1. 基础进步：Gap 缩小
        if self.prev_gap < float('inf') and gap < self.prev_gap:
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 核心突破：找到了更好的整数解 (PB 下降)
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += self.alpha * _safe_tanh((self.prev_pb - pb) / max(abs(self.prev_pb), 1e-9), s=active_scale)
            has_real_progress = True
        # 3. 核心突破：证明了子树，提升了全局下界 (DB 上升)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += self.beta * _safe_tanh((db - self.prev_db) / max(abs(self.prev_db), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 检测是否有实质性突破
        


        # 2. 停滞惩罚逻辑 (Stagnation Penalty)
        if has_real_progress:
            self.stagnation_counter = 0
            stagnation_penalty = 0.0
        else:
            self.stagnation_counter += 1
            # 计算线性累积惩罚：随着步数增加，扣分变多
            raw_penalty = - (self.stagnation_counter * self.penalty_step)
            # 使用 max 取两者中的较大值（因为是负数，所以起到了截断 Floor 的作用）
            stagnation_penalty = max(raw_penalty, self.penalty_floor)


        # 更新历史记录
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            # 只要没解完，每走一步底薪就是负的！
            base_step_penalty = -0.05
            # 代入你的宏伟公式
            reward = base_step_penalty + (self.w1 * r_progress) + (self.w2 * self.last_r_promise) - (self.w3 * self.last_r_switch)+ \
                        stagnation_penalty
            
            # ======== 探针 2 ========
            # print(f"[探针2-Step] 步数: {current_nodes} | 总奖励: {reward:.4f} (截断前)")
            # print(f" ┣━ 基础底薪: {base_step_penalty:.3f}")
            # print(f" ┣━ w1 * 进步(r_progress): {self.w1 * r_progress:.3f}")
            # print(f" ┣━ w2 * 潜力(last_r_promise): {self.w2 * self.last_r_promise:.3f}")
            # print(f" ┣━ -w3 * 跳跃(last_r_switch): {-self.w3 * self.last_r_switch:.3f}")
            # print(f" ┗━ 停滞惩罚: {stagnation_penalty:.3f}")
            # ========================

            return float(np.clip(reward, -5.0, 5.0))
            # return float(np.clip(reward, -0.2, 5.0))
        
            

        # 终止状态（Terminal）：保持一定的加速奖励
        nodes = int(model.getNNodes())
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        status = model.getStatus()
        
        # if status == "optimal": 
        #     bonus = 1.0 + 3.0 * speedup + 1.0 * pdi_gain
        # elif status in ("infeasible", "unbounded"): 
        #     # bonus = 0.0 + 2.0 * speedup + 1.0 * pdi_gain
        #     bonus = 1.0
        # else: 
        #     bonus = 0.0
            
        bonus = _terminal_bonus(status, speedup, pdi_gain, gap)

        # ======== 探针 3 ========
        # print(f"\n[探针3-Done] Episode 结束！状态 (Status): {status}")
        # print(f"[探针3-Done] 探索节点数: {nodes} | Speedup: {speedup:.3f} | PDI Gain: {pdi_gain:.3f}")
        # print(f"[探针3-Done] 最终发放 Bonus: {bonus:.4f}\n")
        # ========================
            
        return float(bonus)
    



class RewardNodeSelection4:
    """
    公式: R = w1 * r_progress + w2 * r_promise - w3 * r_switch
    """
    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale_range = (1.5, 5.0)  # 简单问题用 1.5，极难问题放大到 5.0
        self.depth_boost = 2.0 # 随搜索深度额外放大的倍数上限
        self.current_scale = scale

        # 引入绝对生存税，只要走一步就必须扣分
        self.base_step_penalty = -0.01
        
        # --- 核心权重分配 (初始默认值，会在 reset 中被覆盖) ---
        self.w1 = 1.0  # 进步权重 (最重要，一旦突破给予重奖)
        self.w2 = 0.2  # 潜力权重 (稠密奖励，引导 AI 每一步的选择)
        self.w3 = 0.3  # 跳跃惩罚 (引擎保护器，防止频繁冷启动)

        self.alpha = 2.0  # PB 突破的倍率
        self.beta = 1.5   # DB 突破的倍率
        self.gamma = 0.5  # 估计值 (Estimate) 相对于下界的权重

        # --- 新增：停滞监测参数 ---
        self.stagnation_counter = 0  # 连续无进展计数器
        #self.base_time_penalty = -0.01  # 基础每步惩罚（极小，防止抵消正常奖励）
        self.penalty_step = 0.02  # 停滞累积因子
        self.penalty_floor = -0.5      # 惩罚下限：无论停滞多久，单步惩罚不低于此值

        self.floor_range = (-0.4, -0.1)  # 简单问题扣分狠，难题扣分轻
        self.step_range = (0.01, 0.002)  # 简单问题惩罚累积快，难题累积慢
        
        self.reset(1.0, 0.0, 1.0, "timelimit", 400.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=400.0, logger=None):
        ## 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        
        # 【修复】：必须在这里计算 logB，否则下面计算 d 的时候会报错
        self.logB = math.log1p(self.B)
        
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)

        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 400.0), 1.0)
        self.logger = logger or self.logger
        
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # 用于暂存上一步选择的节点质量和跳跃代价
        self.last_r_promise = 0.0
        self.last_r_switch = 0.0

        self.stagnation_counter = 0 # 重置计数器

        ##归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)

        # 难题更需要“显微镜”去放大微小的进展
        #self.current_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 2. 初始 Scale (根据问题难度锚定起点)
        # 难题起点更高 (e.g., d=1 -> scale=4.0; d=0 -> scale=1.5)
        self.initial_scale = self.scale_range[0] * (1 - d) + self.scale_range[1] * d

        # 难题 (d=1): floor=-0.1, step=0.002 (温柔)
        # 简单 (d=0): floor=-0.4, step=0.01 (严厉)
        self.penalty_floor = self.floor_range[0] * (1 - d) + self.floor_range[1] * d
        self.penalty_step = self.step_range[0] * (1 - d) + self.step_range[1] * d
        
        # ==========================================
        # 动态权重引擎 (Dynamic Weights)
        # ==========================================
        # w1 (全局进步): 永远是北极星，锚定在 1.0 附近，越难的问题越依赖它
        self.w1 = 0.8 * (1 - d) + 1.2 * d   # 0.8 -> 1.2
        
        # # 让难题的 w2 更大，简单问题反而不需要那么多引导。
        self.w2 = 0.1 * (1 - d) + 0.5 * d   # 0.1 -> 0.3
        
        # 跳跃权重，难问题跳跃惩罚重一些
        self.w3 = 0.1 * (1 - d) + 0.3 * d   # 0.1 -> 0.3

        s = self.w1 + self.w2 + self.w3
        self.w1 /= s
        self.w2 /= s
        self.w3 /= s

        # ======== 探针 1 ========
        # if self.logger:
        #     print(f"\n[探针1-Reset] 实例基线节点 B: {self.B:.1f} | 算出的难度 d: {d:.3f}")
        #     print(f"[探针1-Reset] 归一化权重 -> w1(进步): {self.w1:.3f}, w2(潜力): {self.w2:.3f}, w3(跳跃): {self.w3:.3f}")
        # ========================


    
    def set_action_feedback(self, norm_lb, norm_est, switch_penalty):
        norm_lb = _clip_unit_interval(norm_lb)
        norm_est = _clip_unit_interval(norm_est)

        # 最完美的结点给出 0 分，越差的结点扣分越多。绝对不给正分！
        promise_penalty = (norm_lb + self.gamma * norm_est) / (1.0 + self.gamma) 
        
        # 加个负号，让它变成惩罚项
        # 范围在 [-1.0, 0.0]
        # 潜力评估，都是负数
        self.last_r_promise = - _safe_tanh(promise_penalty, s=5.0)
        # 跳跃惩罚
        self.last_r_switch = switch_penalty


    def compute(self, model, done):

        # nodes_ratio 越大，说明搜得越深。利用 log 让增长在前期快，后期稳。
        current_nodes = max(float(model.getNNodes()), 1.0)
        # phi 会从 0 慢慢增长，在达到基线节点数 B 时约为 1.0
        
        phi = min(math.log1p(current_nodes) / max(math.log1p(self.B), 2.3), 1.5) # 2.3 约等于 log(10)
        
        # --- 实时动态 Scale ---
        # 随搜索深度线性增加 scale，确保后期微小进展也能被“显微镜”放大
        # 最终 scale = 初始 scale * (1 + 深度加成因子)
        active_scale = self.initial_scale * (1.0 + phi * 0.5) 
    
        gap = float(model.getGap())
        if math.isinf(gap): gap = 1e6

        #PDI 越小，说明它在实际应用中越优秀
        pdi = float(model.getPrimalDualIntegral())

        ##当前最好可行解(上界)
        pb = model.getPrimalbound()
        ##当前松弛下界
        db = model.getDualbound()

        r_progress = 0.0 #进展奖励

        has_real_progress = False

        # 1. 基础进步：Gap 缩小
        if self.prev_gap < float('inf') and gap < self.prev_gap:
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 核心突破：找到了更好的整数解 (PB 下降)
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += self.alpha * _safe_tanh((self.prev_pb - pb) / max(abs(self.prev_pb), 1e-9), s=active_scale)
            has_real_progress = True
        # 3. 核心突破：证明了子树，提升了全局下界 (DB 上升)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += self.beta * _safe_tanh((db - self.prev_db) / max(abs(self.prev_db), 1e-9), s=active_scale)
            has_real_progress = True
        # 2. 检测是否有实质性突破
        

        # current_gap = float(model.getGap())
        if math.isinf(gap) or gap > 1.0:
            # 【阶段 1：找解期】Gap 极大，说明还没有找到像样的可行解
            # 策略：重罚跳跃，逼迫模型像 DFS 一样深入树底去寻找可行解。
            active_w1 = self.w1       # 保持基础进步权重
            active_w2 = self.w2       # 保持基础潜力引导
            active_w3 = self.w3           # ✨ 强硬的跳跃惩罚
        else:
            # 【阶段 2：证明期】Gap 已经小于 1.0 (100%)，说明找到了不错的解
            # 策略：此时任务变成了提升全局下界。大幅降低跳跃惩罚，鼓励模型全图飞奔（Best-Bound 模式）。
            # 随着 gap 越来越接近 0，跳跃惩罚也越来越趋近于 0。
            active_w1 = self.w1  # 稍微放大进步的奖励，鼓励收尾
            active_w2 = self.w2
            active_w3 = self.w3 * max(gap, 0.0) # ✨ 跳跃惩罚随 Gap 衰减，Gap为0时跳跃免费
  
        # 2. 停滞惩罚逻辑 (Stagnation Penalty)
        if has_real_progress:
            self.stagnation_counter = 0
            stagnation_penalty = 0.0
        else:
            self.stagnation_counter += 1
            # 计算线性累积惩罚：随着步数增加，扣分变多
            raw_penalty = - (self.stagnation_counter * self.penalty_step)
            # 使用 max 取两者中的较大值（因为是负数，所以起到了截断 Floor 的作用）
            stagnation_penalty = max(raw_penalty, self.penalty_floor)


        # 更新历史记录
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            # 只要没解完，每走一步底薪就是负的！
            base_step_penalty = -0.01
            # 代入你的宏伟公式
            reward = base_step_penalty + (active_w1 * r_progress) + (active_w2 * self.last_r_promise) - (active_w3 * self.last_r_switch)+ \
                        stagnation_penalty
            
            # ======== 探针 2 ========
            # print(f"[探针2-Step] 步数: {current_nodes} | 总奖励: {reward:.4f} (截断前)")
            # print(f" ┣━ 基础底薪: {base_step_penalty:.3f}")
            # print(f" ┣━ w1 * 进步(r_progress): {self.w1 * r_progress:.3f}")
            # print(f" ┣━ w2 * 潜力(last_r_promise): {self.w2 * self.last_r_promise:.3f}")
            # print(f" ┣━ -w3 * 跳跃(last_r_switch): {-self.w3 * self.last_r_switch:.3f}")
            # print(f" ┗━ 停滞惩罚: {stagnation_penalty:.3f}")
            # ========================

            return float(np.clip(reward, -5.0, 5.0))
            # return float(np.clip(reward, -0.2, 5.0))
        
            

        # 终止状态（Terminal）：保持一定的加速奖励
        nodes = int(model.getNNodes())
        #衡量智能体相比于默认求解器（基线），在求解同一个问题时，将搜索树的规模缩小了多少倍。
        # speedup>1说明智能体探索的节点数比基线少 
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        status = model.getStatus()
        

        bonus = _terminal_bonus(status, speedup, pdi_gain, gap)

        # ======== 探针 3 ========
        # print(f"\n[探针3-Done] Episode 结束！状态 (Status): {status}")
        # print(f"[探针3-Done] 探索节点数: {nodes} | Speedup: {speedup:.3f} | PDI Gain: {pdi_gain:.3f}")
        # print(f"[探针3-Done] 最终发放 Bonus: {bonus:.4f}\n")
        # ========================
            
        return float(bonus)
