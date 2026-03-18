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

class RewardNodeSelection:
    """
    基于微观动作导向的结点选择奖励函数。
    公式: R = w1 * r_progress + w2 * r_promise - w3 * r_switch
    """
    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale_range = (1.5, 5.0)  # 简单问题用 1.5，极难问题放大到 5.0
        self.depth_boost = 2.0 # 随搜索深度额外放大的倍数上限
        self.current_scale = scale
        
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
        self.penalty_step = 0.005  # 停滞累积因子
        self.penalty_floor = -0.2      # 惩罚下限：无论停滞多久，单步惩罚不低于此值

        self.floor_range = (-0.4, -0.1)  # 简单问题扣分狠，难题扣分轻
        self.step_range = (0.01, 0.002)  # 简单问题惩罚累积快，难题累积慢
        
        self.reset(1.0, 0.0, 1.0, "timelimit", 900.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=900.0, logger=None):
        ## 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        
        # 【修复】：必须在这里计算 logB，否则下面计算 d 的时候会报错
        self.logB = math.log1p(self.B)
        
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)

        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 900.0), 1.0)
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

    
    def set_action_feedback(self, norm_lb, norm_est, switch_penalty):
        # norm_lb 和 norm_est 越小越好 (0 是最好，1 是最差)
        # 我们把它翻转成奖励：1 是最好，0 是最差
        #范围在[-0.75, 0.75]
        self.last_r_promise = (
                (1.0 - norm_lb) + self.gamma * (1.0 - norm_est)
            ) - (1.0 + self.gamma) / 2
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
            # 代入你的宏伟公式
            reward = (self.w1 * r_progress) + (self.w2 * self.last_r_promise) - (self.w3 * self.last_r_switch)+ \
                        stagnation_penalty

            return float(np.clip(reward, -5.0, 5.0))

        # 终止状态（Terminal）：保持一定的加速奖励
        nodes = int(model.getNNodes())
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        status = model.getStatus()
        
        if status == "optimal": 
            bonus = 1.0 + 3.0 * speedup + 1.0 * pdi_gain
        elif status in ("infeasible", "unbounded"): 
            bonus = 0.8 + 2.0 * speedup + 1.0 * pdi_gain
        else: 
            bonus = 0.5 * speedup + 0.5 * pdi_gain
            
        return float(bonus)