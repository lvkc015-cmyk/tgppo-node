from __future__ import annotations
# ---------------- Thread hygiene ----------------
import faulthandler

faulthandler.enable()
#######  我创建的环境为：tgppo_scip7
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

os.environ.setdefault("OMP_NUM_THREADS", "1")
#控制 Intel 数学核心库，强制其单线程运行可规避高并发下的线程冲突。
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import io
import gc
import sys
import json
import math
import time
import argparse
import logging
import random
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, NamedTuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import torch

# ---------------- Project imports ----------------
from project.utils import (
    setup_logging, strip_extension, get_device, settings, state_dims, scip_limits, get_reward
)
from NodeSelect.actor import Actor
from NodeSelect.critic import  Critic
from NodeSelect.agent import Agent
from NodeSelect.enviroment import Environment

from multiprocessing import Manager, Process
import multiprocessing as mp
from queue import Empty

from multiprocessing import Queue, Pool
import sys
import pickle
import psutil
import uuid
import shutil
import re



###  找错
# torch.autograd.set_detect_anomaly(True)

def save_metrics_to_csv(metrics: dict, filepath: str):
    #获取文件路径的目录部分,创建目录，如果目录已存在也不会报错
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #将字典metrics转换为DataFrame
    df = pd.DataFrame([metrics])
    if os.path.exists(filepath):
        #如果文件已存在：追加模式（'a'），不写入列标题，不写入行索引
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        # 如果文件不存在：创建新文件，不写入行索引（默认会写入列标题）
        df.to_csv(filepath, index=False)


def _fix_torch_threads():
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
_fix_torch_threads()

# ---------------- Data structures ----------------
@dataclass
class TrainArgs:
    logs_dir: str
    temp_traj_dir: str
    scip_setting: str
    time_limit: float
    seeds: List[int]
    per_job_timeout: int
    max_workers: int
    train_iterations: int
    checkpoint_every: int
    # Hyperparameters
    actor_lr: float
    critic_lr: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    policy_clip: float
    entropy_weight: float
    gamma: float
    gae_lambda: float
    batch_size: int
    n_epochs: int
    reward_function: str

class EpisodeMetrics(NamedTuple):
    instance: str
    seed: int
    status: str
    nnodes: int
    solve_time: float
    gap: float
    pdi: float
    episode_return: float

# ---------------- Builders ----------------

def build_models(cfg: TrainArgs) -> Tuple[Actor, Critic]:
    actor = Actor(
        # var_dim=state_dims["var_dim"],
        node_cand_dim = state_dims["node_cand_dim"],
        node_dim=state_dims["node_dim"],
        mip_dim=state_dims["mip_dim"],
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    critic = Critic(
        # var_dim=state_dims["var_dim"],
        node_cand_dim = state_dims["node_cand_dim"],
        node_dim=state_dims["node_dim"],
        mip_dim=state_dims["mip_dim"],
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    return actor, critic


def build_agent_env_with_models(actor: Actor, critic: Critic, cfg: TrainArgs, seed: int, logger: logging.Logger,force_cpu=False) -> Tuple[Agent, Environment]:
    # device = get_device(device="cpu")
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

    actor.to(device)
    critic.to(device)
    
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, weight_decay=1e-2)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=1e-2)

    reward_func = get_reward(cfg.reward_function)

    agent = Agent(
        actor_network=actor,
        actor_optimizer=actor_opt,
        critic_network=critic,
        critic_optimizer=critic_opt,
        policy_clip=cfg.policy_clip,
        entropy_weight=cfg.entropy_weight,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        device=device,
        state_dims=state_dims,
        logger=logger,
    )

    scip_params = settings.get(cfg.scip_setting, {}).copy()
    limits = scip_limits.copy()
    limits["time_limit"] = float(cfg.time_limit)

    env = Environment(
        device=device,
        agent=agent,
        state_dims=state_dims,
        scip_limits=limits,
        scip_params=scip_params,
        scip_seed=seed,
        reward_func=reward_func,
        logger=logger,
    )
    return agent, env

# ---------------- Serialization ----------------

def serialize_weights(actor: Actor, critic: Critic) -> bytes:
    buf = io.BytesIO()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, buf)
    return buf.getvalue()

def load_weights_bytes(actor: Actor, critic: Critic, payload: bytes) -> None:
    buf = io.BytesIO(payload)
    state = torch.load(buf, map_location="cpu")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])

# ---------------- Helpers: parallel batches ----------------

def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

#max_workers：进程数
def run_parallel_in_batches(jobs, fn, max_workers, per_job_timeout, tasks_per_child=1,
                            returns_path=False, time_limit_pad=60):
    batch_size = max(1, max_workers * max(1, tasks_per_child))
    outputs = []
    for batch in _batched(jobs, batch_size):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fn, job) for job in batch]
            deadline = per_job_timeout * len(batch) + time_limit_pad
            for fut in as_completed(futs, timeout=deadline):
                try:
                    res = fut.result(timeout=per_job_timeout)
                    outputs.append(res)
                except TimeoutError:
                    outputs.append(None)
                except Exception:
                    outputs.append(None)
    if returns_path:
        metrics, paths = [], []
        for item in outputs:
            if item is None:
                metrics.append(None); paths.append(None)
            else:
                m, p = item
                metrics.append(m); paths.append(p)
        return metrics, paths
    else:
        return outputs

# ---------------- Workers ----------------

def _make_worker_logger() -> logging.Logger:
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger

#在一个独立进程中，用给定策略网络权重，完整求解一个 MILP 实例（一次 B&B episode），收集 RL 轨迹并返回求解指标 + 轨迹文件路径。
def collect_rollout_job(job: Tuple[str, int, bytes, TrainArgs, Dict[str, Any]]) -> Tuple[EpisodeMetrics, str]:
    #instance_path  # MILP 实例路径
    instance_path, seed, weights_bytes, cfg, info_dict = job
    logger = _make_worker_logger()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #在子进程中重建模型并加载权重
    actor, critic = build_models(cfg)
    actor.cpu()
    critic.cpu()
    load_weights_bytes(actor, critic, weights_bytes)

    #构建 agent + SCIP 环境
    agent, env = build_agent_env_with_models(actor, critic, cfg, seed, logger,force_cpu=True)

    name = strip_extension(os.path.basename(instance_path)).split(".")[0]
    meta = info_dict.get(name, {})

    logger.info(f">>> [WORKER] Starting new rollout for instance: {name}")

    

    try:
        # 3. 核心执行逻辑
        env.reset(instance_path, meta.get("cutoff"), meta.get("baseline_nodes"), 
                  meta.get("baseline_gap"), meta.get("baseline_integral"), meta.get("baseline_status"))
        
        # 这个 run_episode 会触发你的 BranchingRule.branchexeclp
        done, info, ep_reward = env.run_episode()

        logger.info(f">>> [WORKER] Finished rollout for {name}, attempting to put data...")
        
        #检查 agent 对象是否有 finalize_advantages 这个方法
        if hasattr(agent, "finalize_advantages"):
            agent.finalize_advantages()
            
        # 4. 【核心修改】：通过 Queue 发送数据，不再写磁盘
        if hasattr(agent, "memory") and hasattr(agent.memory, "export_dict"):
            # export_dict 已经把 tensor 转成了 numpy，这在进程间传输更快
            traj_data = agent.memory.export_dict()
            
            # --- 原子落盘逻辑 ---
            # 定义保存目录（建议在主程序开始前已创建好）
            traj_dir = os.path.abspath(cfg.temp_traj_dir) 
            os.makedirs(traj_dir, exist_ok=True)
            
            # 使用 UUID 保证文件名唯一，避免进程间冲突
            # 先写 .tmp，写完再重命名为 .pkl，保证主进程不会读到损坏的半成品
            unique_id = uuid.uuid4()
            tmp_file = os.path.join(traj_dir, f"{name}_{unique_id}.tmp")
            final_file = tmp_file.replace(".tmp", ".pkl")
            try:
                with open(tmp_file, 'wb') as f:
                    pickle.dump(traj_data, f)
                # rename 在 Linux 下是原子操作，是多进程最稳健的同步方式
                os.rename(tmp_file, final_file)
                
                size_mb = os.path.getsize(final_file) / (1024 * 1024)
                logger.info(f">>> [WORKER] Data saved to {final_file} ({size_mb:.2f} MB)")
            except Exception as save_err:
                logger.error(f"Failed to save trajectory file for {name}: {save_err}")
            
                        
            # 使用原生的 Queue 时，如果队列满了（maxsize=100），put 会卡死。
            # 这里设置超时，如果 60 秒都塞不进去，说明主进程崩了或太慢，直接抛错。
            # 实时放入队列，主进程的 while 循环会立刻取走
            # try:
            #     data_queue.put(traj_data, timeout=60) 
            #     logger.info(f">>> [WORKER] Data successfully put for {name}")
            # except Exception as q_err:
            #     logger.error(f"Queue put timeout or error in worker {name}: {q_err}")

        # 5. 构造返回给 map_async 的统计指标 (不再返回路径)
        status = str(info.get("status"))
        nnodes = int(info.get("nnodes", 0))
        solve_time = float(info.get("scip_solve_time", 0.0))
        gap = float(info.get("gap", 1.0))
        pdi = float(info.get("primalDualIntegral", info.get("pdi", 0.0)))
        
        metrics = EpisodeMetrics(name, seed, status, nnodes, solve_time, gap, pdi, float(ep_reward))
        return metrics, None 

    except Exception as e:
        logger.error(f"Async Job failed on {instance_path}: {e}")
        return EpisodeMetrics(name, seed, "error", int(1e9), cfg.time_limit, 1.0, 1e9, -1.0), None
    finally:
        # 显式清理资源，防止子进程内存泄漏
        try:
            # if hasattr(env, 'close'):
            #     env.close()
            del env, agent, actor, critic
        except:
            pass
        gc.collect()

    # cutoff = meta.get("cutoff")
    # baseline_nodes = meta.get("baseline_nodes")
    # baseline_gap = meta.get("baseline_gap")
    # baseline_integral = meta.get("baseline_integral")
    # baseline_status = meta.get("baseline_status")

    # try:
    #     #核心：一次完整 episode（一次 MILP 求解）
    #     env.reset(instance_path, cutoff, baseline_nodes, baseline_gap, baseline_integral, baseline_status)
    #     done, info, ep_reward = env.run_episode()
    #     if hasattr(agent, "finalize_advantages"):
    #         #优势函数
    #         agent.finalize_advantages()
    #     if not hasattr(agent, "memory") or not hasattr(agent.memory, "export_dict"):
    #         raise RuntimeError("Agent.memory.export_dict() required")
    #     #轨迹导出
    #     traj = agent.memory.export_dict()

    #     #将轨迹写入磁盘
    #     tmp_dir = os.path.join(cfg.logs_dir, "tmp_traj")
    #     os.makedirs(tmp_dir, exist_ok=True)
    #     traj_path = os.path.join(tmp_dir, f"traj_{os.getpid()}_{seed}_{time.time_ns()}.pkl")
    #     with open(traj_path, "wb") as f:
    #         pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     status = str(info.get("status"))
    #     nnodes = int(info.get("nnodes", 0))
    #     solve_time = float(info.get("scip_solve_time", 0.0))
    #     gap = float(info.get("gap", 1.0))
    #     pdi = float(info.get("primalDualIntegral", info.get("pdi", 0.0)))
    #     #构造 episode 指标
    #     metrics = EpisodeMetrics(name, seed, status, nnodes, solve_time, gap, pdi, float(ep_reward))
    #     return metrics, traj_path
    # except Exception:
    #     logger.exception("collect_rollout_job failed")
    #     tmp_dir = os.path.join(cfg.logs_dir, "tmp_traj")
    #     os.makedirs(tmp_dir, exist_ok=True)
    #     traj_path = os.path.join(tmp_dir, f"traj_error_{os.getpid()}_{seed}_{time.time_ns()}.pkl")
    #     with open(traj_path, "wb") as f:
    #         pickle.dump({}, f)
    #     return EpisodeMetrics(name, seed, "error", int(1e9), cfg.time_limit, 1.0, 1e9, -1.0), traj_path
    # finally:
    #     try:
    #         del env, agent
    #     except Exception:
    #         pass
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()


def validation_job(job: Tuple[str, int, bytes, TrainArgs, Dict[str, Any]]) -> EpisodeMetrics:
    instance_path, seed, weights_bytes, cfg, info_dict = job
    logger = _make_worker_logger()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor, critic = build_models(cfg)
    load_weights_bytes(actor, critic, weights_bytes)
    actor.eval(); critic.eval()
    agent, env = build_agent_env_with_models(actor, critic, cfg, seed, logger)

    name = strip_extension(os.path.basename(instance_path)).split(".")[0]
    meta = info_dict.get(name, {})
    cutoff = meta.get("cutoff")
    baseline_nodes = meta.get("baseline_nodes")
    baseline_gap = meta.get("baseline_gap")
    baseline_integral = meta.get("baseline_integral")
    baseline_status = meta.get("baseline_status")

    try:
        env.reset(instance_path, cutoff, baseline_nodes, baseline_gap, baseline_integral, baseline_status)
        done, info, ep_reward = env.run_episode(learn=False)
        status = str(info.get("status"))
        nnodes = int(info.get("nnodes", 0))
        solve_time = float(info.get("scip_solve_time", 0.0))
        gap = float(info.get("gap", 1.0))
        pdi = float(info.get("primalDualIntegral", info.get("pdi", 0.0)))
        return EpisodeMetrics(name, seed, status, nnodes, solve_time, gap, pdi, float(ep_reward))
    except Exception:
        logger.exception("validation_job failed")
        return EpisodeMetrics(name, seed, "error", int(1e9), cfg.time_limit, 1.0, 1e9, -1.0)
    finally:
        try:
            del env, agent
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------- Scoring ----------------

def score_from_metrics(results: List[EpisodeMetrics], time_limit: float) -> float:
    if not results:
        return float("inf")
    times, nodes = [], []
    penalties = 0.0
    for r in results:
        par10 = r.solve_time if r.status == "optimal" else 10.0 * time_limit
        times.append(par10)
        nodes.append(r.nnodes)
        if r.status == "error":
            penalties += 1e6
    arr_t = np.asarray(times, dtype=float)
    arr_n = np.asarray(nodes, dtype=float)
    shift = 100.0
    sgm_time = math.exp(np.mean(np.log(arr_t + shift))) - shift
    sgm_nodes = math.exp(np.mean(np.log(arr_n + shift))) - shift
    return float(0.6 * sgm_time + 0.4 * sgm_nodes + penalties)

# ---------------- IO utils ----------------

def list_instances(dirpath: str) -> List[str]:
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if f.endswith('.mps') or f.endswith('.mps.gz') or f.endswith('.lp')]


def save_checkpoint(path: str, actor: Actor, critic: Critic, agent: Agent, it: int, score: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "iter": it,
        "score": score,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_opt": agent.actor_optimizer.state_dict() if hasattr(agent, "actor_optimizer") else None,
        "critic_opt": agent.critic_optimizer.state_dict() if hasattr(agent, "critic_optimizer") else None,
        "hparams": {k: v for k, v in agent.__dict__.items() if k in []},  # placeholder if you wish
    }
    torch.save(payload, path)

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Final training of Tree‑Gate PPO with parallel rollouts")
    parser.add_argument('--instances_dir', type=str, required=True)
    parser.add_argument('--instances_info_dict', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--best_params_json', type=str, required=True, help='JSON with best hyperparameters')
    parser.add_argument('--output_model', type=str, default='bi_output/models/final_model.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='bi_output/checkpoints')

    parser.add_argument('--time_limit', type=float, default=3600.0)
    parser.add_argument('--per_job_timeout', type=int, default=3700) #每个并行工作任务的最大允许执行时间
    parser.add_argument('--scip_setting', type=str, default='sandbox')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--train_iterations', type=int, default=12)
    parser.add_argument('--checkpoint_every', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=20) #并行进程数量

    parser.add_argument('--val_instances_dir', type=str, default=None)
    parser.add_argument('--val_seeds', type=int, nargs='+', default=[0])
    
    # [新增] 提取硬编码的文件名和路径作为参数
    # ---------------------------------------------------------
    parser.add_argument('--temp_traj_dir', type=str, default='temp_trajs', help='临时轨迹文件存放目录')
    parser.add_argument('--metrics_csv', type=str, default='bi_output/training_metrics.csv', help='训练指标CSV输出路径')
    parser.add_argument('--log_file_name', type=str, default='bi_training.log', help='训练主日志文件名')
    parser.add_argument('--progress_json_name', type=str, default='bi_train_progress.json', help='训练进度JSON文件名')
    # ---------------------------------------------------------

    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    logger = setup_logging(log_file=os.path.join(args.logs_dir, args.log_file_name))
    logger.info("Starting final training…")

    with open(args.instances_info_dict, 'rb') as f:
        info_dict = pickle.load(f) #多个问题实例对象

    with open(args.best_params_json, 'r') as f:
        bp = json.load(f) #参数

    # Resolve train/val instances
    train_instances = list_instances(args.instances_dir) #训练数据
    if not train_instances:
        logger.error("No instances found in --instances_dir")
        sys.exit(1)

    if args.val_instances_dir:
        val_instances = list_instances(args.val_instances_dir)
    else:
        val_instances = []  # optional validation

    # Build TrainArgs from best params
    targs = TrainArgs(
        logs_dir=args.logs_dir,
        temp_traj_dir=args.temp_traj_dir,
        scip_setting=args.scip_setting,
        time_limit=args.time_limit,
        seeds=args.seeds,
        per_job_timeout=args.per_job_timeout,
        max_workers=args.max_workers,
        train_iterations=args.train_iterations,
        checkpoint_every=args.checkpoint_every,
        actor_lr=float(bp["actor_lr"]),
        critic_lr=float(bp["critic_lr"]),
        hidden_dim=int(bp["hidden_dim"]),
        num_layers=int(bp["num_layers"]),
        num_heads=int(bp["num_heads"]),
        dropout=float(bp["dropout"]),
        policy_clip=float(bp["policy_clip"]),
        entropy_weight=float(bp["entropy_weight"]),
        gamma=float(bp["gamma"]),
        gae_lambda=float(bp["gae_lambda"]),
        batch_size=int(bp["batch_size"]),
        n_epochs=int(bp["n_epochs"]),
        reward_function=str(bp["reward_function"]),
    )

    # Central learner
    actor, critic = build_models(targs)
    agent, _ = build_agent_env_with_models(actor, critic, targs, seed=0, logger=logger)

    # Logs
    train_rows: List[Dict[str, Any]] = []
    best_score = float("inf")

    # 1. 创建共享队列和进程池
    # manager = Manager()
    # data_queue = manager.Queue(maxsize=10) # 限制队列长度防止内存溢出

    # m_pid = manager._process.pid 
    # logger.info(f"Manager PID is {m_pid}")

    # # 在进入 train_iterations 循环前，先清理并创建临时目录
    traj_dir = os.path.abspath(args.temp_traj_dir)
    if os.path.exists(traj_dir):
        shutil.rmtree(traj_dir, ignore_errors=True)
    os.makedirs(traj_dir, exist_ok=True)

    #-----------------------断点续训----------------------------------
    # 1. 自动检测最新的 Checkpoint
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_iter_") and f.endswith(".pt")]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_iteration = 1
    if checkpoints:
        # 提取数字并找到最大的
        iters = [int(re.findall(r'\d+', f)[0]) for f in checkpoints]
        latest_it = max(iters)
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_iter_{latest_it}.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型状态
        checkpoint = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        # 如果 agent 也存了优化器状态，也可以加载
        # agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        # 【修复2】严格加载优化器状态，保住 AdamW 的动量！
        if 'actor_opt' in checkpoint and checkpoint['actor_opt'] is not None:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        if 'critic_opt' in checkpoint and checkpoint['critic_opt'] is not None:
            agent.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        
        start_iteration = latest_it + 1
        logger.info(f">>> Found checkpoint. Resuming from Iteration {latest_it}, next is {start_iteration}")

        # 【修复3】继承之前的 JSON 日志，防止历史数据被覆盖清空
        progress_json_path = os.path.join(args.logs_dir, args.progress_json_name)
        if os.path.exists(progress_json_path):
            try:
                with open(progress_json_path, 'r') as f:
                    train_rows = json.load(f)
            except json.JSONDecodeError:
                pass # 如果文件损坏则从空列表开始
        #---------------------------------------------------------------

    for it in range(start_iteration, targs.train_iterations + 1):
        t0 = time.time()
        #把 actor 和 critic 两个神经网络的参数提取出来，并序列化成一个可保存 / 可传输的结构
        # current_seed = (it - 1) % 5

        weights = serialize_weights(actor, critic)
        # 一个 job = 一个 MILP 实例 + 一个随机种子
        jobs = [(inst, s, weights, targs, info_dict) for inst in train_instances for s in targs.seeds]
        # jobs = [(inst, current_seed, weights, targs, info_dict,data_queue) for inst in train_instances]

        # 如果 SBATCH 申请了 20 核，max_workers 设为 16-18 最稳健
        # 留出 2 个核心给主进程做 GPU 训练和数据调度，防止 Manager 通信超时
        actual_workers = max(1, targs.max_workers - 2)
        # 创建一个 multiprocessing 进程池
        pool = Pool(processes=actual_workers)
        #把 jobs 列表分发给进程池,每个 job 调用一次 collect_rollout_job
        #result_async不包含 rollout 数据,只用于判断是否全部 worker 已结束,最后 get() 汇总结果
        result_async = pool.map_async(collect_rollout_job, jobs)

        logger.info(f"Iteration {it}: Started {len(jobs)} parallel rollout workers.")

        # 4. 异步训练循环 (Learner)
        # 只要后台 Worker 还在跑，或者队列里还有数据，主进程就不断 Learn
        num_traj_collected = 0  #从 data_queue 成功取到的 trajectory 数
        total_learn_updates = 0 #调用了多少次 agent.learn()
        last_learn_traj_count = 0  # 新增：记录上一次学习时的轨迹总数

        #result_async.ready():True：所有 Worker 都跑完了;False：至少还有一个 Worker 在跑
        #data_queue.empty():True：Learner 这边暂时没数据;False：还有没消费的 trajectory
        while not result_async.ready() or any(f.endswith('.pkl') for f in os.listdir(traj_dir)):
            # Buffer里的样本总数
            if len(agent.memory) >= (targs.batch_size * 50000):
                logger.warning(f">>> [BACKPRESSURE] Memory full ({len(agent.memory)}). Skipping ingestion to focus on learning.")
                traj_files = [] # 本轮循环不处理新文件
            else:
                # 正常扫描 .pkl 文件
                traj_files = [f for f in os.listdir(traj_dir) if f.endswith('.pkl')]
            try:
                if  traj_files:
                    sample_size = 100
                    sampled_files = random.sample(traj_files, min(sample_size, len(traj_files)))
                    
                    for filename in sampled_files[:100]:
                        file_path = os.path.join(traj_dir, filename)
                        
                        if not os.path.exists(file_path):
                            continue
                            
                        try:
                            # 1. 从磁盘加载数据
                            with open(file_path, 'rb') as f:
                                traj_dict = pickle.load(f)
                            
                            # 2. 导入 Memory 并删除文件（必须删掉，否则会重复读取）
                            if traj_dict:
                                agent.memory.import_dict(traj_dict)
                                num_traj_collected += 1
                            # 必须显式删除并清理
                            del traj_dict
                            os.remove(file_path)
                            
                            # 3. 增量训练逻辑
                            # if len(agent.memory) >= targs.batch_size:
                            #     break # 跳出文件读取循环，去执行下方的 agent.learn()

                        except (EOFError, pickle.UnpicklingError, FileNotFoundError):
                            # 如果文件正在写（虽然 rename 已经规避了大部分情况）或损坏，跳过
                            continue

                should_learn = False
                if num_traj_collected >= last_learn_traj_count + 100 and len(agent.memory) >= targs.batch_size:
                    should_learn = True
                
                # 条件 B: Worker 全部结束，把最后不足 15 个的数据也练了
                elif result_async.ready() and not any(f.endswith('.pkl') for f in os.listdir(traj_dir)) :
                    logger.info(">>> [LEARNER] Workers finished. Final cleanup learning...")
                    should_learn = True

                # if len(agent.memory) >= targs.batch_size:
                if should_learn:
                    #检查系统是否有可用的NVIDIA GPU，如果有GPU可用，返回True
                    if torch.cuda.is_available():
                        #释放那些已经不再被引用的缓存块
                        torch.cuda.empty_cache()

                    logger.info(f">>> [LEARNER] Memory {len(agent.memory)}. Learning...")
                    learn_metrics = agent.learn()
                    total_learn_updates += 1
                    last_learn_traj_count = num_traj_collected # 更新计数器
                    
                    if learn_metrics:
                        learn_metrics['iteration'] = it
                        learn_metrics['total_trajs_so_far'] = num_traj_collected
                        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
                        save_metrics_to_csv(learn_metrics, filepath=args.metrics_csv)
                        
                    # 训练完后的彻底清理
                    # del learn_metrics
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 如果既没有文件要读，数据量又不够 batch，则稍作休息
                # if not traj_files and len(agent.memory) < targs.batch_size:
                if not traj_files and not should_learn:
                    time.sleep(5)


            except Exception as e:
                logger.exception("Fatal error in learner loop")
                raise
            
        # 5. 确保进程池资源回收
        pool.close()
        pool.join()

        # 6. 获取汇总的 Metrics (来自 map_async 的结果)
        # 这里的 metrics_list 包含了每个实例的求解时间、Gap等统计信息
        all_results = result_async.get() 
        metrics_list = [res[0] for res in all_results if res is not None]

        elapsed = time.time() - t0

        # Optional interim validation  可选验证（不参与训练）
        val_score = None
        if val_instances:
            final_w = serialize_weights(actor, critic)
            vjobs = [(inst, s, final_w, targs, info_dict) for inst in val_instances for s in (args.val_seeds or [0])]
            vmetrics = run_parallel_in_batches(
                vjobs,
                fn=validation_job,
                max_workers=targs.max_workers,
                per_job_timeout=targs.per_job_timeout,
                tasks_per_child=2,
                returns_path=False,
            )
            vmetrics = [m for m in vmetrics if m is not None]
            val_score = score_from_metrics(vmetrics, targs.time_limit)

        row = {
            "iter": it,
            "elapsed_sec": elapsed,
            "num_traj": num_traj_collected,
            "num_updates": total_learn_updates,
            "learn_metrics": learn_metrics if total_learn_updates > 0 else {},
            "val_score": val_score,
        }
        #训练日志。 保存每一轮的：轨迹数，PPO loss，验证分数，耗时
        train_rows.append(row)
        pd.DataFrame(train_rows).to_json(os.path.join(args.logs_dir, args.progress_json_name), orient="records", indent=2)

        # Checkpoint  定期保存
        if (it % targs.checkpoint_every == 0) or (val_score is not None and val_score < best_score):
            score_for_ckpt = val_score if val_score is not None else float(it)
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_iter_{it}.pt")
            save_checkpoint(ckpt_path, actor, critic, agent, it, score_for_ckpt)
            if val_score is not None and val_score < best_score:
                best_score = val_score

        logger.info(f"Iter {it}/{targs.train_iterations}  traj={num_traj_collected}  time={elapsed:.1f}s  val_score={val_score}")

    # Save final model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, args.output_model)
    logger.info(f"Saved final model to {args.output_model}")


if __name__ == "__main__":
    main()
