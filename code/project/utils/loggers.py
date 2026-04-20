try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ModuleNotFoundError:
    _TENSORBOARD_AVAILABLE = False

    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.log_dir = kwargs.get("log_dir")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None
import sys
import os
from datetime import datetime
import uuid
import pandas as pd
import logging

# def setup_logging(log_file=None):
#     if log_file is None:
#         log_file = f"logs/ppo_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )
#     return logging.getLogger(__name__)

import multiprocessing
import sys
import logging
from datetime import datetime

def setup_logging(log_file=None):
    if log_file is None:
        log_file = f"logs/ppo_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 获取根日志记录器
    logger = logging.getLogger()
    
    # 如果已经有 handler（说明是子进程继承或重复调用），先清理掉
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO) # 建议生产环境设为 INFO，DEBUG 太刷屏导致 I/O 变慢
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. 所有进程都保留 FileHandler，确保日志记录完整
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 【核心优化】只有主进程才添加 StreamHandler
    # 这样子进程的日志只进文件，不通过 stdout 传给父进程，彻底消除双重打印
    if multiprocessing.current_process().name == 'MainProcess':
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        # 子进程可以降低日志级别，减少写入压力
        logger.setLevel(logging.WARNING)

    # 返回当前模块的 logger
    return logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, args):
        os.makedirs(args.logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]

        self.csv_path = os.path.join(args.logs_dir,f"training_metrics_{unique_id}_{timestamp}.csv")

        # Initialize empty DataFrame
        self.metrics_df = pd.DataFrame()

    def log_episode(self, episode_metrics):
        """
        Log metrics for one episode
        """
        # Convert metrics to DataFrame row
        metrics_row = pd.DataFrame([episode_metrics])

        # Append to existing DataFrame
        self.metrics_df = pd.concat([self.metrics_df, metrics_row], ignore_index=True)

        # Save to CSV after each episode
        self.metrics_df.to_csv(self.csv_path, index=False)


class MetricsTrialLogger:
    def __init__(self, args, trial=None, flush_interval=1):
        """
        Initialize MetricsTrialLogger for CSV and TensorBoard logging.
        
        Args:
            args: Command-line arguments containing logs_dir and num_episodes.
            trial: Optuna trial object (optional).
            flush_interval: Number of episodes to buffer before writing to CSV.
        """
        os.makedirs(args.logs_dir, exist_ok=True)

        if trial:
            trial_id = trial.number
        else:
            trial_id = "manual"

        # Timestamp and file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        self.csv_path = os.path.join(args.logs_dir, f"training_metrics_trial_{trial_id}_{unique_id}_{timestamp}.csv")
        self.tb_path = os.path.join(args.logs_dir, f"tb_trial_{trial_id}_{timestamp}")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tb_path)

        # CSV buffer
        self.buffer = []
        self.flush_interval = flush_interval
        self.file_exists = os.path.exists(self.csv_path)

        self.logger = logging.getLogger(f"MetricsTrialLogger_trial_{trial_id}")
        self.logger.info(f"MetricsTrialLogger initialized with CSV path {self.csv_path} and TensorBoard path {self.tb_path}")
        if not _TENSORBOARD_AVAILABLE:
            self.logger.warning("tensorboard is not installed; TensorBoard logging is disabled.")

    def log_episode(self, episode_metrics, metrics_type='train'):
        """
        Log metrics for one episode to both CSV and TensorBoard.
        
        Args:
            episode_metrics: Dict with keys like 'episode', 'total_reward', 'nodes_explored', 'gap', etc.
            metrics_type: 'train' or 'val'
        """
        try:
            metrics_copy = episode_metrics.copy()
            episode_num = metrics_copy.get('episode', -1)
            metrics_copy['logged_at'] = datetime.now().isoformat()
            metrics_copy['type'] = metrics_type
            self.buffer.append(metrics_copy)

            # Write to TensorBoard
            for key, value in metrics_copy.items():
                if isinstance(value, (int, float)) and key not in ['episode']:
                    self.writer.add_scalar(f"{metrics_type}/{key}", value, global_step=episode_num)

            self.logger.info(f"Buffered and logged {metrics_type} metrics for episode {episode_num}")

            if len(self.buffer) >= self.flush_interval:
                self._flush_metrics()
        except Exception as e:
            self.logger.error(f"Error logging {metrics_type} episode {episode_metrics.get('episode', 'unknown')}: {e}")

    def _flush_metrics(self):
        """
        Flush CSV metrics from buffer.
        """
        try:
            if self.buffer:
                df = pd.DataFrame(self.buffer)
                header = not self.file_exists
                df.to_csv(self.csv_path, mode='a', header=header, index=False)
                self.file_exists = True
                self.logger.info(f"Flushed {len(self.buffer)} metrics to {self.csv_path}")
                self.buffer = []
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")

    def close(self):
        """
        Final cleanup.
        """
        try:
            self._flush_metrics()
            self.writer.close()
            self.logger.info("Closed MetricsTrialLogger and TensorBoard writer")
        except Exception as e:
            self.logger.error(f"Error closing MetricsTrialLogger: {e}")

