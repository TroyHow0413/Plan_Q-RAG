import sys
import os

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
from rl.agents.pqn import PQN
import numpy as np
from envs.qa_env import QAEnv
from envs.parallel_env import ParallelTextEnv
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from hydra import initialize, compose
import random
from datetime import datetime


@torch.no_grad()
def evaluate(env_test, agent):
    s_t = env_test.reset()
    done_t = False
    a_embeds_t, a_embeds_target_t = env_test.get_extra_embeds(agent.action_tokenizer, agent.critic_module.action_embed, agent.action_embed_target)
    r_sum_t = 0
    while not done_t:
        a_embeds_t = env_test.update_embeds(a_embeds_t, agent.critic_module.action_embed)
        a_embeds_target_t = env_test.update_embeds(a_embeds_target_t, agent.action_embed_target)
        
        action_t, _, _ = agent.select_action(s_t, a_embeds_t["rope"], a_embeds_target_t["rope"], random=False, evaluate=True)
        s_t, _, reward_t, done_t = env_test.step(action_t)
        r_sum_t += reward_t
    
    return r_sum_t


def load_config(name, overrides=None):
    with initialize(version_base="1.3", config_path="./configs"):
        cfg = compose(
            config_name=name,
            overrides=sys.argv[1:]
        )
        cfg = prepare_config(cfg)
        return cfg


def prepare_config(cfg):
    """
    modifies config for parameters that should depend on each other
    """
    if cfg.logger.log_dir is not None:
        dir_name = datetime.now().strftime("%b%d_%H-%M-%S") + cfg.logger.tensorboard.comment
        cfg.logger.log_dir = os.path.join(cfg.logger.log_dir, dir_name)
        cfg.logger.tensorboard.log_dir = os.path.join(cfg.logger.log_dir, 'tb_logs/')
    return cfg


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================
# DDP 初始化（必须在任何 CUDA 操作之前）
# ============================================================
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
IS_MAIN = (rank == 0)   # 只有 rank 0 负责 eval / log / save
# ============================================================

cfg: DictConfig = load_config(name="training.yaml")

# 只有 rank 0 初始化 tensorboard / 创建目录 / 保存 config
if IS_MAIN:
    writer: SummaryWriter = instantiate(cfg.logger.tensorboard)
    os.makedirs(cfg.logger.log_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.logger.log_dir, "config.yaml")
    OmegaConf.save(config=cfg, f=config_save_path, resolve=False)
    print(f"[INFO] Training config saved to {config_save_path}")

agent_config: DictConfig = cfg.algo
env_config: DictConfig = cfg.envs

if IS_MAIN:
    print("Embedder model:", agent_config.model.model_name)

# path to checkpoints and metric to determine the best model
ckpt_last_path = os.path.join(cfg.logger.log_dir, "model_last.pt")
ckpt_best_path = os.path.join(cfg.logger.log_dir, "model_best.pt")
best_eval_reward = -float("inf")

# 每个 rank 用不同 seed，保证各自收集不同的环境数据
torch.set_default_device(torch.device(f"cuda:{local_rank}"))
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed + rank)

agent = PQN(agent_config)

# ============================================================
# 用 DDP 包裹 critic（唯一需要梯度同步的模型）
# policy / v_net_target / action_embed_target 靠 soft/hard update
# 跟随 critic，不需要 DDP 包裹
# ============================================================
agent.critic = DDP(agent.critic, device_ids=[local_rank])
# ============================================================

env: QAEnv = instantiate(env_config.env)
env_test: QAEnv = instantiate(env_config.test_env)   # 只有 rank 0 实际使用
parallel_env = ParallelTextEnv(
    [env] + [env.copy() for _ in range(cfg.envs_parallel - 1)],
    state_tokenizer=agent.state_tokenizer,
    action_tokenizer=agent.action_tokenizer)

total_steps   = cfg.steps_count * cfg.accumulate_grads       # 80000
eval_interval = cfg.eval_interval * cfg.accumulate_grads     # 50 * 8 = 400 it
log_interval  = 100                                           # 每 100 it 记录 reward / qf_loss

progress_bar = tqdm(range(total_steps), desc="Training") if IS_MAIN else range(total_steps)

# ---------- log.txt（只有 rank 0 写） ----------
if IS_MAIN:
    log_path = os.path.join(cfg.logger.log_dir, "log.txt")
    log_file = open(log_path, "w", buffering=1)
    print(f"[INFO] Log file: {log_path}")
# -----------------------------------------------

states_list, _ = parallel_env.reset()
step = 0
train_rewards = []
last_eval_reward = float("nan")

for it in progress_bar:

    agent.train()
    states_list, rewards, train_batch = parallel_env.rollout(
        cfg.batch_size, states_list, agent,
        random=(step < 2 * cfg.learning_start))
    step += np.prod(train_batch.reward.shape)
    train_rewards.extend(rewards)

    qf_loss = agent.update(
        train_batch.state,
        train_batch.action,
        train_batch.next_state,
        train_batch.q_values,
        train_batch.reward,
        train_batch.not_done)

    # ---- 每 100 it：rank 0 记录 reward / qf_loss ----
    if IS_MAIN and it % log_interval == 0 and it > 0:
        train_r_mean = np.mean(train_rewards)
        writer.add_scalar("train r_sum", train_r_mean, step)
        writer.add_scalar("qf_loss", qf_loss, step)
        log_file.write(
            f"{it}/{total_steps}, "
            f"reward={train_r_mean:.3f}, "
            f"eval_reward={last_eval_reward:.3f}, "
            f"qf_loss={float(qf_loss):.3f}, "
            f"step={step}\n"
        )

    # ---- 每 400 it：rank 0 跑 eval / 保存模型 ----
    if IS_MAIN and it % eval_interval == 0:

        agent.eval()

        r_eval = []
        for j in range(cfg.eval_episodes):
            r_eval.append(evaluate(env_test, agent))
            print(f"\reval prog: {len(r_eval)}/{cfg.eval_episodes}", end="")

        last_eval_reward = float(np.mean(r_eval))
        writer.add_scalar("eval r_sum", last_eval_reward, step)

        progress_bar.set_postfix({
            'reward': np.mean(train_rewards),
            "eval_reward": last_eval_reward,
            'qf_loss': qf_loss,
            'step': step,
        })

        agent.save(ckpt_last_path)

        if last_eval_reward > best_eval_reward:
            best_eval_reward = last_eval_reward
            agent.save(ckpt_best_path)

        train_rewards = []

if IS_MAIN:
    log_file.close()

dist.destroy_process_group()
