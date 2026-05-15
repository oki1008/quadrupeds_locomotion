
import os
import argparse
import importlib
import pickle
import shutil
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def load_config(config_path):
    """YAMLから環境・観測・報酬・指令・学習設定を読み込む。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['env_cfg'], config['obs_cfg'], config['reward_cfg'], config['command_cfg'], config['train_cfg']


def create_default_config(config_path):
    raise NotImplementedError("Default config creation not implemented. Please create a config.yaml file based on the provided template.")


def resolve_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[go2_train] CUDA was requested but is not available in this container. Falling back to CPU.")
        return "cpu"
    return requested_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="path to configuration YAML file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found. Creating default configuration.")
        create_default_config(args.config)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_config(args.config)

    exp_name = train_cfg["exp_name"]
    num_envs = train_cfg["num_envs"]
    max_iterations = train_cfg["max_iterations"]
    env_name = train_cfg["env"]
    device = train_cfg["device"]

    effective_device = resolve_device(device)
    backend = gs.constants.backend.gpu if effective_device.startswith("cuda") else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir = f"logs/{exp_name}"
    train_cfg_full = get_train_cfg(exp_name, max_iterations)

    if effective_device == "cpu" and num_envs == 4096:
        num_envs = 256
        print("[go2_train] CPU execution detected. Reducing num_envs from 4096 to 256 for a usable training speed.")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 後から同じ条件を確認できるよう、学習時のconfigをログに保存する。
    shutil.copy(args.config, os.path.join(log_dir, "config.yaml"))

    # configのenv名から環境クラスを読み込む。例: Go2Env_Stair.py / Go2Env_Stair
    env_module = importlib.import_module(env_name)
    env_class = getattr(env_module, env_name)
    env = env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=effective_device,
    )
   

    runner = OnPolicyRunner(env, train_cfg_full, log_dir, device=effective_device)

    # 再開学習を使う場合は、ここでrunner.load(...)を呼ぶ。

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_full],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
