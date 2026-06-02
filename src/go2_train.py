
import os
import argparse
import copy
import csv
import importlib
import pickle
import shutil
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from go2_eval import build_command_values, evaluate_policy


def get_train_cfg(exp_name, max_iterations, train_cfg=None):
    train_cfg = train_cfg or {}

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": train_cfg.get("num_learning_epochs", 5),
            "num_mini_batches": train_cfg.get("num_mini_batches", 4),
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
            "num_steps_per_env": train_cfg.get("num_steps_per_env", 24),
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": train_cfg.get("save_interval", 100),
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": train_cfg.get("seed", 1),
    }

    return train_cfg_dict


def load_config(config_path):
    """configから環境・観測・報酬・指令・学習設定を読み込む。"""
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


def make_env(env_name, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, device):
    env_module = importlib.import_module(env_name)
    env_class = getattr(env_module, env_name)
    return env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=device,
    )


def append_eval_csv(path, iteration, command_name, result):
    """学習中評価を後で表にしやすいCSVとして追記する。"""
    row = {"iteration": iteration, "command_name": command_name}
    row.update({key: value for key, value in result.items() if isinstance(value, (int, float, bool))})
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_eval_callback(
    eval_cfg,
    env_name,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    log_dir,
    device,
):
    """OnPolicyRunnerから呼ばれる学習中評価callbackを作る。"""
    if not eval_cfg.get("enabled", False):
        return None

    interval = int(eval_cfg.get("interval", 100))
    eval_num_envs = int(eval_cfg.get("num_envs", 16))
    duration_s = float(eval_cfg.get("duration_s", 10.0))
    success_min_clearance = float(eval_cfg.get("success_min_clearance", 0.16))
    save_csv = bool(eval_cfg.get("save_csv", True))
    csv_path = os.path.join(log_dir, "eval.csv")
    commands = eval_cfg.get(
        "commands",
        [
            {"name": "forward", "lin_vel_x": 0.3, "lin_vel_y": 0.0, "ang_vel": 0.0},
        ],
    )

    eval_reward_cfg = copy.deepcopy(reward_cfg)
    eval_reward_cfg["reward_scales"] = {}
    eval_env = make_env(
        env_name=env_name,
        num_envs=eval_num_envs,
        env_cfg=copy.deepcopy(env_cfg),
        obs_cfg=copy.deepcopy(obs_cfg),
        reward_cfg=eval_reward_cfg,
        command_cfg=copy.deepcopy(command_cfg),
        device=device,
    )

    def eval_callback(runner, iteration):
        if interval <= 0 or iteration % interval != 0:
            return {}

        policy = runner.get_inference_policy(device=device)
        metrics = {}
        try:
            for command_cfg_item in commands:
                name = command_cfg_item.get("name", "eval")
                command = build_command_values(
                    command_cfg=command_cfg,
                    reward_cfg=reward_cfg,
                    device=eval_env.device,
                    num_envs=eval_num_envs,
                    lin_vel_x=float(command_cfg_item.get("lin_vel_x", 0.0)),
                    lin_vel_y=float(command_cfg_item.get("lin_vel_y", 0.0)),
                    ang_vel=float(command_cfg_item.get("ang_vel", 0.0)),
                    base_height=command_cfg_item.get("base_height", None),
                    jump_height=float(command_cfg_item.get("jump_height", 0.0)),
                )
                result = evaluate_policy(
                    env=eval_env,
                    policy=policy,
                    command=command,
                    duration_s=duration_s,
                    success_min_distance_x=command_cfg_item.get("success_min_distance_x", None),
                    success_min_terrain_gain=command_cfg_item.get("success_min_terrain_gain", None),
                    success_min_clearance=success_min_clearance,
                )
                for key, value in result.items():
                    if isinstance(value, (int, float, bool)):
                        metrics[f"{name}/{key}"] = float(value)
                if save_csv:
                    append_eval_csv(csv_path, iteration, name, result)
        finally:
            runner.alg.actor_critic.train()
        return metrics

    return eval_callback


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
    train_cfg_full = get_train_cfg(exp_name, max_iterations, train_cfg)

    if effective_device == "cpu" and num_envs > 256:
        original_num_envs = num_envs
        num_envs = 256
        print(f"[go2_train] CPU execution detected. Reducing num_envs from {original_num_envs} to 256 for a usable training speed.")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 後から同じ条件を確認できるよう、学習時のconfigをログに保存する。
    shutil.copy(args.config, os.path.join(log_dir, "config.yaml"))

    # configのenv名から環境クラスを読み込む。例: Go2Env_Stair.py / Go2Env_Stair
    env = make_env(
        env_name=env_name,
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=effective_device,
    )
   

    runner = OnPolicyRunner(env, train_cfg_full, log_dir, device=effective_device)
    runner.eval_callback = make_eval_callback(
        eval_cfg=train_cfg.get("eval", {}),
        env_name=env_name,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        log_dir=log_dir,
        device=effective_device,
    )

    # 再開学習を使う場合は、ここでrunner.load(...)を呼ぶ。

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_full],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
