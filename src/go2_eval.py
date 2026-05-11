import argparse
import importlib
import os
import pickle
import time

import genesis as gs
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner


def resolve_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[go2_eval] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_run(exp_name, fallback_config):
    log_dir = f"logs/{exp_name}"
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"Training config not found: {cfgs_path}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))

    run_config_path = os.path.join(log_dir, "config.yaml")
    config_path = run_config_path if os.path.exists(run_config_path) else fallback_config
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Environment config not found. Expected {run_config_path} or fallback {fallback_config}."
        )

    yaml_config = load_yaml_config(config_path)
    env_name = yaml_config["train_cfg"]["env"]
    return log_dir, env_name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def make_env(env_name, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device):
    env_module = importlib.import_module(env_name)
    env_class = getattr(env_module, env_name)
    return env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        device=device,
    )


def build_command(args, command_cfg, reward_cfg, device, num_envs):
    num_commands = command_cfg["num_commands"]
    values = torch.zeros((num_envs, num_commands), device=device, dtype=gs.tc_float)
    values[:, 0] = args.lin_vel_x
    if num_commands > 1:
        values[:, 1] = args.lin_vel_y
    if num_commands > 2:
        values[:, 2] = args.ang_vel
    if num_commands > 3:
        values[:, 3] = args.base_height if args.base_height is not None else reward_cfg["base_height_target"]
    if num_commands > 4:
        values[:, 4] = args.jump_height
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("-B", "--num_envs", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lin-vel-x", type=float, default=0.3)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel", type=float, default=0.0)
    parser.add_argument("--base-height", type=float, default=None)
    parser.add_argument("--spawn-height-offset", type=float, default=None)
    parser.add_argument("--jump-height", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--hold-viewer-s", type=float, default=5.0)
    parser.add_argument("--keep-viewer-open", action="store_true")
    parser.add_argument("--auto-reset", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    backend = gs.constants.backend.gpu if device.startswith("cuda") else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir, env_name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_run(args.exp_name, args.config)
    if args.spawn_height_offset is not None:
        env_cfg["spawn_height_offset"] = args.spawn_height_offset
    if args.show_viewer and not args.auto_reset:
        env_cfg["auto_reset"] = False
    reward_cfg["reward_scales"] = {}

    num_envs = 1 if args.show_viewer else args.num_envs
    env = make_env(env_name, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, args.show_viewer, device)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    command = build_command(args, command_cfg, reward_cfg, env.device, num_envs)
    start_pos = env.base_pos.clone()
    max_steps = int(args.duration_s / env.dt)

    active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    done_steps = torch.full((num_envs,), max_steps, device=env.device, dtype=gs.tc_float)
    final_pos = env.base_pos.clone()
    vel_xy_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    vel_yaw_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)

    with torch.no_grad():
        for step in range(max_steps):
            env.commands[:] = command
            pre_step_pos = env.base_pos.clone()
            actions = policy(obs)
            active_before = active.clone()
            obs, _, _, dones, _ = env.step(actions, is_train=False)

            lin_vel_error = torch.sum(torch.square(command[:, :2] - env.base_lin_vel[:, :2]), dim=1)
            yaw_error = torch.square(command[:, 2] - env.base_ang_vel[:, 2])
            vel_xy_error_sum[active_before] += lin_vel_error[active_before] * env.dt
            vel_yaw_error_sum[active_before] += yaw_error[active_before] * env.dt

            newly_done = active_before & dones.bool()
            if newly_done.any():
                done_steps[newly_done] = step + 1
                final_pos[newly_done] = pre_step_pos[newly_done]
                active[newly_done] = False

            if args.real_time:
                time.sleep(env.dt)
            if not active.any():
                break

    if args.show_viewer and args.hold_viewer_s > 0.0:
        time.sleep(args.hold_viewer_s)

    duration = done_steps * env.dt
    if active.any():
        final_pos[active] = env.base_pos[active]
    distance = final_pos - start_pos
    vel_xy_error = vel_xy_error_sum / torch.clamp(duration, min=env.dt)
    vel_yaw_error = vel_yaw_error_sum / torch.clamp(duration, min=env.dt)
    survival_rate = active.float().mean().item()

    print("\nEvaluation summary")
    print(f"exp_name: {args.exp_name}")
    print(f"env: {env_name}")
    print(f"ckpt: {args.ckpt}")
    print(f"num_envs: {num_envs}")
    print(f"command: {command[0].detach().cpu().tolist()}")
    print(f"survival_rate: {survival_rate:.4f}")
    print(f"duration_mean_s: {duration.mean().item():.4f}")
    print(f"distance_x_mean_m: {distance[:, 0].mean().item():.4f}")
    print(f"distance_y_mean_m: {distance[:, 1].mean().item():.4f}")
    print(f"vel_xy_error_mean: {vel_xy_error.mean().item():.4f}")
    print(f"vel_yaw_error_mean: {vel_yaw_error.mean().item():.4f}")


if __name__ == "__main__":
    main()
