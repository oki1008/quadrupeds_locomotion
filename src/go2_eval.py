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
    parser.add_argument("--terrain-type", type=str, default=None, choices=["flat", "stair", "rough"])
    parser.add_argument("--terrain-step-height", type=float, default=None)
    parser.add_argument("--terrain-step-width", type=float, default=None)
    parser.add_argument("--stair-start-x", type=float, default=None)
    parser.add_argument("--stair-end-x", type=float, default=None)
    parser.add_argument("--success-min-distance-x", type=float, default=None)
    parser.add_argument("--success-min-terrain-gain", type=float, default=None)
    parser.add_argument("--success-min-clearance", type=float, default=0.16)
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
    if args.terrain_type is not None:
        env_cfg["terrain_type"] = args.terrain_type
    if args.terrain_step_height is not None:
        env_cfg["terrain_step_height"] = args.terrain_step_height
    if args.terrain_step_width is not None:
        env_cfg["terrain_step_width"] = args.terrain_step_width
    if args.stair_start_x is not None:
        env_cfg["stair_start_x"] = args.stair_start_x
    if args.stair_end_x is not None:
        env_cfg["stair_end_x"] = args.stair_end_x
    if args.show_viewer and not args.auto_reset:
        env_cfg["auto_reset"] = False
    reward_cfg["reward_scales"] = {}

    num_envs = 1 if args.show_viewer else args.num_envs
    env = make_env(env_name, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, args.show_viewer, device)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # rsl_rl internally calls torch.load without map_location. If CUDA is
    # temporarily invisible, this keeps CPU evaluation from crashing on a
    # checkpoint that was saved from CUDA training.
    original_torch_load = torch.load
    if device == "cpu":
        def torch_load_on_cpu(path, *load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(path, *load_args, **load_kwargs)

        torch.load = torch_load_on_cpu
    try:
        runner.load(resume_path, load_optimizer=False)
    finally:
        torch.load = original_torch_load
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    command = build_command(args, command_cfg, reward_cfg, env.device, num_envs)
    start_pos = env.base_pos.clone()
    start_terrain_height = env._terrain_height_at(start_pos[:, 0], start_pos[:, 1])
    max_steps = int(args.duration_s / env.dt)

    active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    done_steps = torch.full((num_envs,), max_steps, device=env.device, dtype=gs.tc_float)
    final_pos = env.base_pos.clone()
    vel_xy_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    vel_yaw_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    term_sums = {
        "roll": torch.zeros(num_envs, device=env.device, dtype=gs.tc_float),
        "pitch": torch.zeros(num_envs, device=env.device, dtype=gs.tc_float),
        "low_height": torch.zeros(num_envs, device=env.device, dtype=gs.tc_float),
        "high_height": torch.zeros(num_envs, device=env.device, dtype=gs.tc_float),
        "timeout": torch.zeros(num_envs, device=env.device, dtype=gs.tc_float),
    }
    viewer_closed = False

    with torch.no_grad():
        for step in range(max_steps):
            env.commands[:] = command
            pre_step_pos = env.base_pos.clone()
            actions = policy(obs)
            active_before = active.clone()
            try:
                obs, _, _, dones, extras = env.step(actions, is_train=False)
            except gs.GenesisException as exc:
                if args.show_viewer and "Viewer closed" in str(exc):
                    viewer_closed = True
                    final_pos[active_before] = pre_step_pos[active_before]
                    done_steps[active_before] = step
                    active[active_before] = False
                    break
                raise

            lin_vel_error = torch.sum(torch.square(command[:, :2] - env.base_lin_vel[:, :2]), dim=1)
            yaw_error = torch.square(command[:, 2] - env.base_ang_vel[:, 2])
            vel_xy_error_sum[active_before] += lin_vel_error[active_before] * env.dt
            vel_yaw_error_sum[active_before] += yaw_error[active_before] * env.dt

            newly_done = active_before & dones.bool()
            if newly_done.any():
                done_steps[newly_done] = step + 1
                final_pos[newly_done] = pre_step_pos[newly_done]
                for key in term_sums:
                    term_key = "term_" + key
                    if term_key in extras:
                        term_sums[key][newly_done] = extras[term_key][newly_done]
                active[newly_done] = False

            if args.real_time:
                time.sleep(env.dt)
            if not active.any():
                break

    if args.show_viewer and args.hold_viewer_s > 0.0 and not viewer_closed:
        time.sleep(args.hold_viewer_s)

    duration = done_steps * env.dt
    if active.any():
        final_pos[active] = env.base_pos[active]
    final_euler = env.base_euler.clone()
    distance = final_pos - start_pos
    final_terrain_height = env._terrain_height_at(final_pos[:, 0], final_pos[:, 1])
    terrain_height_gain = final_terrain_height - start_terrain_height
    base_clearance = final_pos[:, 2] - final_terrain_height
    vel_xy_error = vel_xy_error_sum / torch.clamp(duration, min=env.dt)
    vel_yaw_error = vel_yaw_error_sum / torch.clamp(duration, min=env.dt)
    failure = (
        (term_sums["roll"] > 0.0)
        | (term_sums["pitch"] > 0.0)
        | (term_sums["low_height"] > 0.0)
        | (term_sums["high_height"] > 0.0)
    )
    survival_rate = (~failure).float().mean().item()
    success_min_distance_x = (
        args.success_min_distance_x
        if args.success_min_distance_x is not None
        else args.lin_vel_x * args.duration_s * 0.8
    )
    success_min_terrain_gain = (
        args.success_min_terrain_gain
        if args.success_min_terrain_gain is not None
        else (0.0 if env_cfg.get("terrain_type") != "stair" else env_cfg.get("terrain_step_height", 0.0) * 3.0)
    )
    strict_success = (
        active
        & (distance[:, 0] >= success_min_distance_x)
        & (torch.abs(distance[:, 1]) <= 0.25)
        & (terrain_height_gain >= success_min_terrain_gain)
        & (base_clearance >= args.success_min_clearance)
    )
    strict_success_rate = strict_success.float().mean().item()

    print("\nEvaluation summary")
    print(f"exp_name: {args.exp_name}")
    print(f"env: {env_name}")
    print(f"ckpt: {args.ckpt}")
    print(f"terrain_type: {env_cfg.get('terrain_type', 'unknown')}")
    if env_cfg.get("terrain_type") == "stair":
        print(f"terrain_step_height: {env_cfg.get('terrain_step_height')}")
        print(f"terrain_step_width: {env_cfg.get('terrain_step_width')}")
    print(f"num_envs: {num_envs}")
    print(f"command: {command[0].detach().cpu().tolist()}")
    print(f"survival_rate: {survival_rate:.4f}")
    print(f"duration_mean_s: {duration.mean().item():.4f}")
    print(f"distance_x_mean_m: {distance[:, 0].mean().item():.4f}")
    print(f"distance_y_mean_m: {distance[:, 1].mean().item():.4f}")
    print(f"distance_z_mean_m: {distance[:, 2].mean().item():.4f}")
    print(f"terrain_height_gain_mean_m: {terrain_height_gain.mean().item():.4f}")
    print(f"terrain_height_gain_max_m: {terrain_height_gain.max().item():.4f}")
    print(f"terrain_height_gain_min_m: {terrain_height_gain.min().item():.4f}")
    print(f"final_base_height_mean_m: {final_pos[:, 2].mean().item():.4f}")
    print(f"final_base_clearance_mean_m: {base_clearance.mean().item():.4f}")
    print(f"final_base_clearance_min_m: {base_clearance.min().item():.4f}")
    print(f"final_abs_roll_mean_rad: {torch.abs(final_euler[:, 0]).mean().item():.4f}")
    print(f"final_abs_pitch_mean_rad: {torch.abs(final_euler[:, 1]).mean().item():.4f}")
    print(f"final_abs_yaw_mean_rad: {torch.abs(final_euler[:, 2]).mean().item():.4f}")
    print(f"vel_xy_error_mean: {vel_xy_error.mean().item():.4f}")
    print(f"vel_yaw_error_mean: {vel_yaw_error.mean().item():.4f}")
    for key, values in term_sums.items():
        print(f"term_{key}_rate: {values.mean().item():.4f}")
    print(f"strict_success_rate: {strict_success_rate:.4f}")
    print(f"success_min_distance_x: {success_min_distance_x:.4f}")
    print(f"success_min_terrain_gain: {success_min_terrain_gain:.4f}")
    print(f"success_min_clearance: {args.success_min_clearance:.4f}")
    if viewer_closed:
        print("viewer_closed: true")


if __name__ == "__main__":
    main()
