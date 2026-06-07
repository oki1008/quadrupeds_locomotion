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
    return build_command_values(
        command_cfg=command_cfg,
        reward_cfg=reward_cfg,
        device=device,
        num_envs=num_envs,
        lin_vel_x=args.lin_vel_x,
        lin_vel_y=args.lin_vel_y,
        ang_vel=args.ang_vel,
        base_height=args.base_height,
        jump_height=args.jump_height,
    )


def build_command_values(
    command_cfg,
    reward_cfg,
    device,
    num_envs,
    lin_vel_x=0.3,
    lin_vel_y=0.0,
    ang_vel=0.0,
    base_height=None,
    jump_height=0.0,
):
    """評価用の一定commandを作る。"""
    num_commands = command_cfg["num_commands"]
    values = torch.zeros((num_envs, num_commands), device=device, dtype=gs.tc_float)
    values[:, 0] = lin_vel_x
    if num_commands > 1:
        values[:, 1] = lin_vel_y
    if num_commands > 2:
        values[:, 2] = ang_vel
    if num_commands > 3:
        values[:, 3] = base_height if base_height is not None else reward_cfg["base_height_target"]
    if num_commands > 4:
        values[:, 4] = jump_height
    return values


def _mean(tensor):
    return tensor.mean().item()


def evaluate_policy(
    env,
    policy,
    command,
    duration_s=20.0,
    success_min_distance_x=None,
    success_min_terrain_gain=None,
    success_min_clearance=0.16,
    real_time=False,
    show_viewer=False,
    hold_viewer_s=0.0,
):
    """現在のpolicyを一定commandで評価し、数値指標をdictで返す。"""
    obs, _ = env.reset()
    env_cfg = env.env_cfg
    num_envs = env.num_envs
    lin_vel_x = command[0, 0].item()
    lin_vel_y = command[0, 1].item() if command.shape[1] > 1 else 0.0
    ang_vel = command[0, 2].item() if command.shape[1] > 2 else 0.0

    start_pos = env.base_pos.clone()
    start_terrain_height = env._terrain_height_at(start_pos[:, 0], start_pos[:, 1])
    max_steps = int(duration_s / env.dt)

    active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    done_steps = torch.full((num_envs,), max_steps, device=env.device, dtype=gs.tc_float)
    final_pos = env.base_pos.clone()
    vel_xy_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    vel_yaw_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    has_wheel_metrics = all(
        hasattr(env, name)
        for name in ("wheel_dof_vel", "last_contacts", "base_lin_vel")
    )
    has_power_metrics = all(
        hasattr(env, name)
        for name in ("motor_dof_force", "dof_vel")
    )
    wheel_radius = env_cfg.get("wheel_radius", 0.055)
    if has_wheel_metrics:
        wheel_abs_vel_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
        contact_wheel_abs_vel_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
        wheel_ground_speed_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
        rolling_to_body_speed_ratio_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
        wheel_contact_ratio_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
        contact_leg_forward_penalty_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    if has_power_metrics:
        joint_power_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
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
                if show_viewer and "Viewer closed" in str(exc):
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
            if has_wheel_metrics:
                wheel_abs_vel = torch.mean(torch.abs(env.wheel_dof_vel), dim=1)
                contact_mask = env.last_contacts.float()
                contact_count = torch.clamp(torch.sum(contact_mask, dim=1), min=1.0)
                contact_wheel_abs_vel = torch.sum(torch.abs(env.wheel_dof_vel) * contact_mask, dim=1) / contact_count
                wheel_ground_speed = wheel_abs_vel * wheel_radius
                body_xy_speed = torch.norm(env.base_lin_vel[:, :2], dim=1)
                rolling_to_body_speed_ratio = wheel_ground_speed / torch.clamp(body_xy_speed, min=1e-4)
                wheel_contact_ratio = torch.mean(contact_mask, dim=1)

                wheel_abs_vel_sum[active_before] += wheel_abs_vel[active_before] * env.dt
                contact_wheel_abs_vel_sum[active_before] += contact_wheel_abs_vel[active_before] * env.dt
                wheel_ground_speed_sum[active_before] += wheel_ground_speed[active_before] * env.dt
                rolling_to_body_speed_ratio_sum[active_before] += rolling_to_body_speed_ratio[active_before] * env.dt
                wheel_contact_ratio_sum[active_before] += wheel_contact_ratio[active_before] * env.dt
                if hasattr(env, "_reward_contact_leg_forward_vel_penalty"):
                    contact_penalty = env._reward_contact_leg_forward_vel_penalty()
                    contact_leg_forward_penalty_sum[active_before] += contact_penalty[active_before] * env.dt
            if has_power_metrics:
                joint_power = torch.sum(torch.abs(env.motor_dof_force * env.dof_vel), dim=1)
                joint_power_sum[active_before] += joint_power[active_before] * env.dt

            newly_done = active_before & dones.bool()
            if newly_done.any():
                done_steps[newly_done] = step + 1
                final_pos[newly_done] = pre_step_pos[newly_done]
                for key in term_sums:
                    term_key = "term_" + key
                    if term_key in extras:
                        term_sums[key][newly_done] = extras[term_key][newly_done]
                active[newly_done] = False

            if real_time:
                time.sleep(env.dt)
            if not active.any():
                break

    if show_viewer and hold_viewer_s > 0.0 and not viewer_closed:
        time.sleep(hold_viewer_s)

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
    if has_wheel_metrics:
        wheel_abs_vel_mean = wheel_abs_vel_sum / torch.clamp(duration, min=env.dt)
        contact_wheel_abs_vel_mean = contact_wheel_abs_vel_sum / torch.clamp(duration, min=env.dt)
        wheel_ground_speed_mean = wheel_ground_speed_sum / torch.clamp(duration, min=env.dt)
        rolling_to_body_speed_ratio_mean = rolling_to_body_speed_ratio_sum / torch.clamp(duration, min=env.dt)
        wheel_contact_ratio_mean = wheel_contact_ratio_sum / torch.clamp(duration, min=env.dt)
        contact_leg_forward_penalty_mean = contact_leg_forward_penalty_sum / torch.clamp(duration, min=env.dt)
    if has_power_metrics:
        joint_power_mean = joint_power_sum / torch.clamp(duration, min=env.dt)
    failure = (
        (term_sums["roll"] > 0.0)
        | (term_sums["pitch"] > 0.0)
        | (term_sums["low_height"] > 0.0)
        | (term_sums["high_height"] > 0.0)
    )
    survival_rate = (~failure).float().mean().item()
    success_min_distance_x = (
        success_min_distance_x
        if success_min_distance_x is not None
        else lin_vel_x * duration_s * 0.8
    )
    success_min_terrain_gain = (
        success_min_terrain_gain
        if success_min_terrain_gain is not None
        else (0.0 if env_cfg.get("terrain_type") != "stair" else env_cfg.get("terrain_step_height", 0.0) * 3.0)
    )
    target_distance_x = lin_vel_x * duration_s
    target_distance_y = lin_vel_y * duration_s
    target_yaw = ang_vel * duration_s
    abs_cmd_x = abs(lin_vel_x)
    abs_cmd_y = abs(lin_vel_y)
    abs_cmd_yaw = abs(ang_vel)

    if abs_cmd_yaw > max(abs_cmd_x, abs_cmd_y) and abs_cmd_yaw > 1e-6:
        motion_type = "yaw"
        target_yaw_min = abs(target_yaw) * 0.6
        motion_success = (
            active
            & (torch.abs(final_euler[:, 2]) >= target_yaw_min)
            & (torch.abs(distance[:, 0]) <= max(0.5, abs(target_yaw) * 0.15))
            & (torch.abs(distance[:, 1]) <= max(0.5, abs(target_yaw) * 0.15))
            & (base_clearance >= success_min_clearance)
        )
    elif abs_cmd_y > abs_cmd_x and abs_cmd_y > 1e-6:
        motion_type = "lateral_positive" if lin_vel_y > 0.0 else "lateral_negative"
        target_distance_y_min = abs(target_distance_y) * 0.8
        motion_success = (
            active
            & ((distance[:, 1] * (1.0 if lin_vel_y > 0.0 else -1.0)) >= target_distance_y_min)
            & (torch.abs(distance[:, 0]) <= max(0.4, target_distance_y_min * 0.25))
            & (base_clearance >= success_min_clearance)
        )
    elif abs_cmd_x > 1e-6:
        motion_type = "forward" if lin_vel_x > 0.0 else "backward"
        target_distance_x_min = abs(target_distance_x) * 0.8
        motion_success = (
            active
            & ((distance[:, 0] * (1.0 if lin_vel_x > 0.0 else -1.0)) >= target_distance_x_min)
            & (torch.abs(distance[:, 1]) <= max(0.25, target_distance_x_min * 0.15))
            & (terrain_height_gain >= success_min_terrain_gain)
            & (base_clearance >= success_min_clearance)
        )
    else:
        motion_type = "stand"
        motion_success = (
            active
            & (torch.norm(distance[:, :2], dim=1) <= 0.25)
            & (base_clearance >= success_min_clearance)
        )
    motion_success_rate = motion_success.float().mean().item()

    strict_success = (
        active
        & (distance[:, 0] >= success_min_distance_x)
        & (torch.abs(distance[:, 1]) <= 0.25)
        & (terrain_height_gain >= success_min_terrain_gain)
        & (base_clearance >= success_min_clearance)
    )
    strict_success_rate = strict_success.float().mean().item()

    result = {
        "terrain_type": env_cfg.get("terrain_type", "unknown"),
        "num_envs": num_envs,
        "command": command[0].detach().cpu().tolist(),
        "motion_type": motion_type,
        "survival_rate": survival_rate,
        "duration_mean_s": _mean(duration),
        "distance_x_mean_m": _mean(distance[:, 0]),
        "distance_y_mean_m": _mean(distance[:, 1]),
        "distance_z_mean_m": _mean(distance[:, 2]),
        "terrain_height_gain_mean_m": _mean(terrain_height_gain),
        "terrain_height_gain_max_m": terrain_height_gain.max().item(),
        "terrain_height_gain_min_m": terrain_height_gain.min().item(),
        "final_base_height_mean_m": _mean(final_pos[:, 2]),
        "final_base_clearance_mean_m": _mean(base_clearance),
        "final_base_clearance_min_m": base_clearance.min().item(),
        "final_abs_roll_mean_rad": _mean(torch.abs(final_euler[:, 0])),
        "final_abs_pitch_mean_rad": _mean(torch.abs(final_euler[:, 1])),
        "final_abs_yaw_mean_rad": _mean(torch.abs(final_euler[:, 2])),
        "vel_xy_error_mean": _mean(vel_xy_error),
        "vel_yaw_error_mean": _mean(vel_yaw_error),
        "motion_success_rate": motion_success_rate,
        "target_distance_x_m": target_distance_x,
        "target_distance_y_m": target_distance_y,
        "target_yaw_rad": target_yaw,
        "strict_success_rate": strict_success_rate,
        "success_min_distance_x": success_min_distance_x,
        "success_min_terrain_gain": success_min_terrain_gain,
        "success_min_clearance": success_min_clearance,
        "viewer_closed": viewer_closed,
    }
    if has_wheel_metrics:
        result.update(
            {
                "wheel_abs_vel_mean_rad_s": _mean(wheel_abs_vel_mean),
                "contact_wheel_abs_vel_mean_rad_s": _mean(contact_wheel_abs_vel_mean),
                "wheel_ground_speed_mean_m_s": _mean(wheel_ground_speed_mean),
                "rolling_to_body_speed_ratio_mean": _mean(rolling_to_body_speed_ratio_mean),
                "wheel_contact_ratio_mean": _mean(wheel_contact_ratio_mean),
                "contact_leg_forward_penalty_raw_mean": _mean(contact_leg_forward_penalty_mean),
            }
        )
    if has_power_metrics:
        result["joint_power_mean"] = _mean(joint_power_mean)
    if env_cfg.get("terrain_type") == "stair":
        result["terrain_step_height"] = env_cfg.get("terrain_step_height")
        result["terrain_step_width"] = env_cfg.get("terrain_step_width")
    for key, values in term_sums.items():
        result[f"term_{key}_rate"] = values.mean().item()
    return result


def print_evaluation_summary(result, exp_name, env_name, ckpt):
    print("\nEvaluation summary")
    print(f"exp_name: {exp_name}")
    print(f"env: {env_name}")
    print(f"ckpt: {ckpt}")
    ordered_keys = [
        "terrain_type",
        "terrain_step_height",
        "terrain_step_width",
        "num_envs",
        "command",
        "motion_type",
        "survival_rate",
        "duration_mean_s",
        "distance_x_mean_m",
        "distance_y_mean_m",
        "distance_z_mean_m",
        "terrain_height_gain_mean_m",
        "terrain_height_gain_max_m",
        "terrain_height_gain_min_m",
        "final_base_height_mean_m",
        "final_base_clearance_mean_m",
        "final_base_clearance_min_m",
        "final_abs_roll_mean_rad",
        "final_abs_pitch_mean_rad",
        "final_abs_yaw_mean_rad",
        "vel_xy_error_mean",
        "vel_yaw_error_mean",
        "wheel_abs_vel_mean_rad_s",
        "contact_wheel_abs_vel_mean_rad_s",
        "wheel_ground_speed_mean_m_s",
        "rolling_to_body_speed_ratio_mean",
        "wheel_contact_ratio_mean",
        "contact_leg_forward_penalty_raw_mean",
        "joint_power_mean",
        "term_roll_rate",
        "term_pitch_rate",
        "term_low_height_rate",
        "term_high_height_rate",
        "term_timeout_rate",
        "motion_success_rate",
        "target_distance_x_m",
        "target_distance_y_m",
        "target_yaw_rad",
        "strict_success_rate",
        "success_min_distance_x",
        "success_min_terrain_gain",
        "success_min_clearance",
    ]
    for key in ordered_keys:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    if result.get("viewer_closed"):
        print("viewer_closed: true")


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
    # CUDAで保存したモデルをCPU評価する場合でも読み込めるようにする。
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

    command = build_command(args, command_cfg, reward_cfg, env.device, num_envs)
    result = evaluate_policy(
        env=env,
        policy=policy,
        command=command,
        duration_s=args.duration_s,
        success_min_distance_x=args.success_min_distance_x,
        success_min_terrain_gain=args.success_min_terrain_gain,
        success_min_clearance=args.success_min_clearance,
        real_time=args.real_time,
        show_viewer=args.show_viewer,
        hold_viewer_s=args.hold_viewer_s,
    )
    print_evaluation_summary(result, args.exp_name, env_name, args.ckpt)


if __name__ == "__main__":
    main()
