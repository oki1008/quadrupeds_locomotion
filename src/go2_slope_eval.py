import argparse
import csv
import time
import os
import pickle

import genesis as gs
import numpy as np
import torch
from go2_slope_env import Go2SlopeEnv
from rsl_rl.runners import OnPolicyRunner


def resolve_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[go2_slope_eval] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def summarize(values):
    if values.numel() == 0:
        return 0.0, 0.0
    return values.mean().item(), values.std(unbiased=False).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--slope-angle-deg", type=float, default=None)
    parser.add_argument("--lin-vel-x", type=float, default=0.3)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel", type=float, default=0.0)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument("--hold-viewer-s", type=float, default=5.0)
    parser.add_argument("--eval-min-height", type=float, default=0.18)
    parser.add_argument("--success-vel-xy-error", type=float, default=0.06)
    parser.add_argument("--success-min-distance-x", type=float, default=1.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    backend = gs.constants.backend.gpu if device.startswith("cuda") else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    if args.slope_angle_deg is not None:
        env_cfg["slope_angle_deg"] = args.slope_angle_deg
    env_cfg["termination_min_height"] = max(env_cfg.get("termination_min_height", 0.0), args.eval_min_height)
    reward_cfg["reward_scales"] = {}

    env = Go2SlopeEnv(
        num_envs=1 if args.show_viewer else args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
        device=device,
    )
    num_envs = 1 if args.show_viewer else args.num_envs

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    start_x = env.base_pos[:, 0].clone()
    command = torch.tensor(
        [args.lin_vel_x, args.lin_vel_y, args.ang_vel],
        device=env.device,
        dtype=gs.tc_float,
    ).repeat(num_envs, 1)

    active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    done_steps = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    distance_x = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    vel_xy_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    world_x_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    vel_yaw_error_sum = torch.zeros(num_envs, device=env.device, dtype=gs.tc_float)
    reason_flags = {
        "timeout": torch.zeros(num_envs, device=env.device, dtype=torch.bool),
        "roll": torch.zeros(num_envs, device=env.device, dtype=torch.bool),
        "pitch": torch.zeros(num_envs, device=env.device, dtype=torch.bool),
        "low_height": torch.zeros(num_envs, device=env.device, dtype=torch.bool),
        "high_height": torch.zeros(num_envs, device=env.device, dtype=torch.bool),
    }

    with torch.no_grad():
        for step in range(env.max_episode_length + 5):
            env.commands[:] = command
            if args.show_viewer and args.follow_camera:
                lookat = env.base_pos[0].detach().cpu().numpy()
                camera_pos = np.array([lookat[0] - 2.5, lookat[1] - 2.0, lookat[2] + 1.2])
                camera_lookat = np.array([lookat[0], lookat[1], lookat[2] + 0.1])
                env.scene.viewer.set_camera_pose(
                    pos=camera_pos,
                    lookat=camera_lookat,
                )
            actions = policy(obs)

            active_before = active.clone()
            obs, _, _, dones, _ = env.step(actions, is_train=False)
            if args.real_time:
                time.sleep(env.dt)

            lin_vel_error = torch.sum(torch.square(command[:, :2] - env.base_lin_vel[:, :2]), dim=1)
            world_x_error = torch.square(command[:, 0] - env.base_world_lin_vel[:, 0])
            yaw_error = torch.square(command[:, 2] - env.base_ang_vel[:, 2])
            vel_xy_error_sum[active_before] += lin_vel_error[active_before] * env.dt
            world_x_error_sum[active_before] += world_x_error[active_before] * env.dt
            vel_yaw_error_sum[active_before] += yaw_error[active_before] * env.dt

            newly_done = active_before & dones.bool()
            if newly_done.any():
                done_steps[newly_done] = step + 1
                distance_x[newly_done] = env.last_pre_reset_base_pos[newly_done, 0] - start_x[newly_done]
                for key, flags in reason_flags.items():
                    flags[newly_done] = env.last_term_flags[key][newly_done]
                active[newly_done] = False

            if not active.any():
                break

    if args.show_viewer and args.hold_viewer_s > 0.0:
        time.sleep(args.hold_viewer_s)

    unfinished = active
    if unfinished.any():
        done_steps[unfinished] = env.max_episode_length
        distance_x[unfinished] = env.base_pos[unfinished, 0] - start_x[unfinished]
        reason_flags["timeout"][unfinished] = True

    duration = done_steps * env.dt
    vel_xy_error = vel_xy_error_sum / torch.clamp(duration, min=env.dt)
    world_x_error = world_x_error_sum / torch.clamp(duration, min=env.dt)
    vel_yaw_error = vel_yaw_error_sum / torch.clamp(duration, min=env.dt)

    duration_mean, duration_std = summarize(duration)
    distance_mean, distance_std = summarize(distance_x)
    vel_xy_mean, vel_xy_std = summarize(vel_xy_error)
    world_x_mean, world_x_std = summarize(world_x_error)
    vel_yaw_mean, vel_yaw_std = summarize(vel_yaw_error)
    task_success = (
        reason_flags["timeout"]
        & (world_x_error < args.success_vel_xy_error)
        & (distance_x > args.success_min_distance_x)
    )
    survival_rate = reason_flags["timeout"].float().mean().item()
    success_rate = task_success.float().mean().item()

    result = {
        "exp_name": args.exp_name,
        "ckpt": args.ckpt,
        "slope_angle_deg": env_cfg["slope_angle_deg"],
        "use_height_obs": obs_cfg.get("use_height_obs", False),
        "num_envs": num_envs,
        "command_lin_vel_x": args.lin_vel_x,
        "survival_rate": survival_rate,
        "success_rate": success_rate,
        "success_vel_xy_error_threshold": args.success_vel_xy_error,
        "success_min_distance_x": args.success_min_distance_x,
        "duration_mean_s": duration_mean,
        "duration_std_s": duration_std,
        "distance_x_mean_m": distance_mean,
        "distance_x_std_m": distance_std,
        "vel_xy_error_mean": vel_xy_mean,
        "vel_xy_error_std": vel_xy_std,
        "world_x_error_mean": world_x_mean,
        "world_x_error_std": world_x_std,
        "vel_yaw_error_mean": vel_yaw_mean,
        "vel_yaw_error_std": vel_yaw_std,
    }
    for key, flags in reason_flags.items():
        result[f"term_{key}_rate"] = flags.float().mean().item()

    print("\nEvaluation summary")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"\nSaved CSV row to {args.csv}")


if __name__ == "__main__":
    main()
