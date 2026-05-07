import argparse
import os
import pickle
import shutil

import genesis as gs
import torch
from go2_slope_env import Go2SlopeEnv
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    return {
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
            "init_noise_std": 0.4,
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


def get_cfgs(use_height_obs, slope_angle_deg):
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 0.7,
        "termination_if_pitch_greater_than": 0.7,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.20,
        "simulate_action_latency": True,
        "clip_actions": 2.0,
        "target_dof_pos_clip": 0.8,
        "contact_force_threshold": 0.1,
        "sim_substeps": 4,
        "slope_angle_deg": slope_angle_deg,
        "terrain_width": 24.0,
        "terrain_length": 24.0,
        "horizontal_scale": 0.05,
        "spawn_x_range": [-4.0, -2.0],
        "spawn_y_range": [-3.0, 3.0],
        "spawn_height_offset": 0.28,
        "termination_min_height": 0.18,
        "termination_max_height": 1.2,
        "terminate_on_height": True,
    }
    obs_cfg = {
        "use_height_obs": use_height_obs,
        "num_history": 3,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.20,
        "base_height_target": 0.28,
        "max_feet_air_time": 0.30,
        "reward_scales": {
            "alive": 0.05,
            "tracking_lin_vel": 2.0,
            "tracking_world_lin_vel_x": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -0.6,
            "orientation": -4.0,
            "base_height": -5.0,
            "termination": -20.0,
            "action_rate": -0.005,
            "action_magnitude": -0.002,
            "dof_vel": -0.0001,
            "no_contact": -0.05,
            "feet_air_time_penalty": -0.1,
            "similar_to_default": -0.01,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.4],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def resolve_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[go2_slope_train] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_slope_blind")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--slope-angle-deg", type=float, default=6.0)
    parser.add_argument("--use-height-obs", action="store_true")
    args = parser.parse_args()

    effective_device = resolve_device(args.device)
    backend = gs.constants.backend.gpu if effective_device.startswith("cuda") else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    if effective_device == "cpu" and args.num_envs == parser.get_default("num_envs"):
        args.num_envs = 256
        print("[go2_slope_train] CPU execution detected. Reducing num_envs from 4096 to 256.")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.use_height_obs, args.slope_angle_deg)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = Go2SlopeEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=effective_device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=effective_device)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
