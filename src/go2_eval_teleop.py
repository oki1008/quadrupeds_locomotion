import argparse
import importlib
import os
import pickle

import genesis as gs
import numpy as np
import torch
import yaml
from pynput import keyboard
from rsl_rl.runners import OnPolicyRunner


lin_x = 0.0
lin_y = 0.0
ang_z = 0.0
base_height = 0.3
jump_height = 0.0
toggle_jump = False
stop = False
command_cfg_global = None


def clip_command_values():
    global lin_x, lin_y, ang_z, base_height, jump_height
    lin_x = float(np.clip(lin_x, *command_cfg_global["lin_vel_x_range"]))
    lin_y = float(np.clip(lin_y, *command_cfg_global["lin_vel_y_range"]))
    ang_z = float(np.clip(ang_z, *command_cfg_global["ang_vel_range"]))
    base_height = float(np.clip(base_height, *command_cfg_global["height_range"]))
    jump_height = float(np.clip(jump_height, *command_cfg_global["jump_range"]))


def print_command():
    jump_cmd = jump_height if toggle_jump else 0.0
    print(
        f"lin_x={lin_x:.2f}, lin_y={lin_y:.2f}, ang_z={ang_z:.2f}, "
        f"base_height={base_height:.2f}, jump={jump_cmd:.2f}"
    )


def on_press(key):
    global lin_x, lin_y, ang_z, base_height, jump_height, toggle_jump, stop
    try:
        if key.char == "w":
            lin_x += 0.1
        elif key.char == "s":
            lin_x -= 0.1
        elif key.char == "a":
            lin_y += 0.1
        elif key.char == "d":
            lin_y -= 0.1
        elif key.char == "q":
            ang_z += 0.1
        elif key.char == "e":
            ang_z -= 0.1
        elif key.char == "r":
            base_height += 0.02
        elif key.char == "f":
            base_height -= 0.02
        elif key.char == "j":
            toggle_jump = True
        elif key.char == "u":
            jump_height += 0.1
        elif key.char == "m":
            jump_height -= 0.1
        elif key.char == "x":
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0
        elif key.char == "8":
            stop = True

        clip_command_values()
        print_command()
    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:
        return False


def resolve_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[go2_eval_teleop] CUDA was requested but is not available. Falling back to CPU.")
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


def make_env(env_name, env_cfg, obs_cfg, reward_cfg, command_cfg, device, add_camera):
    env_module = importlib.import_module(env_name)
    env_class = getattr(env_module, env_name)
    return env_class(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=device,
        add_camera=add_camera,
    )


def make_command(command_cfg, device):
    values = torch.zeros((1, command_cfg["num_commands"]), device=device, dtype=gs.tc_float)
    values[:, 0] = lin_x
    if command_cfg["num_commands"] > 1:
        values[:, 1] = lin_y
    if command_cfg["num_commands"] > 2:
        values[:, 2] = ang_z
    if command_cfg["num_commands"] > 3:
        values[:, 3] = base_height
    if command_cfg["num_commands"] > 4:
        values[:, 4] = jump_height if toggle_jump else 0.0
    return values


def main():
    global base_height, jump_height, toggle_jump, stop, command_cfg_global

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--add-camera", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    backend = gs.constants.backend.gpu if device.startswith("cuda") else gs.constants.backend.cpu
    gs.init(logger_verbose_time=False, logging_level="warning", backend=backend)

    log_dir, env_name, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_run(args.exp_name, args.config)
    command_cfg_global = command_cfg
    reward_cfg["reward_scales"] = {}

    base_height = reward_cfg["base_height_target"]
    jump_height = command_cfg["jump_range"][1]
    clip_command_values()
    print(f"Loaded {env_name}. Keys: w/s forward, a/d lateral, q/e yaw, r/f height, j jump, x stop command, 8 quit.")
    print_command()

    env = make_env(env_name, env_cfg, obs_cfg, reward_cfg, command_cfg, device, args.add_camera)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    images_buffer = []
    commands_buffer = []
    reset_jump_iter = None
    step = 0

    with torch.no_grad():
        while not stop:
            env.commands[:] = make_command(command_cfg, env.device)
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions, is_train=False)

            if toggle_jump and reset_jump_iter is None:
                reset_jump_iter = step + 3
            if reset_jump_iter is not None and step >= reset_jump_iter:
                toggle_jump = False
                reset_jump_iter = None

            cam = getattr(env, "cam_0", None)
            if cam is not None and args.save_data:
                rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False)
                images_buffer.append(rgb)
                commands_buffer.append(make_command(command_cfg, env.device)[0].detach().cpu().numpy())

            if dones.any():
                obs, _ = env.reset()
            step += 1

    listener.stop()

    if args.save_data:
        pickle.dump(np.array(images_buffer), open("images_buffer.pkl", "wb"))
        pickle.dump(np.array(commands_buffer), open("commands_buffer.pkl", "wb"))


if __name__ == "__main__":
    main()
