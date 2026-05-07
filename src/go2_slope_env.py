import math

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2SlopeEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        device="cuda:0",
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.dt = 0.02
        self.sim_substeps = env_cfg.get("sim_substeps", 4)
        self.rigid_dt = self.dt / self.sim_substeps
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.use_height_obs = obs_cfg.get("use_height_obs", False)
        self.num_history = obs_cfg.get("num_history", 3)
        self.num_height_points = 0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, -3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=min(num_envs, 4), show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.rigid_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self._build_slope_terrain()

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        slope_pitch = -math.radians(self.env_cfg.get("slope_angle_deg", 0.0))
        self.slope_init_quat = torch.tensor(
            [math.cos(slope_pitch / 2.0), 0.0, math.sin(slope_pitch / 2.0), 0.0],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.feet_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        link_names = [link.name for link in self.robot.links]
        self.feet_indices = torch.tensor(
            [link_names.index(name) for name in self.feet_names],
            device=self.device,
            dtype=torch.long,
        )
        self.feet_air_time = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.feet_contacts = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.episode_error_sums = {
            "vel_xy": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "vel_yaw": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
        }

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_world_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_pre_reset_base_pos = torch.zeros_like(self.base_pos)
        self.extras = {}
        self.episode_term_sums = {
            "roll": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "pitch": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "low_height": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "high_height": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "timeout": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
        }
        self.last_term_flags = {
            key: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for key in self.episode_term_sums.keys()
        }

        self.raw_obs_dim = 45 + (self.num_height_points if self.use_height_obs else 0)
        self.num_obs = self.raw_obs_dim * self.num_history
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_history, self.raw_obs_dim),
            device=self.device,
            dtype=gs.tc_float,
        )

    def _build_slope_terrain(self):
        terrain_width = self.env_cfg.get("terrain_width", 24.0)
        terrain_length = self.env_cfg.get("terrain_length", 24.0)
        self.horizontal_scale = self.env_cfg.get("horizontal_scale", 0.05)
        self.n_rows = int(terrain_width / self.horizontal_scale)
        self.n_cols = int(terrain_length / self.horizontal_scale)

        self.slope_angle = math.radians(self.env_cfg.get("slope_angle_deg", 6.0))
        self.slope_tan = math.tan(self.slope_angle)
        if abs(self.slope_angle) < 1.0e-6:
            self.terrain = self.scene.add_entity(gs.morphs.Plane())
        else:
            thickness = 0.1
            self.terrain = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0.0, 0.0, -math.cos(self.slope_angle) * thickness / 2.0),
                    euler=(0.0, -math.degrees(self.slope_angle), 0.0),
                    size=(terrain_length, terrain_width, thickness),
                    fixed=True,
                )
            )

        measure_x = torch.linspace(-0.8, 0.8, 11, device=self.device)
        measure_y = torch.linspace(-0.5, 0.5, 11, device=self.device)
        scan_x, scan_y = torch.meshgrid(measure_x, measure_y, indexing="ij")
        self.num_height_points = scan_x.numel()
        self.local_height_points = torch.zeros((self.num_envs, self.num_height_points, 3), device=self.device)
        self.local_height_points[:, :, 0] = scan_x.flatten()
        self.local_height_points[:, :, 1] = scan_y.flatten()

    def _terrain_height_at(self, x, y):
        del y
        return torch.as_tensor(self.slope_tan, device=self.device, dtype=gs.tc_float) * x

    def _get_heights(self):
        yaw = self.base_euler[:, 2]
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        x = (
            self.local_height_points[:, :, 0] * cos_yaw
            - self.local_height_points[:, :, 1] * sin_yaw
            + self.base_pos[:, 0].unsqueeze(1)
        )
        y = (
            self.local_height_points[:, :, 0] * sin_yaw
            + self.local_height_points[:, :, 1] * cos_yaw
            + self.base_pos[:, 1].unsqueeze(1)
        )
        heights = self._terrain_height_at(x, y)
        target_height = self.reward_cfg["base_height_target"]
        clearance_error = self.base_pos[:, 2].unsqueeze(1) - heights - target_height
        return torch.clip(clearance_error * self.obs_scales["height_measurements"], -1.0, 1.0)

    def _sample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions, is_train=True):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        joint_clip = self.env_cfg.get("target_dof_pos_clip", 0.8)
        target_dof_pos = torch.clip(target_dof_pos, self.default_dof_pos - joint_clip, self.default_dof_pos + joint_clip)
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_world_lin_vel[:] = self.robot.get_vel()
        self.base_lin_vel[:] = transform_by_quat(self.base_world_lin_vel, inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        forces = self.robot.get_links_net_contact_force()[:, self.feet_indices, :]
        self.feet_contacts = torch.norm(forces, dim=-1) > self.env_cfg.get("contact_force_threshold", 0.1)
        self.feet_air_time += self.dt
        self.feet_air_time[self.feet_contacts] = 0.0

        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if is_train:
            self._sample_commands(envs_idx)

        ground_height = self._terrain_height_at(self.base_pos[:, 0], self.base_pos[:, 1])
        local_height = self.base_pos[:, 2] - ground_height
        timeout = self.episode_length_buf > self.max_episode_length
        term_pitch = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        term_roll = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        term_low_height = local_height < self.env_cfg.get("termination_min_height", 0.12)
        term_high_height = local_height > self.env_cfg.get("termination_max_height", 1.2)
        self.non_timeout_reset = term_pitch | term_roll | term_low_height | term_high_height
        self.reset_buf = timeout | term_pitch | term_roll
        if self.env_cfg.get("terminate_on_height", True):
            self.reset_buf |= term_low_height | term_high_height

        self.last_term_flags["timeout"] = timeout
        self.last_term_flags["pitch"] = term_pitch
        self.last_term_flags["roll"] = term_roll
        self.last_term_flags["low_height"] = term_low_height
        self.last_term_flags["high_height"] = term_high_height

        self.episode_term_sums["timeout"] += timeout.float()
        self.episode_term_sums["pitch"] += term_pitch.float()
        self.episode_term_sums["roll"] += term_roll.float()
        self.episode_term_sums["low_height"] += term_low_height.float()
        self.episode_term_sums["high_height"] += term_high_height.float()

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.episode_error_sums["vel_xy"] += lin_vel_error * self.dt
        self.episode_error_sums["vel_yaw"] += ang_vel_error * self.dt

        time_out_idx = timeout.nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.last_pre_reset_base_pos[:] = self.base_pos[:]
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        raw_obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ]
        if self.use_height_obs:
            raw_obs_parts.append(self._get_heights())

        raw_obs = torch.cat(raw_obs_parts, dim=-1)
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=1, dims=1)
        self.obs_history_buf[:, 0] = raw_obs
        self.obs_buf = self.obs_history_buf.reshape(self.num_envs, -1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        spawn_x_range = self.env_cfg.get("spawn_x_range", [-4.0, -2.0])
        spawn_y_range = self.env_cfg.get("spawn_y_range", [-3.0, 3.0])
        self.base_pos[envs_idx, 0] = gs_rand_float(spawn_x_range[0], spawn_x_range[1], (len(envs_idx),), self.device)
        self.base_pos[envs_idx, 1] = gs_rand_float(spawn_y_range[0], spawn_y_range[1], (len(envs_idx),), self.device)
        ground_height = self._terrain_height_at(self.base_pos[envs_idx, 0], self.base_pos[envs_idx, 1])
        self.base_pos[envs_idx, 2] = ground_height + self.env_cfg.get("spawn_height_offset", 0.55)

        self.base_quat[envs_idx] = self.slope_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_world_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_contacts[envs_idx] = False
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.obs_history_buf[envs_idx] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        for key in self.episode_error_sums.keys():
            self.extras["episode"]["error_" + key] = (
                torch.mean(self.episode_error_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_error_sums[key][envs_idx] = 0.0

        for key in self.episode_term_sums.keys():
            self.extras["episode"]["term_" + key] = torch.mean(self.episode_term_sums[key][envs_idx]).item()
            self.episode_term_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_world_lin_vel_x(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_world_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_magnitude(self):
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_no_contact(self):
        return (torch.sum(self.feet_contacts.float(), dim=1) == 0).float()

    def _reward_feet_air_time_penalty(self):
        max_air_time = self.reward_cfg.get("max_feet_air_time", 0.35)
        return torch.sum(torch.clamp(self.feet_air_time - max_air_time, min=0.0), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        ground_height = self._terrain_height_at(self.base_pos[:, 0], self.base_pos[:, 1])
        local_base_height = self.base_pos[:, 2] - ground_height
        return torch.square(local_base_height - self.reward_cfg["base_height_target"])

    def _reward_termination(self):
        return self.non_timeout_reset.float()
