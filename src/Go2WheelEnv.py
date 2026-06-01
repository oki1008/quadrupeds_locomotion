import math

import genesis as gs
import torch
import torch.nn.functional as F
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat


"""Go2 + 受動車輪用の強化学習環境。

車輪付きGo2ロボット、地形、観測、報酬、reset/stepをまとめて定義する。
"""


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2WheelEnv:
    """Go2 + 受動車輪のRL環境クラス。

    主な役割:
    - GenesisのSceneを作る。
    - Go2 + 受動車輪のURDFモデルを読み込む。
    - flat / stair / rough の地形を作る。
    - Actorに渡す観測を作る。
    - 報酬と終了条件を計算する。
    - reset() / step() でPPO学習用の環境として動く。

    最初はGo2と同じ12関節だけを制御し、4つの車輪関節は受動関節として扱う。
    """

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        device="cuda",
        add_camera=False,
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        print("読み込むクラス: Go2WheelEnv")

        self.num_envs = num_envs
        self.num_obs = obs_cfg.get("num_obs", 0)
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg.get("sim_substeps", 30)),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # 地形を先に作り、その上にロボットを配置する。
        self._build_terrain()

        # Go2 + 受動車輪のURDFを読み込む場所。
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=(
                    "assets/go2_wheel/go2w_passive_small_wheel_description/"
                    "urdf/go2w_passive_small_wheel_description.urdf"
                ),
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # 学習で制御する12関節だけを取り出し、PD制御のゲインを設定する。
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # 受動車輪4関節。制御はしないが、後で車輪回転速度の観測に使う。
        self.wheel_dof_names = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
        self.wheel_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.wheel_dof_names]

        # configに書いた報酬名から、対応する_reward_xxx関数を自動登録する。
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.episode_error_sums = {
            "vel_xy": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            "vel_yaw": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
        }

        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        # Go2Wheelではhipを小さく動かし、横倒れしやすい初期探索を抑える。
        self.action_scales = torch.tensor(
            [
                0.125 if "hip" in name else self.env_cfg["action_scale"]
                for name in self.env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.wheel_dof_vel = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # 車輪付きモデルでは接地リンクを車輪リンクとして扱う。
        self.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        link_names = [link.name for link in self.robot.links]
        self.feet_indices = torch.tensor(
            [link_names.index(name) for name in self.feet_names],
            device=self.device,
            dtype=torch.long,
        )
        self.feet_air_time = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.jump_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        self.jump_target_height = torch.zeros((self.num_envs,), device=self.device)
        self.extras = {}

        # 高さ観測用のローカル格子点。use_height_obs=falseなら計算しても観測には入れない。
        measure_points_x = torch.linspace(-0.8, 0.8, 11, device=self.device)
        measure_points_y = torch.linspace(-0.5, 0.5, 11, device=self.device)
        grid_x, grid_y = torch.meshgrid(measure_points_x, measure_points_y, indexing="ij")
        self.num_height_points = grid_x.numel()
        self.local_height_points = torch.zeros((self.num_envs, self.num_height_points, 3), device=self.device)
        self.local_height_points[:, :, 0] = grid_x.flatten()
        self.local_height_points[:, :, 1] = grid_y.flatten()

        # configに合わせて観測次元を決める。
        self.use_height_obs = obs_cfg.get("use_height_obs", False)
        self.use_wheel_vel_obs = obs_cfg.get("use_wheel_vel_obs", True)
        self.num_history = obs_cfg.get("num_history", 3)
        self.num_base_obs = 48 + (4 if self.use_wheel_vel_obs else 0)
        self.num_raw_obs = self.num_base_obs + (self.num_height_points if self.use_height_obs else 0)
        self.num_obs = self.num_raw_obs * self.num_history
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_history, self.num_raw_obs),
            device=self.device,
            dtype=gs.tc_float,
        )

    def _build_terrain(self):
        """configのterrain_typeに応じてflat / stair / rough地形を作る。"""
        terrain_type = self.env_cfg.get("terrain_type", "stair")
        terrain_width = self.env_cfg.get("terrain_width", 20.0)
        terrain_length = self.env_cfg.get("terrain_length", 20.0)
        horizontal_scale = self.env_cfg.get("horizontal_scale", 0.05)

        n_rows = int(terrain_width / horizontal_scale)
        n_cols = int(terrain_length / horizontal_scale)

        if terrain_type == "rough":
            # 粗いランダム地形を補間して、なめらかな不整地を作る。
            seed_resolution = self.env_cfg.get("rough_seed_resolution", 20)
            small_noise = torch.rand((1, 1, seed_resolution, seed_resolution), device=self.device)
            height_field_raw = F.interpolate(
                small_noise,
                size=(n_rows, n_cols),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            height_field_raw *= self.env_cfg.get("terrain_height_scale", 0.10)
        else:
            height_field_raw = torch.zeros((n_rows, n_cols), device=self.device)

        if terrain_type == "stair":
            # x方向に進むほど高さが増える階段地形を作る。
            x = (torch.arange(n_rows, device=self.device) - n_rows // 2) * horizontal_scale
            grid_x = x[:, None].repeat(1, n_cols)
            step_width = self.env_cfg.get("terrain_step_width", 0.35)
            step_height = self.env_cfg.get("terrain_step_height", 0.03)
            stair_start_x = self.env_cfg.get("stair_start_x", 0.5)
            stair_end_x = self.env_cfg.get("stair_end_x", 4.0)
            active_x = torch.clamp(grid_x - stair_start_x, min=0.0)
            step_idx = torch.floor(active_x / step_width)
            max_step_idx = math.floor(max(stair_end_x - stair_start_x, 0.0) / step_width)
            height_field_raw = torch.where(grid_x < stair_start_x, torch.zeros_like(height_field_raw), step_idx * step_height)
            height_field_raw = torch.clamp(height_field_raw, max=max_step_idx * step_height)

        # 初期転倒ではなく歩行能力を評価するため、スタート周辺だけ平らにする。
        flat_radius = self.env_cfg.get("start_flat_radius_cells", 20)
        center_x, center_y = n_rows // 2, n_cols // 2
        height_field_raw[
            center_x - flat_radius : center_x + flat_radius,
            center_y - flat_radius : center_y + flat_radius,
        ] = 0.0

        self.height_field_tensor = height_field_raw
        self.terrain_height_field = height_field_raw.cpu().numpy()
        self.horizontal_scale = horizontal_scale
        self.n_rows = n_rows
        self.n_cols = n_cols
        # GenesisのTerrainはposを地形の端として扱うため、原点が地形中央になるようにずらす。
        self.terrain_origin = (
            -0.5 * n_rows * horizontal_scale,
            -0.5 * n_cols * horizontal_scale,
            0.0,
        )

        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                height_field=self.terrain_height_field,
                horizontal_scale=horizontal_scale,
                vertical_scale=1.0,
                pos=self.terrain_origin,
            )
        )

    def _sample_commands(self, envs_idx):
        """学習中に各環境へ与える目標速度・目標高さをランダムに更新する。"""
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 3] = gs_rand_float(
            *self.command_cfg["height_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 4] = 0.0

        height_span = self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]
        if abs(height_span) < 1e-6:
            height_diff_scale = torch.ones((len(envs_idx),), device=self.device)
        else:
            height_diff_scale = (
                0.5
                + torch.abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"])
                / height_span
                * 0.5
            )
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale

    def _sample_jump_commands(self, envs_idx):
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["jump_range"], (len(envs_idx),), self.device)

    def _get_heights(self):
        """ロボット周囲121点の地形高さを、胴体高さからの相対値として返す。"""
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

        px = ((x + (self.n_rows * self.horizontal_scale) / 2) / self.horizontal_scale).long()
        py = ((y + (self.n_cols * self.horizontal_scale) / 2) / self.horizontal_scale).long()
        px = torch.clip(px, 0, self.n_rows - 1)
        py = torch.clip(py, 0, self.n_cols - 1)

        heights = self.height_field_tensor[px, py]
        return heights - self.base_pos[:, 2].unsqueeze(1)

    def step(self, actions, is_train=True):
        # Actorの出力を関節目標角に変換し、PD制御でロボットへ入力する。
        self.actions = torch.clip(actions,-self.env_cfg["clip_actions"],self.env_cfg["clip_actions"],)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.action_scales + self.default_dof_pos
        joint_clip = self.env_cfg.get("target_dof_pos_clip", 0.8)
        target_dof_pos = torch.clip(
        target_dof_pos,
        self.default_dof_pos - joint_clip,
        self.default_dof_pos + joint_clip,
    )

        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # 足先接触を見て、feet_air_time報酬用の空中時間を更新する。
        all_forces = self.robot.get_links_net_contact_force()
        forces = all_forces[:, self.feet_indices, :]
        contacts = torch.norm(forces, dim=-1) > 0.1
        self.feet_air_time += self.dt
        self.feet_air_time[contacts] = 0.0
        self.last_contacts = contacts

        self.episode_length_buf += 1
        # シミュレーション後のロボット状態をGenesisから取得し、観測と報酬に使う。
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.wheel_dof_vel[:] = self.robot.get_dofs_velocity(self.wheel_dofs)

        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if is_train:
            # 学習時だけ目標速度を定期的に変える。評価時はコマンド固定。
            self._sample_commands(envs_idx)
            random_idxs_1 = torch.randperm(self.num_envs, device=self.device)[: int(self.num_envs * 0.05)]
            self._sample_commands(random_idxs_1)
            random_idxs_2 = torch.randperm(self.num_envs, device=self.device)[: int(self.num_envs * 0.05)]
            self._sample_jump_commands(random_idxs_2)

        jump_cmd_now = (self.commands[:, 4] > 0.0).float()
        toggle_mask = ((self.jump_toggled_buf == 0.0) & (jump_cmd_now > 0.0)).float()
        self.jump_toggled_buf += toggle_mask * self.reward_cfg["jump_reward_steps"]
        self.jump_toggled_buf = torch.clamp(self.jump_toggled_buf - 1.0, min=0.0)
        self.jump_target_height = torch.where(jump_cmd_now > 0.0, self.commands[:, 4], self.jump_target_height)

        # 転倒・低すぎる高さ・タイムアウトを終了条件として扱う。
        term_timeout = self.episode_length_buf > self.max_episode_length
        term_pitch = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        term_roll = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        term_low_height = self.base_pos[:, 2] < self.env_cfg.get("termination_abs_min_height", 0.25)
        term_high_height = self.base_pos[:, 2] > self.env_cfg.get("termination_max_height", 1.2)
        self.reset_buf = term_timeout | term_pitch | term_roll | term_low_height | term_high_height

        time_out_idx = term_timeout.nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.extras["term_roll"] = term_roll.to(gs.tc_float)
        self.extras["term_pitch"] = term_pitch.to(gs.tc_float)
        self.extras["term_low_height"] = term_low_height.to(gs.tc_float)
        self.extras["term_high_height"] = term_high_height.to(gs.tc_float)
        self.extras["term_timeout"] = term_timeout.to(gs.tc_float)

        if self.env_cfg.get("auto_reset", True):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 登録された報酬関数を順に計算して合計する。
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # 1フレーム分の観測を作る。
        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            (self.jump_toggled_buf / self.reward_cfg["jump_reward_steps"]).unsqueeze(-1),
        ]
        if self.use_wheel_vel_obs:
            obs_parts.insert(5, self.wheel_dof_vel * self.obs_scales["dof_vel"])
        if self.use_height_obs:
            obs_parts.append(self._get_heights() * self.obs_scales["height_measurements"])
        raw_obs = torch.cat(obs_parts, dim=-1)

        # raw_obsを履歴方向に積む。num_history=3なら3フレーム分をActor入力にする。
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=1, dims=1)
        self.obs_history_buf[:, 0] = raw_obs
        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.commands[:, 4] = 0.0

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.episode_error_sums["vel_xy"] += lin_vel_error * self.dt
        self.episode_error_sums["vel_yaw"] += ang_vel_error * self.dt

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """指定された環境だけ初期姿勢・初期位置・履歴をリセットする。"""
        if len(envs_idx) == 0:
            return

        # 関節をデフォルト姿勢に戻す。
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.wheel_dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # 地形高さにspawn_height_offsetを足して、地面に埋まらない位置から開始する。
        spawn_x_range = self.env_cfg.get("spawn_x_range", [1.5, 2.5])
        spawn_y_range = self.env_cfg.get("spawn_y_range", [1.5, 2.5])
        self.base_pos[envs_idx, 0] = gs_rand_float(spawn_x_range[0], spawn_x_range[1], (len(envs_idx),), self.device)
        self.base_pos[envs_idx, 1] = gs_rand_float(spawn_y_range[0], spawn_y_range[1], (len(envs_idx),), self.device)
        self.base_pos[envs_idx, 2] = self._terrain_height_at(self.base_pos[envs_idx, 0], self.base_pos[envs_idx, 1])
        self.base_pos[envs_idx, 2] += self.env_cfg.get("spawn_height_offset", 0.45)

        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.jump_toggled_buf[envs_idx] = 0.0
        self.jump_target_height[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False
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

        self._sample_commands(envs_idx)
        self.commands[envs_idx, 3] = self.reward_cfg["base_height_target"]

    def _terrain_height_at(self, x, y):
        """ワールド座標(x, y)に対応する地形高さをheight fieldから読む。"""
        px = ((x + (self.n_rows * self.horizontal_scale) / 2) / self.horizontal_scale).long()
        py = ((y + (self.n_cols * self.horizontal_scale) / 2) / self.horizontal_scale).long()
        px = torch.clip(px, 0, self.n_rows - 1)
        py = torch.clip(py, 0, self.n_cols - 1)
        return self.height_field_tensor[px, py]

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel(self):
        # 目標の前後・左右速度に近いほど高い報酬。
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # 目標ヨー角速度に近いほど高い報酬。
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_y(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_lin_vel[:, 1])

    def _reward_lin_vel_z(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_lin_vel[:, 2])

    def _reward_orientation(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.base_euler[:, :2]), dim=1)

    def _reward_yaw_orientation(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_euler[:, 2])

    def _reward_action_rate(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_magnitude(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.actions), dim=1)

    def _reward_dof_vel(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_similar_to_default(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # 地形高さを引いた胴体高さが、目標高さに近いほどペナルティが小さい。
        active_mask = (self.jump_toggled_buf < 0.01).float()
        ground_height = self._terrain_height_at(self.base_pos[:, 0], self.base_pos[:, 1])
        local_base_height = self.base_pos[:, 2] - ground_height
        return active_mask * torch.square(local_base_height - self.commands[:, 3])

    def _reward_feet_air_time(self):
        return torch.sum(self.feet_air_time, dim=1)
