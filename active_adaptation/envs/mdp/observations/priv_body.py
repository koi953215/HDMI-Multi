from active_adaptation.envs.mdp.base import Observation

import torch
import active_adaptation.utils.symmetry as sym_utils
from active_adaptation.utils.math import EMA
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import Imu

class body_pos_b(Observation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

        # Use first robot to get body structure (all robots have same bodies)
        asset_for_bodies = self.asset if self.num_agents == 1 else self.assets[0]
        self.body_indices, self.body_names = asset_for_bodies.find_bodies(body_names)
        self.update()
        if self.env.backend == "mujoco":
            self.feet_marker_0 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])
            self.feet_marker_1 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])

    def update(self):
        if self.num_agents == 1:
            self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
            self.root_pos_w = self.asset.data.root_pos_w.unsqueeze(1).clone()
            # TODO: now assume ground height is 0
            self.root_pos_w[..., 2] = 0.0
            self.body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        else:
            # Multi-agent: collect from all robots and interleave
            root_quat_list = []
            root_pos_list = []
            body_pos_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                root_quat = yaw_quat(robot.data.root_quat_w).unsqueeze(1)
                root_pos = robot.data.root_pos_w.unsqueeze(1).clone()
                root_pos[..., 2] = 0.0
                body_pos = robot.data.body_pos_w[:, self.body_indices]
                root_quat_list.append(root_quat)
                root_pos_list.append(root_pos)
                body_pos_list.append(body_pos)
            # Interleave: [num_envs_per_agent, num_agents, ...] -> [batch_size_total, ...]
            self.root_quat_w = torch.stack(root_quat_list, dim=1).reshape(-1, 1, 4)
            self.root_pos_w = torch.stack(root_pos_list, dim=1).reshape(-1, 1, 3)
            self.body_pos_w = torch.stack(body_pos_list, dim=1).reshape(-1, len(self.body_indices), 3)

    def compute(self):
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        body_pos_b = quat_apply_inverse(self.root_quat_w, self.body_pos_w - self.root_pos_w)
        return body_pos_b.reshape(batch_size_total, -1)

    def symmetry_transforms(self):
        asset_for_symmetry = self.asset if self.num_agents == 1 else self.assets[0]
        return sym_utils.cartesian_space_symmetry(asset_for_symmetry, self.body_names)

    def debug_draw(self):
        if self.env.backend == "mujoco":
            asset_for_debug = self.asset if self.num_agents == 1 else self.assets[0]
            self.feet_marker_0.geom.pos = asset_for_debug.data.body_pos_w[0, self.body_indices[0]]
            self.feet_marker_1.geom.pos = asset_for_debug.data.body_pos_w[0, self.body_indices[1]]


class body_vel_b(Observation):
    def __init__(self, env, body_names: str, yaw_only: bool=False):
        super().__init__(env)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

        self.yaw_only = yaw_only
        # Use first robot to get body structure (all robots have same bodies)
        asset_for_bodies = self.asset if self.num_agents == 1 else self.assets[0]
        self.body_indices, self.body_names = asset_for_bodies.find_bodies(body_names)
        self.update()

    def update(self):
        if self.num_agents == 1:
            if self.yaw_only:
                self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
            else:
                self.root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
            self.body_vel_w = self.asset.data.body_vel_w[:, self.body_indices]
        else:
            # Multi-agent: collect from all robots and interleave
            root_quat_list = []
            body_vel_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                if self.yaw_only:
                    root_quat = yaw_quat(robot.data.root_quat_w).unsqueeze(1)
                else:
                    root_quat = robot.data.root_quat_w.unsqueeze(1)
                body_vel = robot.data.body_vel_w[:, self.body_indices]
                root_quat_list.append(root_quat)
                body_vel_list.append(body_vel)
            # Interleave: [num_envs_per_agent, num_agents, ...] -> [batch_size_total, ...]
            self.root_quat_w = torch.stack(root_quat_list, dim=1).reshape(-1, 1, 4)
            self.body_vel_w = torch.stack(body_vel_list, dim=1).reshape(-1, len(self.body_indices), 6)

    def compute(self):
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        body_lin_vel_b = quat_apply_inverse(self.root_quat_w, self.body_vel_w[:, :, :3])
        body_ang_vel_b = quat_apply_inverse(self.root_quat_w, self.body_vel_w[:, :, 3:])
        return body_lin_vel_b.reshape(batch_size_total, -1)

    def symmetry_transforms(self):
        asset_for_symmetry = self.asset if self.num_agents == 1 else self.assets[0]
        return sym_utils.cartesian_space_symmetry(asset_for_symmetry, self.body_names)


class body_acc(Observation):
    
    def __init__(self, env, body_names, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        print(f"Track body acc for {self.body_names}")
        self.body_acc_b = torch.zeros(self.env.num_envs, len(self.body_indices), 3, device=self.env.device)

    def update(self):
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        body_acc_w = self.asset.data.body_lin_acc_w[:, self.body_indices]
        self.body_acc_b[:] = quat_apply_inverse(quat, body_acc_w)
        
    def compute(self):
        return self.body_acc_b.reshape(self.env.num_envs, -1)


class imu_acc(Observation):
    def __init__(self, env, smoothing_window: int=3):
        super().__init__(env)
        self.imu: Imu = self.env.scene["imu"]
        self.smoothing_window = smoothing_window
        self.acc_buf = torch.zeros(self.env.num_envs, 3, smoothing_window, device=self.env.device)

    def reset(self, env_ids):
        self.acc_buf[env_ids] = 0.0

    def update(self):
        self.acc_buf[:, :, 1:] = self.acc_buf[:, :, :-1]
        self.acc_buf[:, :, 0] = self.imu.data.lin_acc_b

    def compute(self):
        return self.acc_buf.mean(dim=2).view(self.env.num_envs, -1)
    

class imu_angvel(Observation):
    def __init__(self, env, smoothing_window: int=3):
        super().__init__(env)
        self.imu: Imu = self.env.scene["imu"]
        self.smoothing_window = smoothing_window
        self.angvel_buf = torch.zeros(self.env.num_envs, 3, smoothing_window, device=self.env.device)
    
    def reset(self, env_ids):
        self.angvel_buf[env_ids] = 0.0

    def update(self):
        self.angvel_buf[:, :, 1:] = self.angvel_buf[:, :, :-1]
        self.angvel_buf[:, :, 0] = self.imu.data.ang_vel_b

    def compute(self):
        return self.angvel_buf.mean(dim=2).view(self.env.num_envs, -1)

   

class root_linvel_b(Observation):
    def __init__(self, env, gammas=(0.,), yaw_only: bool=False):
        super().__init__(env)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
            self.yaw_only = yaw_only
            self.ema = EMA(self.asset.data.root_lin_vel_w, gammas=gammas)
            self.ema.update(self.asset.data.root_lin_vel_w)
        else:
            self.assets = self.env.robots
            self.yaw_only = yaw_only
            # Create EMAs for each agent
            self.emas = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                ema = EMA(robot.data.root_lin_vel_w, gammas=gammas)
                ema.update(robot.data.root_lin_vel_w)
                self.emas.append(ema)
        self.update()

    def reset(self, env_ids: torch.Tensor):
        if self.num_agents == 1:
            self.ema.reset(env_ids)
        else:
            # Multi-agent: reset each agent's EMA
            # Convert interleaved env_ids to physical env_ids
            # env_ids are interleaved batch indices: [0, 1, 2, 3, ...] for [env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]
            # We need physical env indices: [0, 1, ...] for the actual physical environments
            physical_env_ids = torch.unique(env_ids // self.num_agents)

            for agent_id in range(self.num_agents):
                self.emas[agent_id].reset(physical_env_ids)

    def post_step(self, substep):
        if self.num_agents == 1:
            self.ema.update(self.asset.data.root_lin_vel_w)
        else:
            # Multi-agent: update each agent's EMA
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                self.emas[agent_id].update(robot.data.root_lin_vel_w)

    def update(self):
        if self.num_agents == 1:
            if self.yaw_only:
                self.quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
            else:
                self.quat = self.asset.data.root_quat_w.unsqueeze(1)
        else:
            # Multi-agent: collect from all robots and interleave
            quat_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                if self.yaw_only:
                    quat = yaw_quat(robot.data.root_quat_w).unsqueeze(1)
                else:
                    quat = robot.data.root_quat_w.unsqueeze(1)
                quat_list.append(quat)
            # Interleave: [num_envs_per_agent, num_agents, 1, 4] -> [batch_size_total, 1, 4]
            self.quat = torch.stack(quat_list, dim=1).reshape(-1, 1, 4)

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        if self.num_agents == 1:
            linvel = self.ema.ema
        else:
            # Multi-agent: collect from all EMAs and interleave
            linvel_list = []
            for agent_id in range(self.num_agents):
                linvel_list.append(self.emas[agent_id].ema)
            # Interleave: [num_envs_per_agent, num_agents, ...] -> [batch_size_total, ...]
            linvel = torch.stack(linvel_list, dim=1).reshape(-1, *linvel_list[0].shape[1:])
        linvel = quat_apply_inverse(self.quat, linvel)
        return linvel.reshape(batch_size_total, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform

    # def debug_draw(self):
    #     if self.env.sim.has_gui() and self.env.backend == "isaac":
    #         if self.body_ids is None:
    #             linvel = self.asset.data.root_lin_vel_w
    #         else:
    #             linvel = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).mean(1)
    #         self.env.debug_draw.vector(
    #             self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
    #             linvel,
    #             color=(0.8, 0.1, 0.1, 1.)
    #         )

class body_height(Observation):
    # this will use ray casting to compute the height of the ground under the body
    def __init__(self, env, body_names: str):
        super().__init__(env)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

        # Use first robot to get body structure (all robots have same bodies)
        asset_for_bodies = self.asset if self.num_agents == 1 else self.assets[0]
        self.body_ids, self.body_names = asset_for_bodies.find_bodies(body_names)
        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)

    def compute(self):
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        if self.num_agents == 1:
            body_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
            body_height = body_pos_w[:, :, 2] - self.env.get_ground_height_at(body_pos_w)
            return body_height.reshape(self.num_envs, -1)
        else:
            # Multi-agent: collect from all robots and interleave
            body_height_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                body_pos_w = robot.data.body_pos_w[:, self.body_ids]
                body_height = body_pos_w[:, :, 2] - self.env.get_ground_height_at(body_pos_w)
                body_height_list.append(body_height)
            # Interleave: [num_envs_per_agent, num_agents, num_bodies] -> [batch_size_total, num_bodies]
            body_height = torch.stack(body_height_list, dim=1).reshape(-1, len(self.body_ids))
            return body_height.reshape(batch_size_total, -1)

    def symmetry_transforms(self):
        asset_for_symmetry = self.asset if self.num_agents == 1 else self.assets[0]
        return sym_utils.cartesian_space_symmetry(asset_for_symmetry, self.body_names, sign=(1,))
