from active_adaptation.envs.mdp.base import Reward

import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import ContactSensor
    
class survival(Reward):
    def compute(self):
        # Multi-agent: use batch_size_total instead of num_envs
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        return torch.ones(batch_size_total, 1, device=self.device)

class linvel_z_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

    def compute(self) -> torch.Tensor:
        if self.num_agents == 1:
            linvel_z = self.asset.data.root_lin_vel_b[:, 2].unsqueeze(1)
            return -linvel_z.square()
        else:
            # Multi-agent: collect from all robots and interleave
            linvel_z_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                linvel_z_list.append(robot.data.root_lin_vel_b[:, 2])
            linvel_z = torch.stack(linvel_z_list, dim=1).reshape(-1, 1)
            return -linvel_z.square()

class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_names: str = None):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

        if body_names is not None:
            asset_for_bodies = self.asset if self.num_agents == 1 else self.assets[0]
            self.body_ids, self.body_names = asset_for_bodies.find_bodies(body_names)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None

    def update(self):
        if self.num_agents == 1:
            if self.body_ids is not None:
                angvel = self.asset.data.body_ang_vel_w[:, self.body_ids]
            else:
                angvel = self.asset.data.root_ang_vel_w.unsqueeze(1)
            self.angvel_w = angvel
        else:
            # Multi-agent: collect from all robots and interleave
            angvel_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                if self.body_ids is not None:
                    angvel = robot.data.body_ang_vel_w[:, self.body_ids]
                else:
                    angvel = robot.data.root_ang_vel_w.unsqueeze(1)
                angvel_list.append(angvel)
            self.angvel_w = torch.stack(angvel_list, dim=1).reshape(-1, angvel_list[0].shape[1], 3)

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        r = -self.angvel_w[:, :, :2].square().sum(-1).mean(1)
        return r.reshape(batch_size_total, 1).clamp_min(-1.0)

class body_upright(Reward):
    """
    Reward for keeping the specified body upright.
    """
    def __init__(self, env, body_name: str, weight, enabled = True):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
        else:
            self.assets = self.env.robots

        asset_for_bodies = self.asset if self.num_agents == 1 else self.assets[0]
        self.body_id, body_name = asset_for_bodies.find_bodies(body_name)
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        self.down = torch.tensor([[0., 0., -1.]], device=self.device).expand(batch_size_total, len(self.body_id), 3)

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        if self.num_agents == 1:
            g = quat_apply_inverse(
                self.asset.data.body_quat_w[:, self.body_id],
                self.down
            )
            rew = 1. - g[:, :, :2].square().sum(-1)
            return rew.mean(1, True)
        else:
            # Multi-agent: collect from all robots and interleave
            rew_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                g = quat_apply_inverse(
                    robot.data.body_quat_w[:, self.body_id],
                    self.down[agent_id::self.num_agents]
                )
                rew = 1. - g[:, :, :2].square().sum(-1)
                rew_list.append(rew.mean(1))
            return torch.stack(rew_list, dim=1).reshape(batch_size_total, 1)

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
            jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        else:
            self.assets = self.env.robots
            asset_for_joints = self.assets[0]
            self.joint_ids, self.joint_names = resolve_matching_names(joint_names, asset_for_joints.joint_names)
            jpos_limits = asset_for_joints.data.joint_pos_limits[:, self.joint_ids]

        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        if self.num_agents == 1:
            jpos = self.asset.data.joint_pos[:, self.joint_ids]
            violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
            violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
            return -(violation_min + violation_max).sum(1, keepdim=True)
        else:
            # Multi-agent: collect from all robots and interleave
            violation_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                jpos = robot.data.joint_pos[:, self.joint_ids]
                violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
                violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
                violation_list.append((violation_min + violation_max).sum(1))
            return -torch.stack(violation_list, dim=1).reshape(batch_size_total, 1)

class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
            self.soft_limits = self.asset.data.joint_effort_limits[:, self.joint_ids] * soft_factor
        else:
            self.assets = self.env.robots
            asset_for_joints = self.assets[0]
            self.joint_ids, self.joint_names = resolve_matching_names(joint_names, asset_for_joints.joint_names)
            self.soft_limits = asset_for_joints.data.joint_effort_limits[:, self.joint_ids] * soft_factor

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        if self.num_agents == 1:
            applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
            violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
            violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
            return - (violation_high + violation_low).sum(dim=1, keepdim=True)
        else:
            # Multi-agent: collect from all robots and interleave
            violation_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                applied_torque = robot.data.applied_torque[:, self.joint_ids]
                violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
                violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
                violation_list.append((violation_high + violation_low).sum(dim=1))
            return -torch.stack(violation_list, dim=1).reshape(batch_size_total, 1)

class action_rate_l2(Reward):
    """Penalize the rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
        # Multi-agent support: action_manager already handles flattened batch
        self.num_agents = getattr(self.env, 'num_agents', 1)

    def compute(self) -> torch.Tensor:
        # action_manager.action_buf already has shape [batch_size_total, action_dim, history]
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew

class action_rate2_l2(Reward):
    """Penalize the second order rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
        # Multi-agent support: action_manager already handles flattened batch
        self.num_agents = getattr(self.env, 'num_agents', 1)

    def compute(self) -> torch.Tensor:
        # action_manager.action_buf already has shape [batch_size_total, action_dim, history]
        action_buf = self.action_manager.action_buf
        action_diff = (
            action_buf[:, :, 0] - 2 * action_buf[:, :, 1] + action_buf[:, :, 2]
        )
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        if self.num_agents == 1:
            self.asset: Articulation = self.env.scene["robot"]
            self.joint_ids, _ = self.asset.find_joints(joint_names)
        else:
            self.assets = self.env.robots
            asset_for_joints = self.assets[0]
            self.joint_ids, _ = asset_for_joints.find_joints(joint_names)

        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        self.joint_vel = torch.zeros(
            batch_size_total, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        if self.num_agents == 1:
            self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]
        else:
            # Multi-agent: collect from all robots and interleave
            joint_vel_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                joint_vel_list.append(robot.data.joint_vel[:, self.joint_ids])
            self.joint_vel[:, substep % 2] = torch.stack(joint_vel_list, dim=1).reshape(-1, len(self.joint_ids))

    def compute(self) -> torch.Tensor:
        batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().clamp_max(5.0).sum(1, True)


class undesired_contact_force_xy(Reward):
    def __init__(self, body_names: str | List[str], thres: float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.contact_forces.find_bodies(body_names)[0]
        self.thres = thres
    
    def compute(self):
        contact_forces = self.contact_forces.data.net_forces_w[:, self.feet_ids]
        contact_forces = (contact_forces[:, :, :2].norm(dim=-1) - self.thres).clamp_min(0.0)
        return - contact_forces.mean(dim=1, keepdim=True)
    