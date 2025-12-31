import torch
import numpy as np
from typing import Dict, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]
        self.action_buf: torch.Tensor

        # Multi-agent support
        self.num_agents = getattr(self.env, 'num_agents', 1)
        self.batch_size_total = getattr(self.env, 'batch_size_total', self.env.num_envs)

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: float | Dict[str, float] = 0.5,
        min_delay: int = 0,
        max_delay: int = 0,
        alpha: float | Tuple[float, float] = 0.5,
        **kwargs,
    ):
        super().__init__(env)
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.min_delay = min_delay if min_delay is not None else 0
        self.max_delay = max_delay if max_delay is not None else 0

        import omegaconf
        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        elif isinstance(alpha, omegaconf.listconfig.ListConfig):
            self.alpha_range = tuple(alpha)
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

        self.default_joint_pos = self.asset.data.default_joint_pos.clone()

        # Multi-agent: offset should have batch_size_total (flattened batch dimension)
        if self.num_agents == 1:
            self.offset = torch.zeros_like(self.default_joint_pos)
        else:
            # Create offset with shape [batch_size_total, num_joints]
            num_joints = self.default_joint_pos.shape[1]
            self.offset = torch.zeros(self.batch_size_total, num_joints, device=self.device)

        # Multi-agent: Use batch_size_total for all action buffers (flattened batch dimension)
        with torch.device(self.device):
            action_buf_hist = max((self.max_delay - 1) // self.env.decimation + 1, 3)
            self.action_buf = torch.zeros(
                self.batch_size_total, self.action_dim, action_buf_hist
            )  # at least 3 for action_rate_2_l2 reward
            self.applied_action = torch.zeros(self.batch_size_total, self.action_dim)
            self.alpha = torch.ones(self.batch_size_total, 1)
            self.delay = torch.zeros(self.batch_size_total, 1, dtype=int)

    def resolve(self, spec):
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        delay = torch.randint(self.min_delay, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.delay[env_ids] = delay
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.alpha_range
        )
        self.alpha[env_ids] = alpha

    def __call__(self, action: torch.Tensor, substep: int):
        if substep == 0:
            if isinstance(action, TensorDictBase):
                action = action["action"]
            self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]
            self.action_buf[:, :, 0] = action
        # if delay = 1
        #     substep = 0, action_dim: 1
        #     substep = 1, action_dim: 0
        #     substep = 2, action_dim: 0
        #     substep = 3, action_dim: 0
        # if delay = 2
        #     substep = 0, action_dim: 1
        #     substep = 1, action_dim: 1
        #     substep = 2, action_dim: 0
        #     substep = 3, action_dim: 0
        action_dim = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        action = self.action_buf.take_along_dim(action_dim.unsqueeze(1), dim=-1)
        self.applied_action.lerp_(action.squeeze(-1), self.alpha)

        # Multi-agent: Distribute actions to corresponding robots
        if self.num_agents == 1:
            # Single agent: use original logic
            pos_target = self.default_joint_pos + self.offset
            pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
            self.asset.set_joint_position_target(pos_target)
        else:
            # Multi-agent: distribute flattened actions to each robot
            # Flattened ordering: [env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]
            for agent_id in range(self.num_agents):
                # Extract agent's actions: [agent_id, agent_id + num_agents, agent_id + 2*num_agents, ...]
                agent_indices = torch.arange(agent_id, self.batch_size_total, self.num_agents, device=self.device)
                agent_applied_action = self.applied_action[agent_indices]  # [num_envs_per_agent, action_dim]
                agent_offset = self.offset[agent_indices]  # [num_envs_per_agent, num_joints]

                # Get agent's robot
                robot = self.env.robots[agent_id]

                # Compute target position for this agent (with offset support)
                pos_target = robot.data.default_joint_pos + agent_offset
                pos_target[:, self.joint_ids] += agent_applied_action * self.action_scaling

                # Set target for this agent's robot
                robot.set_joint_position_target(pos_target)
        
