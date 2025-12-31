import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp.base import Termination

class crash(Termination):
    def __init__(
        self,
        env,
        body_names_expr: str,
        t_thres: float = 0.,
        min_time: float = 0.,
        **kwargs
    ):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
            self.contact_sensors = [self.env.scene["contact_forces"]]
        else:
            self.assets = self.env.robots
            self.contact_sensors = self.env.contact_sensors

        # Find body indices for each agent's contact sensor
        self.body_indices_list = []
        self.body_names_list = []
        for contact_sensor in self.contact_sensors:
            body_indices, body_names = contact_sensor.find_bodies(body_names_expr)
            self.body_indices_list.append(body_indices)
            self.body_names_list.append(body_names)

        self.t_thres = t_thres
        self._decay = 0.98
        self._thres = (self.t_thres / self.env.physics_dt) * 0.9

        batch_size = getattr(self.env, 'batch_size_total', self.num_envs)
        num_bodies = len(self.body_indices_list[0]) if self.body_indices_list else 0
        self.count = torch.zeros(batch_size, num_bodies, device=self.env.device)

        self.min_steps = int(min_time / self.env.step_dt)
        print(f"Terminate upon contact on {self.body_names_list[0]}")

    def reset(self, env_ids):
        self.count[env_ids] = 0.

    def update(self):
        if self.num_agents == 1:
            in_contact = self.contact_sensors[0].data.net_forces_w[:, self.body_indices_list[0]].norm(dim=-1) > 1.0
            self.count.add_(in_contact.float()).mul_(self._decay)
        else:
            # Gather from all agents and interleave
            count_list = []
            for agent_id in range(self.num_agents):
                contact_sensor = self.contact_sensors[agent_id]
                body_indices = self.body_indices_list[agent_id]
                in_contact = contact_sensor.data.net_forces_w[:, body_indices].norm(dim=-1) > 1.0
                count_list.append(in_contact.float())

            # Interleave: [num_envs_per_agent, num_agents, num_bodies]
            in_contact_interleaved = torch.stack(count_list, dim=1).reshape(-1, len(self.body_indices_list[0]))
            self.count.add_(in_contact_interleaved).mul_(self._decay)

    def __call__(self):
        batch_size = getattr(self.env, 'batch_size_total', self.num_envs)
        valid = (self.env.episode_length_buf > self.min_steps)
        undesired_contact = (self.count > self._thres).any(-1)
        return (undesired_contact & valid).reshape(batch_size, 1)

class soft_contact(Termination):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.contact_sensors = [self.env.scene["contact_forces"]]
        else:
            self.contact_sensors = self.env.contact_sensors

        # Find body indices for each agent's contact sensor
        self.body_indices_list = []
        self.body_names_list = []
        for contact_sensor in self.contact_sensors:
            body_indices, body_names = contact_sensor.find_bodies(body_names)
            self.body_indices_list.append(body_indices)
            self.body_names_list.append(body_names)

    def update(self):
        if self.num_agents == 1:
            forces = self.contact_sensors[0].data.net_forces_w[:, self.body_indices_list[0]].norm(dim=-1, keepdim=True)
            in_contact = (forces > 1.0).sum(dim=1)
            self.env.discount.mul_(0.4 ** in_contact)
        else:
            # Gather from all agents and interleave
            in_contact_list = []
            for agent_id in range(self.num_agents):
                contact_sensor = self.contact_sensors[agent_id]
                body_indices = self.body_indices_list[agent_id]
                forces = contact_sensor.data.net_forces_w[:, body_indices].norm(dim=-1, keepdim=True)
                in_contact = (forces > 1.0).sum(dim=1)
                in_contact_list.append(in_contact)

            # Interleave: [num_envs_per_agent, num_agents, 1]
            in_contact_interleaved = torch.stack(in_contact_list, dim=1).reshape(-1, 1)
            self.env.discount.mul_(0.4 ** in_contact_interleaved)

    def __call__(self):
        batch_size = getattr(self.env, 'batch_size_total', self.num_envs)
        return torch.zeros(batch_size, 1, device=self.env.device, dtype=bool)
    

class fall_over(Termination):
    def __init__(
        self,
        env,
        xy_thres: float=0.8,
    ):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
        else:
            self.assets = self.env.robots

        self.xy_thres = xy_thres

    def __call__(self):
        if self.num_agents == 1:
            gravity_xy = self.assets[0].data.projected_gravity_b[:, :2]
            fall_over = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
            return fall_over
        else:
            # Gather from all robots and interleave
            fall_over_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                gravity_xy = robot.data.projected_gravity_b[:, :2]
                fall_over_agent = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
                fall_over_list.append(fall_over_agent)

            # Interleave: [num_envs_per_agent, num_agents, 1] -> [batch_size_total, 1]
            fall_over = torch.stack(fall_over_list, dim=1).reshape(-1, 1)
            return fall_over


class tracking_error(Termination):
    def __init__(self, env, tracking_error_threshold):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
        else:
            self.assets = self.env.robots

        self.tracking_error_threshold = tracking_error_threshold

    def __call__(self) -> torch.Tensor:
        if self.num_agents == 1:
            return self.assets[0].data._tracking_error > self.tracking_error_threshold
        else:
            # Gather from all robots and interleave
            error_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                error_agent = robot.data._tracking_error > self.tracking_error_threshold
                error_list.append(error_agent)

            # Interleave results
            return torch.stack(error_list, dim=1).reshape(-1, error_list[0].shape[-1])


class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85, min_steps: int = 50):
        super().__init__(env)
        self.thres = torch.tensor(thres, device=self.env.device)
        self.min_steps = min_steps # tolerate the first few steps
        self.error_exceeded_count = torch.zeros(self.env.num_envs, 1, device=self.env.device, dtype=torch.int32)
        self.command_manager = self.env.command_manager
    
    def reset(self, env_ids):
        self.error_exceeded_count[env_ids] = 0

    def update(self):
        error_exceeded = (self.command_manager._cum_error > self.thres).any(-1, True)
        self.error_exceeded_count[error_exceeded] += 1
        self.error_exceeded_count[~error_exceeded] = 0
    
    def __call__(self) -> torch.Tensor:
        return (self.error_exceeded_count > self.min_steps).reshape(-1, 1)

class ee_cum_error(Termination):
    def __init__(self, env, thres: float = 1.0, min_steps: int = 50):
        super().__init__(env)
        from .commands import CommandEEPose_Cont
        self.thres = torch.as_tensor(thres, device=self.env.device)
        self.min_steps = min_steps
        self.command_manager: CommandEEPose_Cont = self.env.command_manager
    
    def __call__(self) -> torch.Tensor:
        a = (self.command_manager._cum_error > self.thres).any(-1)
        b = self.env.episode_length_buf > self.min_steps
        return (a & b).reshape(-1, 1)


class joint_acc_exceeds(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
        else:
            self.assets = self.env.robots

        self.thres = thres

    def __call__(self) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)

        if self.num_agents == 1:
            return (
                valid &
                (self.assets[0].data.joint_acc.abs() > self.thres).any(1, True)
            )
        else:
            # Gather from all robots and interleave
            exceeds_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                exceeds_agent = (robot.data.joint_acc.abs() > self.thres).any(1, True)
                exceeds_list.append(exceeds_agent)

            # Interleave: [num_envs_per_agent, num_agents, 1] -> [batch_size_total, 1]
            exceeds = torch.stack(exceeds_list, dim=1).reshape(-1, 1)
            return valid & exceeds

class impact_exceeds(Termination):
    def __init__(self, env, body_names: str, thres: float):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
            self.contact_sensors = [self.env.scene["contact_forces"]]
        else:
            self.assets = self.env.robots
            self.contact_sensors = self.env.contact_sensors

        # Find body indices for each agent's contact sensor
        self.body_ids_list = []
        for contact_sensor in self.contact_sensors:
            body_ids = contact_sensor.find_bodies(body_names)[0]
            self.body_ids_list.append(body_ids)

        self.thres = thres

    def __call__(self) -> torch.Tensor:
        if self.num_agents == 1:
            impact_force = self.contact_sensors[0].data.net_forces_w_history[:, :, self.body_ids_list[0]]
            return (impact_force.norm(dim=-1).mean(1) > self.thres).any(1, True)
        else:
            # Gather from all agents and interleave
            exceeds_list = []
            for agent_id in range(self.num_agents):
                contact_sensor = self.contact_sensors[agent_id]
                body_ids = self.body_ids_list[agent_id]
                impact_force = contact_sensor.data.net_forces_w_history[:, :, body_ids]
                exceeds_agent = (impact_force.norm(dim=-1).mean(1) > self.thres).any(1, True)
                exceeds_list.append(exceeds_agent)

            # Interleave: [num_envs_per_agent, num_agents, 1] -> [batch_size_total, 1]
            exceeds = torch.stack(exceeds_list, dim=1).reshape(-1, 1)
            return exceeds


class impedance_pos_error(Termination):
    def __init__(self, env, thres: float = 0.3):
        super().__init__(env)
        self.num_agents = getattr(self.env, 'num_agents', 1)

        if self.num_agents == 1:
            self.assets = [self.env.scene["robot"]]
        else:
            self.assets = self.env.robots

        self.thres = thres
        self.command_manger = self.env.command_manager

    def __call__(self):
        if self.num_agents == 1:
            error = (self.assets[0].data.root_pos_w - self.command_manger.des_pos_w)[:, :2].norm(dim=-1, keepdim=True)
            return error > self.thres
        else:
            # Gather from all robots and interleave
            error_list = []
            for agent_id in range(self.num_agents):
                robot = self.assets[agent_id]
                error_agent = (robot.data.root_pos_w - self.command_manger.des_pos_w)[:, :2].norm(dim=-1, keepdim=True)
                error_exceeds = error_agent > self.thres
                error_list.append(error_exceeds)

            # Interleave: [num_envs_per_agent, num_agents, 1] -> [batch_size_total, 1]
            return torch.stack(error_list, dim=1).reshape(-1, 1)