from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.motion import MotionDataset, MotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation, RigidObject

import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse
from tensordict import TensorDict
from active_adaptation.utils.math import batchify
quat_apply = batchify(quat_apply)
quat_apply_inverse = batchify(quat_apply_inverse)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

class RobotTracking(Command):
    def __init__(
        self, env, data_path: List[str] | str,
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
        # reset parameters
        root_body_name: str = "pelvis",
        reset_range: Tuple[float, float] | None = None,
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0., 0.),
            "y": (-0., 0.),
            "z": (-0., 0.),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
        sample_motion: bool = False,
        replay_motion: bool = False,
        record_motion: bool = False,
    ):
        from . import observations
        from . import rewards
        from . import randomizations
        from . import terminations
        super().__init__(env)

        # Multi-agent support: load datasets for each agent
        self.num_agents = getattr(self.env, 'num_agents', 1)

        datasets_or_single = MotionDataset.create_from_path(
            data_path,
            isaac_joint_names=self.asset.joint_names,
            target_fps=int(1/self.env.step_dt),
            num_agents=self.num_agents
        )

        # Store dataset(s)
        if self.num_agents == 1:
            self.dataset = datasets_or_single.to(self.device)
            self.datasets = [self.dataset]  # For uniform access
        else:
            self.datasets = [ds.to(self.device) for ds in datasets_or_single]
            self.dataset = self.datasets[0]  # Keep for backward compatibility

        # Set tracking body and joint names for observation and termination
        # (Use first dataset since all agents should have the same body/joint structure)
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = [self.dataset.body_names.index(name) for name in self.tracking_keypoint_names]
        self.tracking_body_indices_asset = [self.asset.body_names.index(name) for name in self.tracking_keypoint_names]

        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [self.dataset.joint_names.index(name) for name in self.tracking_joint_names]
        self.tracking_joint_indices_asset = [self.asset.joint_names.index(name) for name in self.tracking_joint_names]

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)

        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        # Multi-agent: Use flattened batch size for all tracking buffers
        # batch_size_total = num_envs_per_agent * num_agents (interleaved ordering)
        self.batch_size_total = getattr(self.env, 'batch_size_total', self.num_envs)

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.batch_size_total, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            # All tracking buffers now use flattened batch dimension
            self.motion_ids = torch.zeros(self.batch_size_total, dtype=torch.long)
            self.motion_len = torch.zeros(self.batch_size_total, dtype=torch.long)
            self.motion_starts = torch.zeros(self.batch_size_total, dtype=torch.long)
            self.motion_ends = torch.zeros(self.batch_size_total, dtype=torch.long)
            self.t = torch.zeros(self.batch_size_total, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.batch_size_total, dtype=torch.long)

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.batch_size_total,), device=self.device)

        self.reset_range = reset_range

        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.first_sample_motion = True
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion
        self.record_motion = record_motion

        if self.replay_motion:
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0
        
        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        if call_update:
            self._init_debug_draw()
            self.update()
            if self.record_motion:
                self.motion_frames = []
        
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        """
        Sample motions for given flat indices (env_ids are flattened batch indices).
        For multi-agent: each agent samples from its own dataset.
        """
        if self.sample_motion or self.first_sample_motion:
            if self.num_agents == 1:
                # Single agent: sample from single dataset
                motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
                self.motion_ids[env_ids] = motion_ids
                self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
                self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
                self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            else:
                # Multi-agent: sample from corresponding agent's dataset
                for flat_idx in env_ids:
                    _, agent_id = self.env.get_env_agent_ids(flat_idx.unsqueeze(0))
                    agent_id = agent_id.item()

                    # Sample from agent's dataset
                    dataset = self.datasets[agent_id]
                    motion_id = torch.randint(0, dataset.num_motions, size=(1,), device=self.device).item()

                    self.motion_ids[flat_idx] = motion_id
                    self.motion_len[flat_idx] = dataset.lengths[motion_id]
                    self.motion_starts[flat_idx] = dataset.starts[motion_id]
                    self.motion_ends[flat_idx] = dataset.ends[motion_id]

            self.first_sample_motion = False
            motion_len = self.motion_len[env_ids]
        else:
            motion_len = self.motion_len[env_ids]

        if self.reset_range is None:
            max_len = motion_len - self.future_steps[-1]
            start_phase = torch.rand(len(env_ids), device=self.device)
            start_t = (start_phase * max_len).long()
        else:
            start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)

        if not self.env.training or self.record_motion:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


    def sample_init(self, env_ids: torch.Tensor) -> None:
        """
        Initialize robot states from motion data.
        env_ids: flattened batch indices [batch_size_total]
        """
        self._sample_motions(env_ids)

        if self.num_agents == 1:
            # Single agent: use original logic
            self._motion_reset: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)

            motion = self._motion_reset
            init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
            init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
            init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
            init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

            # poses
            rand_samples = sample_uniform(self.pose_range[:, 0], self.pose_range[:, 1], (len(env_ids), 6), device=self.device)
            if not self.env.training:
                rand_samples.fill_(0.0)
            positions = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            orientations = quat_mul(init_root_quat, orientations_delta)

            # velocities
            rand_samples = sample_uniform(self.velocity_range[:, 0], self.velocity_range[:, 1], (len(env_ids), 6), device=self.device)
            if not self.env.training:
                rand_samples.fill_(0.0)
            velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

            self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

            init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
            init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

            joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.init_joint_pos_noise
            joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.init_joint_vel_noise

            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise

            joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)
        else:
            # Multi-agent: reset each agent's robot separately
            for agent_id in range(self.num_agents):
                # Extract flat indices for this agent
                agent_mask = torch.tensor([(idx % self.num_agents) == agent_id for idx in env_ids], device=self.device)
                agent_flat_indices = env_ids[agent_mask]

                if len(agent_flat_indices) == 0:
                    continue

                # Get physical env_ids for this agent
                physical_env_ids, _ = self.env.get_env_agent_ids(agent_flat_indices)

                # Get motion data from agent's dataset
                dataset = self.datasets[agent_id]
                motion: MotionData = dataset.get_slice(self.motion_ids[agent_flat_indices], self.t[agent_flat_indices], 1).squeeze(1)

                init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
                init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
                init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
                init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

                # poses
                rand_samples = sample_uniform(self.pose_range[:, 0], self.pose_range[:, 1], (len(agent_flat_indices), 6), device=self.device)
                if not self.env.training:
                    rand_samples.fill_(0.0)
                positions = init_root_pos + self.env.scene.env_origins[physical_env_ids] + rand_samples[:, 0:3]
                orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
                orientations = quat_mul(init_root_quat, orientations_delta)

                # velocities
                rand_samples = sample_uniform(self.velocity_range[:, 0], self.velocity_range[:, 1], (len(agent_flat_indices), 6), device=self.device)
                if not self.env.training:
                    rand_samples.fill_(0.0)
                velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

                # Write to agent's robot
                robot = self.env.robots[agent_id]

                robot.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=physical_env_ids)
                robot.write_root_com_velocity_to_sim(velocities, env_ids=physical_env_ids)

                init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
                init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

                joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.init_joint_pos_noise
                joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.init_joint_vel_noise

                init_joint_pos += joint_pos_noise
                init_joint_vel += joint_vel_noise

                joint_pos_limits = robot.data.soft_joint_pos_limits[physical_env_ids]
                joint_vel_limits = robot.data.soft_joint_vel_limits[physical_env_ids]
                init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
                init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

                robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=physical_env_ids)

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []
    
    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()
            

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)

    @property
    def finished(self):
        if self.replay_motion:
            return torch.ones(self.batch_size_total, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def _interleave_motion_data(self, motion_list: List[MotionData]) -> MotionData:
        """
        Interleave motion data from multiple agents.

        Args:
            motion_list: List of MotionData, one per agent
                Each has shape [num_envs_per_agent, num_future_steps, ...]

        Returns:
            Interleaved MotionData with shape [batch_size_total, num_future_steps, ...]
            Ordering: [env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]
        """
        num_envs_per_agent = motion_list[0].body_pos_w.shape[0]

        # Interleave each field
        interleaved_fields = {}
        for field_name in ['motion_id', 'step', 'body_pos_w', 'body_lin_vel_w', 'body_quat_w', 'body_ang_vel_w', 'joint_pos', 'joint_vel']:
            # Stack along new agent dimension: [num_envs_per_agent, num_agents, ...]
            stacked = torch.stack([getattr(m, field_name) for m in motion_list], dim=1)
            # Reshape to interleaved: [num_envs_per_agent * num_agents, ...]
            interleaved_fields[field_name] = stacked.reshape(self.batch_size_total, *stacked.shape[2:])

        # Explicitly specify batch_size to ensure TensorClass doesn't compress it
        # This is critical when num_envs_per_agent == 1, otherwise batch_dims might become 0
        batch_size = list(interleaved_fields['body_pos_w'].shape[:2])  # [batch_size_total, num_future_steps]
        return MotionData(**interleaved_fields, batch_size=batch_size)

    def update(self):
        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))

        # Multi-agent: Get future reference motion from corresponding datasets
        if self.num_agents == 1:
            # Single agent: use original logic
            self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
            # shape: [batch_size_total, len(future_steps), num_bodies/num_joints, 3/4/...]
        else:
            # Multi-agent: gather from each agent's dataset
            future_motions = []
            for agent_id in range(self.num_agents):
                # Get flat indices for this agent: [agent_id, agent_id + num_agents, agent_id + 2*num_agents, ...]
                agent_indices = torch.arange(agent_id, self.batch_size_total, self.num_agents, device=self.device)

                # Get motion data from agent's dataset
                dataset = self.datasets[agent_id]
                agent_motion = dataset.get_slice(self.motion_ids[agent_indices], self.t[agent_indices], steps=self.future_steps)
                future_motions.append(agent_motion)

            # Interleave the motions back: [env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]
            # For now, we need to manually interleave each field
            self.future_ref_motion = self._interleave_motion_data(future_motions)

        # Multi-agent: Prepare env_origins for flattened batch (interleaved ordering)
        # Each physical env has multiple agents, so repeat env_origins for each agent
        if self.num_agents == 1:
            env_origins_flat = self.env.scene.env_origins  # [num_envs, 3]
        else:
            # Repeat each env_origin num_agents times: [env0, env0, env1, env1, ...]
            env_origins_flat = self.env.scene.env_origins.repeat_interleave(self.num_agents, dim=0)  # [batch_size_total, 3]

        # Observations: future ref body and joint states
        self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + env_origins_flat[:, None, None, :]
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[..., self.tracking_joint_indices_motion]
        self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + env_origins_flat[:, None, :]
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
        self.ref_root_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.root_body_idx_motion, :]
        self.ref_root_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.root_body_idx_motion, :]

        # Multi-agent safety check: skip robot state gathering if robots not initialized yet
        # This can happen during environment construction when __init__ calls update()
        if self.num_agents > 1 and not hasattr(self.env, 'robots'):
            # print("[DEBUG] RobotTracking.update() - skipping robot state gathering (env.robots not initialized yet)")
            # Early return - we've set self.future_ref_motion which child classes need
            return

        # Multi-agent: Gather current robot states from all robots
        if self.num_agents == 1:
            # Single agent: use original logic
            self.robot_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
            self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset]
            self.robot_body_quat_w = self.asset.data.body_link_quat_w[:, self.tracking_body_indices_asset]
            self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset]
            self.robot_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
            self.robot_joint_vel = self.asset.data.joint_vel[:, self.tracking_joint_indices_asset]
            self.robot_root_pos_w = self.asset.data.root_link_pos_w
            self.robot_root_quat_w = self.asset.data.root_link_quat_w
        else:
            # Multi-agent: gather and interleave from all robots
            robot_states = []
            for agent_id in range(self.num_agents):
                robot = self.env.robots[agent_id]
                # Each robot has shape [num_envs_per_agent, ...]
                agent_state = {
                    'body_pos_w': robot.data.body_link_pos_w[:, self.tracking_body_indices_asset],
                    'body_lin_vel_w': robot.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset],
                    'body_quat_w': robot.data.body_link_quat_w[:, self.tracking_body_indices_asset],
                    'body_ang_vel_w': robot.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset],
                    'joint_pos': robot.data.joint_pos[:, self.tracking_joint_indices_asset],
                    'joint_vel': robot.data.joint_vel[:, self.tracking_joint_indices_asset],
                    'root_pos_w': robot.data.root_link_pos_w,
                    'root_quat_w': robot.data.root_link_quat_w,
                }
                robot_states.append(agent_state)

            # Interleave: stack along agent dimension then reshape
            # [num_envs_per_agent, num_agents, ...] -> [batch_size_total, ...]
            self.robot_body_pos_w = torch.stack([s['body_pos_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1, 3)
            self.robot_body_lin_vel_w = torch.stack([s['body_lin_vel_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1, 3)
            self.robot_body_quat_w = torch.stack([s['body_quat_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1, 4)
            self.robot_body_ang_vel_w = torch.stack([s['body_ang_vel_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1, 3)
            self.robot_joint_pos = torch.stack([s['joint_pos'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1)
            self.robot_joint_vel = torch.stack([s['joint_vel'] for s in robot_states], dim=1).reshape(self.batch_size_total, -1)
            self.robot_root_pos_w = torch.stack([s['root_pos_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, 3)
            self.robot_root_quat_w = torch.stack([s['root_quat_w'] for s in robot_states], dim=1).reshape(self.batch_size_total, 4)

        # Reward: current ref body and joint states
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = self.robot_body_pos_w
            self.all_marker_pos_w[1] = self.ref_body_pos_w
            # self.all_marker_pos_w[0] = self.ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = self.ref_body_pos_future_w[:, -1]

        self.t += 1
    
    def _init_debug_draw(self):
        if self.env.backend != "isaac":
            return

        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Keypoints",
            markers={
                "robot": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        # Multi-agent: use batch_size_total for markers
        num_ref_markers = self.batch_size_total * self.num_tracking_bodies
        self.marker_indices = [0] * num_ref_markers + [1] * num_ref_markers
        self.all_marker_pos_w = torch.zeros(2, self.batch_size_total, self.num_tracking_bodies, 3, device=self.device)

    def debug_draw(self):
        if self.env.backend != "isaac":
            return

        if self.replay_motion:
            self.all_marker_pos_w.fill_(-1000)
        
        # shape: [2, num_envs, num_tracking_bodies, 3]
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

        # robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        # target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     robot_keypoints_w,
        #     target_keypoints_w - robot_keypoints_w,
        #     color=(0, 0, 1, 1)
        # )

class RobotObjectTracking(RobotTracking):
    def __init__(
        self,
        extra_object_names: List[str],
        object_asset_name: str, # for finding the object in the scene
        object_body_name: str, # for the body that defines the contact target position
        object_joint_name: str | None = None, # object joint to track
        # for reset
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        object_init_joint_pos_noise: float = 0.1, 
        object_init_joint_vel_noise: float = 0.1,
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact_frc_eef_body_name: List[str | List[str]] = ["left_wrist_(roll|pitch|yaw)_link", "right_wrist_(roll|pitch|yaw)_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        self.extra_objects: List[Articulation | RigidObject] = [self.env.scene[name] for name in extra_object_names]
        self.extra_object_body_id_motion = [self.dataset.body_names.index(name) for name in extra_object_names]

        self.object_asset_name = object_asset_name

        # Multi-agent: objects are named with agent ID suffix (e.g., "suitcase_0", "suitcase_1")
        # For batch_size_total calculation, we use the first agent's object
        if self.num_agents > 1:
            object_name_for_init = f"{object_asset_name}_0"
        else:
            object_name_for_init = object_asset_name

        if object_joint_name is None:
            self.object = self.env.scene.rigid_objects[object_name_for_init]
            self.object_joint_idx_motion = None
            self.object_joint_idx_asset = None
        else:
            self.object = self.env.scene.articulations[object_name_for_init]
            self.object_joint_idx_motion = self.dataset.joint_names.index(object_joint_name)
            self.object_joint_idx_asset = self.object.joint_names.index(object_joint_name)
        
        self.object_body_id_asset = self.object.body_names.index(object_body_name)
        self.object_body_id_motion = self.dataset.body_names.index(object_asset_name)

        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        self.object_init_joint_pos_noise = object_init_joint_pos_noise
        self.object_init_joint_vel_noise = object_init_joint_vel_noise

        if self.replay_motion or self.record_motion:
            self.object_pose_range.fill_(0.0)
            self.object_init_joint_pos_noise = 0.0
            self.object_init_joint_vel_noise = 0.0

        # setup contact body indices
        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"
        self.num_eefs = len(contact_eef_body_name)
        self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]

        self.eef_filtered_sensor: List[List[ContactSensor]] = []
        # [self.env.scene.sensors[f"{eef_name}_{object_asset_name}_contact_forces"] for eef_name in contact_eef_body_name] for object_asset_name in self.asset.data.object_names]
        self.eef_filtered_sensor_indices: List[List[int]] = []
        # = [eef_sensor.body_names.index(eef_name) for (eef_name, eef_sensor) in zip(contact_eef_body_name, self.eef_object_contact_forces)]

        # Multi-agent support: handle contact sensors with agent-specific naming
        if self.num_agents == 1:
            # Single agent: original behavior
            for eef_name in contact_eef_body_name:
                eef_names = self.asset.find_bodies(eef_name)[1]
                sensors_for_this_eef = []
                sensor_indices_for_this_eef = []
                for eef_name in eef_names:
                    eef_sensor_name = f"{eef_name}_{object_asset_name}_contact_forces"
                    eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                    sensors_for_this_eef.append(eef_sensor_filtered)
                    sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
                self.eef_filtered_sensor.append(sensors_for_this_eef)
                self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)
        else:
            # Multi-agent: sensors are named with agent IDs
            for eef_name in contact_eef_body_name:
                eef_names = self.asset.find_bodies(eef_name)[1]  # Use first robot for structure
                sensors_for_this_eef = []
                sensor_indices_for_this_eef = []
                for eef_name in eef_names:
                    # Collect sensors from all agents
                    for agent_id in range(self.num_agents):
                        eef_sensor_name = f"{eef_name}_{object_asset_name}_{agent_id}_contact_forces"
                        eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                        sensors_for_this_eef.append(eef_sensor_filtered)
                        sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
                self.eef_filtered_sensor.append(sensors_for_this_eef)
                self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)

        with torch.device(self.device):
            # Multi-agent: use batch_size_total for contact buffers
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).repeat(self.batch_size_total, 1, 1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).repeat(self.batch_size_total, 1, 1)

            self.contact_target_pos_w = torch.zeros(self.batch_size_total, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.batch_size_total, len(contact_eef_body_name), 3, device=self.device)

            self.eef_contact_forces_w = torch.zeros(self.batch_size_total, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces_b = torch.zeros(self.batch_size_total, len(contact_eef_body_name), 3, device=self.device)

        scale = getattr(self.object.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            # Use contact_target_pos_offset's actual batch size
            actual_batch_size = self.contact_target_pos_offset.shape[0]
            scale_tensor = torch.ones(actual_batch_size, 3, device=self.device)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale, device=self.device)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        else:
            # scale is a tensor with shape [num_physical_envs, 3]
            # In multi-agent mode, we need to repeat it for each agent
            if self.num_agents > 1:
                # Repeat each env's scale for all agents: [env0_scale, env0_scale, env1_scale, env1_scale, ...]
                scale = scale.repeat_interleave(self.num_agents, dim=0)
        self.contact_target_pos_offset *= scale.unsqueeze(1).to(self.device)

        # load object contact data
        motion_paths = self.dataset.motion_paths
        assert len(motion_paths) == 1, "Only one motion path is supported for RobotObjectTracking"
        motion_data = np.load(motion_paths[0], allow_pickle=True)
        object_contact = motion_data["object_contact"]
        self._object_contact = torch.tensor(object_contact, device=self.device, dtype=torch.bool)
        # if self._object_contact.shape[1] == 1:
        #     # expand to num_eefs
        #     self._object_contact = self._object_contact.repeat(1, self.num_eefs)
        # # shape: [num_steps, num_eefs/1]

        self._init_debug_draw()
        self.update()
        if self.record_motion:
            self.motion_frames = []
    
    def sample_init(self, env_ids):
        """
        Initialize object states from motion data.
        env_ids: flattened batch indices [batch_size_total]
        """
        super().sample_init(env_ids)

        if self.num_agents == 1:
            # Single agent: use original logic
            init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
            init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]

            rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(env_ids), 6), device=self.device)

            init_object_pos += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            init_object_quat = quat_mul(init_object_quat, orientations_delta)

            init_object_state_w = self.object.data.default_root_state[env_ids]
            init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
            init_object_state_w[:, 3:7] = init_object_quat
            init_object_state_w[:, 7:]  = 0.0  # zero velocity

            self.object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
            self.object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

            for object_, object_body_id_motion in zip(self.extra_objects, self.extra_object_body_id_motion):
                init_object_pos = self._motion_reset.body_pos_w[:, object_body_id_motion]
                init_object_quat = self._motion_reset.body_quat_w[:, object_body_id_motion]

                init_object_pos += rand_samples[:, 0:3]
                init_object_quat = quat_mul(init_object_quat, orientations_delta)

                init_object_state_w = object_.data.default_root_state[env_ids]
                init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
                init_object_state_w[:, 3:7] = init_object_quat
                init_object_state_w[:, 7:]  = 0.0  # zero velocity

                object_.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
                object_.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)
        else:
            # Multi-agent: reset each agent's object separately
            for agent_id in range(self.num_agents):
                # Extract flat indices for this agent
                agent_mask = torch.tensor([(idx % self.num_agents) == agent_id for idx in env_ids], device=self.device)
                agent_flat_indices = env_ids[agent_mask]

                if len(agent_flat_indices) == 0:
                    continue

                # Get physical env_ids and motion data
                physical_env_ids, _ = self.env.get_env_agent_ids(agent_flat_indices)
                dataset = self.datasets[agent_id]
                motion: MotionData = dataset.get_slice(self.motion_ids[agent_flat_indices], self.t[agent_flat_indices], 1).squeeze(1)

                init_object_pos = motion.body_pos_w[:, self.object_body_id_motion]
                init_object_quat = motion.body_quat_w[:, self.object_body_id_motion]

                rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(agent_flat_indices), 6), device=self.device)

                init_object_pos += rand_samples[:, 0:3]
                orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
                init_object_quat = quat_mul(init_object_quat, orientations_delta)

                # Get agent's object
                object_name_with_id = f"{self.object_asset_name}_{agent_id}"
                if object_name_with_id in self.env.scene.rigid_objects:
                    agent_object = self.env.scene.rigid_objects[object_name_with_id]
                elif object_name_with_id in self.env.scene.articulations:
                    agent_object = self.env.scene.articulations[object_name_with_id]
                else:
                    raise RuntimeError(f"Object {object_name_with_id} not found in scene")

                init_object_state_w = agent_object.data.default_root_state[physical_env_ids]
                init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[physical_env_ids]
                init_object_state_w[:, 3:7] = init_object_quat
                init_object_state_w[:, 7:]  = 0.0  # zero velocity

                agent_object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=physical_env_ids)
                agent_object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=physical_env_ids)

                # Handle extra objects for this agent
                for extra_obj_name, object_body_id_motion in zip(self.extra_objects, self.extra_object_body_id_motion):
                    init_object_pos = motion.body_pos_w[:, object_body_id_motion]
                    init_object_quat = motion.body_quat_w[:, object_body_id_motion]

                    init_object_pos += rand_samples[:, 0:3]
                    init_object_quat = quat_mul(init_object_quat, orientations_delta)

                    # For extra objects, we assume they are NOT duplicated per agent (use the original object)
                    # If they ARE duplicated, we'd need to modify this logic
                    object_ = extra_obj_name  # This is already the object instance from __init__
                    init_object_state_w = object_.data.default_root_state[physical_env_ids]
                    init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[physical_env_ids]
                    init_object_state_w[:, 3:7] = init_object_quat
                    init_object_state_w[:, 7:]  = 0.0  # zero velocity

                    object_.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=physical_env_ids)
                    object_.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=physical_env_ids)

        # robot_pos_w = self.asset.data.root_link_pos_w[env_ids]
        # robot_quat_w = self.asset.data.root_link_quat_w[env_ids]
        # object_pos_b = quat_apply_inverse(robot_quat_w, (init_object_pos + self.env.scene.env_origins[env_ids]) - robot_pos_w)
        # from isaaclab.utils.math import quat_conjugate
        # object_quat_b = quat_mul(quat_conjugate(robot_quat_w), init_object_quat)
        # print(f"Object initial position in robot frame: {object_pos_b}, orientation: {object_quat_b}")

        if self.object_joint_idx_asset is not None:
            init_joint_pos = self._motion_reset.joint_pos[:, self.object_joint_idx_motion].unsqueeze(1)
            init_joint_vel = self._motion_reset.joint_vel[:, self.object_joint_idx_motion].unsqueeze(1)

            joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.object_init_joint_pos_noise
            joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.object_init_joint_vel_noise
            
            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise
            
            joint_pos_limits = self.object.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.object.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.object.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids, joint_ids=[self.object_joint_idx_asset])

    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        motion_data["object_contact"] = self._object_contact[25:].cpu().numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names + [self.object_asset_name],
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()

    def update(self):
        super().update()

        # Multi-agent: Handle motion frame recording
        if hasattr(self, "motion_frames"):
            motion_frame = self.motion_frames[-1]
            # add object data to the motion frame
            if self.num_agents == 1:
                # Single agent: use original logic
                object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset].cpu()
                object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset].cpu()
                object_lin_vel_w = self.object.data.body_com_lin_vel_w[:, self.object_body_id_asset].cpu()
                object_ang_vel_w = self.object.data.body_com_ang_vel_w[:, self.object_body_id_asset].cpu()
                motion_frame["body_pos_w"] = torch.cat([motion_frame["body_pos_w"], object_pos_w.unsqueeze(1)], dim=1)
                motion_frame["body_quat_w"] = torch.cat([motion_frame["body_quat_w"], object_quat_w.unsqueeze(1)], dim=1)
                motion_frame["body_lin_vel_w"] = torch.cat([motion_frame["body_lin_vel_w"], object_lin_vel_w.unsqueeze(1)], dim=1)
                motion_frame["body_ang_vel_w"] = torch.cat([motion_frame["body_ang_vel_w"], object_ang_vel_w.unsqueeze(1)], dim=1)
            else:
                # Multi-agent: gather and interleave object data from all agents
                object_states = []
                for agent_id in range(self.num_agents):
                    object_name_with_id = f"{self.object_asset_name}_{agent_id}"
                    if object_name_with_id in self.env.scene.rigid_objects:
                        agent_object = self.env.scene.rigid_objects[object_name_with_id]
                    elif object_name_with_id in self.env.scene.articulations:
                        agent_object = self.env.scene.articulations[object_name_with_id]
                    else:
                        raise RuntimeError(f"Object {object_name_with_id} not found in scene")

                    agent_state = {
                        'pos_w': agent_object.data.body_link_pos_w[:, self.object_body_id_asset].cpu(),
                        'quat_w': agent_object.data.body_link_quat_w[:, self.object_body_id_asset].cpu(),
                        'lin_vel_w': agent_object.data.body_com_lin_vel_w[:, self.object_body_id_asset].cpu(),
                        'ang_vel_w': agent_object.data.body_com_ang_vel_w[:, self.object_body_id_asset].cpu(),
                    }
                    object_states.append(agent_state)

                # Interleave: stack along agent dimension then reshape
                object_pos_w = torch.stack([s['pos_w'] for s in object_states], dim=1).reshape(-1).unsqueeze(1)
                object_quat_w = torch.stack([s['quat_w'] for s in object_states], dim=1).reshape(-1, 4).unsqueeze(1)
                object_lin_vel_w = torch.stack([s['lin_vel_w'] for s in object_states], dim=1).reshape(-1, 3).unsqueeze(1)
                object_ang_vel_w = torch.stack([s['ang_vel_w'] for s in object_states], dim=1).reshape(-1, 3).unsqueeze(1)

                motion_frame["body_pos_w"] = torch.cat([motion_frame["body_pos_w"], object_pos_w], dim=1)
                motion_frame["body_quat_w"] = torch.cat([motion_frame["body_quat_w"], object_quat_w], dim=1)
                motion_frame["body_lin_vel_w"] = torch.cat([motion_frame["body_lin_vel_w"], object_lin_vel_w], dim=1)
                motion_frame["body_ang_vel_w"] = torch.cat([motion_frame["body_ang_vel_w"], object_ang_vel_w], dim=1)

        # Multi-agent: Prepare env_origins for flattened batch (interleaved ordering)
        if self.num_agents == 1:
            env_origins_flat = self.env.scene.env_origins  # [num_envs, 3]
        else:
            # Repeat each env_origin num_agents times: [env0, env0, env1, env1, ...]
            env_origins_flat = self.env.scene.env_origins.repeat_interleave(self.num_agents, dim=0)  # [batch_size_total, 3]

        # Reference object states from motion data (already flattened and interleaved by parent update())
        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + env_origins_flat[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]

        # Multi-agent: Gather current object states from all agents' objects
        if self.num_agents == 1:
            # Single agent: use original logic
            self.object_pos_w = self.object.data.root_link_pos_w
            self.object_quat_w = self.object.data.root_link_quat_w
        else:
            # Multi-agent: gather and interleave from all objects
            object_states = []
            for agent_id in range(self.num_agents):
                object_name_with_id = f"{self.object_asset_name}_{agent_id}"
                if object_name_with_id in self.env.scene.rigid_objects:
                    agent_object = self.env.scene.rigid_objects[object_name_with_id]
                elif object_name_with_id in self.env.scene.articulations:
                    agent_object = self.env.scene.articulations[object_name_with_id]
                else:
                    raise RuntimeError(f"Object {object_name_with_id} not found in scene")

                agent_state = {
                    'pos_w': agent_object.data.root_link_pos_w,
                    'quat_w': agent_object.data.root_link_quat_w,
                }
                object_states.append(agent_state)

            # Interleave: stack along agent dimension then reshape
            self.object_pos_w = torch.stack([s['pos_w'] for s in object_states], dim=1).reshape(self.batch_size_total, 3)
            self.object_quat_w = torch.stack([s['quat_w'] for s in object_states], dim=1).reshape(self.batch_size_total, 4)

        # Multi-agent: Handle articulated objects (if applicable)
        if self.object_joint_idx_asset is not None:
            self.ref_object_joint_pos_future = self.future_ref_motion.joint_pos[..., self.object_joint_idx_motion]
            self.ref_object_joint_vel_future = self.future_ref_motion.joint_vel[..., self.object_joint_idx_motion]
            self.ref_object_joint_pos = self.ref_object_joint_pos_future[:, 0]
            self.ref_object_joint_vel = self.ref_object_joint_vel_future[:, 0]

            if self.num_agents == 1:
                # Single agent: use original logic
                self.object_joint_pos = self.object.data.joint_pos[:, self.object_joint_idx_asset]
                self.object_joint_vel = self.object.data.joint_vel[:, self.object_joint_idx_asset]
            else:
                # Multi-agent: gather and interleave articulated object joint states
                object_joint_states = []
                for agent_id in range(self.num_agents):
                    object_name_with_id = f"{self.object_asset_name}_{agent_id}"
                    agent_object = self.env.scene.articulations[object_name_with_id]
                    agent_state = {
                        'joint_pos': agent_object.data.joint_pos[:, self.object_joint_idx_asset],
                        'joint_vel': agent_object.data.joint_vel[:, self.object_joint_idx_asset],
                    }
                    object_joint_states.append(agent_state)

                # Interleave
                self.object_joint_pos = torch.stack([s['joint_pos'] for s in object_joint_states], dim=1).reshape(self.batch_size_total, -1)
                self.object_joint_vel = torch.stack([s['joint_vel'] for s in object_joint_states], dim=1).reshape(self.batch_size_total, -1)

        # Reference object contact future states
        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]

        # Multi-agent safety check: skip contact target/eef positions if robots not initialized yet
        # This can happen during environment construction when __init__ calls update()
        if self.num_agents > 1 and not hasattr(self.env, 'robots'):
            # print("[DEBUG] RobotObjectTracking.update() - skipping contact target/eef positions (env.robots not initialized yet)")
            # Early return - we've set all the motion reference data that other components need
            return

        # Multi-agent: Contact target and eef positions
        if self.num_agents == 1:
            # Single agent: use original logic
            object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset]
            object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
            self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)

            eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
            eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
            self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        else:
            # Multi-agent: gather contact positions from all agents' objects and robots
            object_contact_states = []
            robot_eef_states = []

            for agent_id in range(self.num_agents):
                # Get agent's object
                object_name_with_id = f"{self.object_asset_name}_{agent_id}"
                if object_name_with_id in self.env.scene.rigid_objects:
                    agent_object = self.env.scene.rigid_objects[object_name_with_id]
                elif object_name_with_id in self.env.scene.articulations:
                    agent_object = self.env.scene.articulations[object_name_with_id]
                else:
                    raise RuntimeError(f"Object {object_name_with_id} not found in scene")

                # Get agent's robot
                robot = self.env.robots[agent_id]

                object_state = {
                    'pos_w': agent_object.data.body_link_pos_w[:, self.object_body_id_asset],
                    'quat_w': agent_object.data.body_link_quat_w[:, self.object_body_id_asset],
                }
                robot_state = {
                    'eef_pos_w': robot.data.body_link_pos_w[:, self.contact_eef_body_indices_asset],
                    'eef_quat_w': robot.data.body_link_quat_w[:, self.contact_eef_body_indices_asset],
                }
                object_contact_states.append(object_state)
                robot_eef_states.append(robot_state)

            # Interleave object contact states
            object_pos_w = torch.stack([s['pos_w'] for s in object_contact_states], dim=1).reshape(self.batch_size_total, 3)
            object_quat_w = torch.stack([s['quat_w'] for s in object_contact_states], dim=1).reshape(self.batch_size_total, 4)
            self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)

            # Interleave robot EEF states
            eef_pos_w = torch.stack([s['eef_pos_w'] for s in robot_eef_states], dim=1).reshape(self.batch_size_total, -1, 3)
            eef_quat_w = torch.stack([s['eef_quat_w'] for s in robot_eef_states], dim=1).reshape(self.batch_size_total, -1, 4)
            self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)

        # Multi-agent: Contact forces
        self.eef_contact_forces_w.zero_()
        if self.num_agents == 1:
            # Single agent: original logic
            for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
                for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                    self.eef_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]
        else:
            # Multi-agent: sensors are organized as [eef_type][agent_id]
            # eef_filtered_sensor structure: [[left_ag0, left_ag1], [right_ag0, right_ag1]]
            for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
                # Collect forces from all agents for this EEF type
                forces_per_agent = []
                for agent_id in range(self.num_agents):
                    sensor_idx = agent_id  # Sensors are ordered by agent_id
                    eef_sensor = eef_sensors[sensor_idx]
                    eef_sensor_id = eef_sensor_indices[sensor_idx]
                    # Force for this agent: [num_envs_per_agent, 3]
                    force = eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]
                    forces_per_agent.append(force)
                # Interleave: [num_envs_per_agent, num_agents, 3] -> [batch_size_total, 3]
                forces_interleaved = torch.stack(forces_per_agent, dim=1).reshape(self.batch_size_total, 3)
                self.eef_contact_forces_w[:, eef_idx] = forces_interleaved

        # Transform contact forces to object body frame
        if self.num_agents == 1:
            object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
        else:
            # Already computed above for multi-agent
            pass
        self.eef_contact_forces_b[:] = quat_apply_inverse(object_quat_w.unsqueeze(1), self.eef_contact_forces_w)

    def _init_debug_draw(self):
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3),
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.3, 1.0),
                        metallic=1.0,
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        # Multi-agent: use batch_size_total for markers
        self.eef_contact_markers_indices = [0, 1] * (self.batch_size_total * self.num_eefs)
        self.eef_contact_markers_pos_w = torch.zeros(self.batch_size_total, 2, self.num_eefs, 3, device=self.device)

    def debug_draw(self):
        super().debug_draw()

        if self.env.backend != "isaac":
            return
        
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        out_of_range_mask = ~self.ref_object_contact[:, None, :, None].expand_as(self.eef_contact_markers_pos_w)
        self.eef_contact_markers_pos_w[out_of_range_mask] = -1000.0
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )

        # visualize contact forces
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.reshape(-1, 3),
            self.eef_contact_forces_w.reshape(-1, 3) / 80,
            color=(1.0, 1.0, 1.0, 1.0),
            size=4.0,
        )

        # draw vector from robot root to contact target

        self.env.debug_draw.vector(
            self.contact_eef_pos_w.view(-1, 3),
            (self.contact_target_pos_w - self.contact_eef_pos_w).view(-1, 3),
            color=(0, 1, 0, 1),
            size=4.0,
        )
