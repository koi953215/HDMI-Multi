# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HDMI (Humanoid Whole-Body Control from Human Videos) is a research framework that trains humanoid robots (Unitree G1) to perform complex whole-body interaction tasks by learning from monocular RGB videos of human demonstrations. This is built on Isaac Sim 4.5.0, Isaac Lab v2.2.0, PyTorch 2.7.0, and TorchRL 0.7.0.

## Development Commands

### Setup
```bash
# Install in editable mode (run from repository root)
pip install -e .

# Test Isaac Sim installation
isaacsim
```

### Training
```bash
# Train teacher policy (privileged, with full state access)
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase

# Train student policy (with limited observations, distilled from teacher)
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb-run-path>

# Alternative: use wrapper scripts that set environment variables
./my_train.sh algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

### Evaluation
```bash
# Evaluate a trained policy
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path>

# Export to ONNX for deployment
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path> export_policy=true

# Alternative: use wrapper script
./my_play.sh algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path>
```

### Motion Data Visualization
```bash
# Visualize motion data in Isaac Sim (verify training data)
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase +task.command.replay_motion=true

# Visualize motion.npz in MuJoCo (two terminals)
# Terminal 1:
python scripts/vis/mujoco_mocap_viewer.py
# Terminal 2:
python scripts/vis/motion_data_publisher.py data/motion/g1/omomo/sub1_suitcase_011
```

### Batch Evaluation
```bash
# Evaluate multiple checkpoints
python scripts/eval_multiple.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase

# Evaluate a specific W&B run
python scripts/eval_run.py <wandb-run-id>
```

## Architecture Overview

### Modular MDP Framework

The codebase is built around a composable, callback-driven environment architecture in [active_adaptation/envs/base.py](active_adaptation/envs/base.py):

**Callback Lifecycle**:
1. `reset` - Initialize MDP components when environments reset
2. `pre_step` - Called before each physics step (e.g., command updates)
3. `post_step` - Called after each physics step (e.g., reward computation)
4. `update` - Called after all physics substeps (e.g., command manager updates)
5. `debug_draw` - Visualization hooks (GUI mode only)

**MDP Component Interfaces** ([active_adaptation/envs/mdp/base.py](active_adaptation/envs/mdp/base.py)):
- `Action`: Joint position control with configurable scaling and delays
- `Command`: Motion tracking commands (RobotTracking, RobotObjectTracking)
- `Observation`: Modular observation groups (policy, privileged, command, object, depth)
- `Reward`: Tracking rewards for body keypoints, joints, and object trajectories
- `Termination`: Task-specific done conditions
- `Randomization`: Domain randomization for sim-to-real transfer

### HDMI-Specific Components

Core HDMI implementation is in [active_adaptation/envs/mdp/commands/hdmi/](active_adaptation/envs/mdp/commands/hdmi/):

- **[command.py](active_adaptation/envs/mdp/commands/hdmi/command.py)**: `RobotTracking` and `RobotObjectTracking` classes that sample initial states and provide reference trajectories
- **[observations.py](active_adaptation/envs/mdp/commands/hdmi/observations.py)**: Future reference states (`ref_joint_pos_future`), action-aligned references (`ref_joint_pos_action`), object features, contact targets
- **[rewards.py](active_adaptation/envs/mdp/commands/hdmi/rewards.py)**: Keypoint/body tracking (pos/ori/vel), joint tracking, object tracking, contact rewards (EEF-to-target distance)
- **[terminations.py](active_adaptation/envs/mdp/commands/hdmi/terminations.py)**: Task-specific termination conditions
- **[randomizations.py](active_adaptation/envs/mdp/commands/hdmi/randomizations.py)**: Domain randomization parameters

### PPO-ROA Algorithm

The core learning algorithm is **PPO with Regularized Online Adaptation** ([active_adaptation/learning/ppo/ppo_roa.py](active_adaptation/learning/ppo/ppo_roa.py)):

**Three Training Phases**:
1. `ppo_roa_train`: Train privileged teacher policy with full state access (including privileged observations)
2. `ppo_roa_adapt`: Optional adaptation phase (rarely used)
3. `ppo_roa_finetune`: Train student policy with limited observations, distilled from teacher via residual action matching

**Residual Action Distillation** (Section 4.3 of the paper):
- During `phase=train` with `enable_residual_distillation=True`:
  - Teacher policy optimized with PPO on privileged observations
  - Student policy (adapter) supervised to match teacher's actions in residual action space
- During `phase=finetune`:
  - Student policy further optimized with PPO on limited observations

**Key Features**:
- GRU or MLP adapters for partial observability
- Optional depth camera and object observations
- Entropy scheduling for exploration
- Value normalization and GAE

### Configuration System

**Hydra-based hierarchy**:
- Base configs: [cfg/base/](cfg/base/) (not found in repo, likely defined in code)
- Task configs: [cfg/task/G1/hdmi/](cfg/task/G1/hdmi/) - 13 HDMI tasks
- Algorithm configs: Registered in [active_adaptation/learning/ppo/ppo_roa.py](active_adaptation/learning/ppo/ppo_roa.py) via ConfigStore
- Main training config: [cfg/train.yaml](cfg/train.yaml)

**Overriding configs**:
```bash
# Override specific parameters
python scripts/train.py task=G1/hdmi/move_suitcase task.max_episode_length=300

# Add new parameters
python scripts/train.py task=G1/hdmi/move_suitcase +task.command.replay_motion=true
```

### Motion Data Format

Motion data is stored as `motion.npz` with metadata in `meta.json` ([active_adaptation/utils/motion.py](active_adaptation/utils/motion.py)):

**Required arrays**:
- Body states: `pos`, `quat`, `lin_vel`, `ang_vel` → `[T, B, 3/4]`
- Joint states: `pos`, `vel` → `[T, J]`

Where `T` = timesteps, `B` = bodies (robot + objects), `J` = joints.

**Data Pipeline**:
Human video → GVHMR → GMR/LocoMujoco → Robot motion + Object trajectory

**Important**: Object trajectories are **appended** to robot body states. For tasks with object manipulation:
1. Extract object trajectory (position, orientation, velocities)
2. Append object name to `meta.json`
3. Concatenate object body states to robot body states: `[T, B_robot + B_object, 3/4]`

## Repository Structure

### Key Directories
- [active_adaptation/envs/](active_adaptation/envs/) — Environment framework with callback system and MDP components
- [active_adaptation/learning/ppo/](active_adaptation/learning/ppo/) — Single-file PPO implementations (standard, ROA, AMP)
- [active_adaptation/utils/](active_adaptation/utils/) — Motion loading, math utils, symmetry transforms, W&B integration
- [scripts/](scripts/) — Training/evaluation entry points
- [cfg/](cfg/) — Hydra configuration files
- [data/motion/](data/motion/) — Motion reference datasets

### Important Files
- [scripts/train.py](scripts/train.py) — Main training loop with W&B integration
- [scripts/play.py](scripts/play.py) — Evaluation and ONNX export
- [scripts/helpers.py](scripts/helpers.py) — `make_env_policy()` - creates environment and policy instances
- [active_adaptation/envs/base.py](active_adaptation/envs/base.py) — Base environment with callback orchestration
- [active_adaptation/learning/ppo/common.py](active_adaptation/learning/ppo/common.py) — GAE, MLP builders, shared utilities

## Training Workflow

### Typical Two-Phase Training

**Phase 1: Train Teacher (Privileged Policy)**
```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```
- Uses privileged observations (full state information)
- Trains for ~150M frames by default
- Outputs checkpoint to W&B
- Simultaneously trains student adapter via residual distillation

**Phase 2: Finetune Student**
```bash
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb-run-path>
```
- Uses limited observations (deployable on real robot)
- Initializes from teacher's student adapter
- Further optimizes with PPO

### TensorDict Data Flow

The training loop uses TorchRL's TensorDict for zero-copy tensor operations:

```python
TensorDict({
  "action": Float[num_envs, T, act_dim],
  "sample_log_prob": Float[num_envs, T, 1],
  "state_value": Float[num_envs, T, 1],
  "policy": Float[..., obs_dim],           # Policy observations
  "priv": Float[..., priv_dim],            # Privileged observations
  "reward": Float[num_envs, T, n_rewards],
  "is_init": Bool[num_envs, T, 1],         # Episode start mask
  "next": {
    "state_value": Float[num_envs, T, 1],
    "discount": Float[num_envs, T, 1],
    "done": Bool[num_envs, T, 1],
  }
})
```

## Common Task Configurations

Available HDMI tasks in [cfg/task/G1/hdmi/](cfg/task/G1/hdmi/):
- `move_suitcase` - Moving a suitcase
- `carry_and_place_bread_box` - Carrying and placing objects
- `open_door-feet` - Opening doors with feet
- `push_door-hand` - Pushing doors with hands
- `roll_ball-hand` - Rolling a ball
- `topple_wood_board_and_cross` - Toppling obstacles
- `traverse_door` - Traversing through doorways
- And 6 more tasks

Each task config specifies:
- `robot.robot_type` - Robot model variant (e.g., `g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule`)
- `command.data_path` - Path to motion data
- `command.object_asset_name` / `object_body_name` - Object to manipulate
- `command.contact_target_pos_offset` - Contact target offsets for end-effectors
- `randomization` - Domain randomization parameters (friction, mass, scale)

## W&B Integration

Training automatically logs to Weights & Biases:
- Project: `hdmi` (configured in [cfg/train.yaml](cfg/train.yaml))
- Set `WANDB_API_KEY` environment variable (see [my_train.sh](my_train.sh))
- Checkpoint loading via `checkpoint_path=run:<wandb-run-path>`
- Disable logging: `wandb.mode=disabled`

## Important Implementation Details

### Object Trajectory Annotation

From [FAQ.md](FAQ.md): Object trajectories are **manually annotated** for each video clip, then post-processed with task-specific scripts. For example:
- Mark pickup/drop-off frames
- Align object with wrist during holding
- Place on ground otherwise
- Interpolate and smooth the trajectory

Future releases may integrate automated pipelines like OmniRetarget.

### Motion Tracking Observations

From [FAQ.md](FAQ.md): The framework uses `link_pos/quat_w` and `com_lin/ang_vel` for efficiency. Getting link velocities from Isaac Sim is significantly slower than using center-of-mass velocities.

### Sim2Real Deployment

6 of 14 tasks successfully deployed on real Unitree G1 hardware. See separate repository: [github.com/EGalahad/sim2real](https://github.com/EGalahad/sim2real)

### Multi-Backend Support

The environment supports both Isaac Sim (GPU-accelerated) and MuJoCo backends:
- Isaac Sim: Primary training backend ([active_adaptation/envs/humanoid.py](active_adaptation/envs/humanoid.py))
- MuJoCo: Lightweight visualization ([active_adaptation/envs/mujoco.py](active_adaptation/envs/mujoco.py))

## Code Modification Guidelines

### Adding a New HDMI Task

1. Create motion data in required format (`motion.npz` + `meta.json`)
2. Create task config in [cfg/task/G1/hdmi/](cfg/task/G1/hdmi/) inheriting from `base/hdmi-base`
3. Specify robot model, motion data path, object assets, contact targets
4. Configure domain randomization parameters
5. Visualize motion with `+task.command.replay_motion=true`
6. Train teacher → finetune student

### Adding New MDP Components

1. Inherit from base class in [active_adaptation/envs/mdp/base.py](active_adaptation/envs/mdp/base.py)
2. Implement lifecycle methods: `__init__`, `reset`, `step`, `update`
3. For observations: inherit from `ObservationBuilder`, return observation tensor
4. For rewards: inherit from `Reward`, return reward tensor
5. Register in task config under appropriate section

### Modifying PPO Algorithm

Single-file implementations in [active_adaptation/learning/ppo/](active_adaptation/learning/ppo/):
- Modify `__init__` for architecture changes
- Modify `train_op` for new loss terms
- Add ConfigStore registration for new algorithm variants
- Use `common.make_mlp` for network construction

### TensorDict Best Practices

- Access nested keys: `td.get(("next", "state_value"))`
- Update in-place: `td.update(new_dict)`
- Batch operations work across `[num_envs, T]` dimensions automatically
- Use `td.select(*keys)` to extract subsets

## Output and Logging

- **Training outputs**: `./outputs/<date>/<time>-<task_name>-<algo_name>/`
- **Checkpoints**: Saved at intervals (default: every 300 iterations)
- **W&B artifacts**: Checkpoints and metrics synced automatically
- **Hydra logs**: Config snapshots in output directory

---

## Multi-Agent iPPO Extension (In Development)

This codebase is being extended to support **Independent PPO (iPPO)** training with arbitrary numbers of agents.

### Goal and Requirements

**Objective**: Support training N independent agents in the same environment, each with their own:
- Motion reference data
- Robot instance
- Manipulation object (e.g., separate suitcases)
- Policy and value networks (completely independent)

**Key Requirements**:
1. **Arbitrary number of agents**: Configurable via `num_agents` parameter (e.g., 1, 2, 4, ...)
2. **Complete independence**: Each agent has its own dataset, observations, rewards, and neural networks
3. **Batch dimension flattening**: `batch_size = [num_envs * num_agents]` for TorchRL compatibility
4. **Physical coexistence**: Agents exist in the same Isaac Sim environment and can physically collide, but motion data has built-in offsets to prevent this
5. **Backward compatibility**: `num_agents=1` must behave identically to the current single-agent setup

### Multi-Agent Data Structure

**Single agent** (current):
```
data/motion/g1/omomo/sub1_suitcase_011/
├── motion.npz
└── meta.json
```

**Multi-agent** (new):
```
data/motion/g1/omomo/sub1_suitcase_011_dual/
├── agent_0/
│   ├── motion.npz
│   └── meta.json
└── agent_1/
    ├── motion.npz
    └── meta.json
```

Each agent's data is completely independent with spatial offsets pre-applied.

### Batch Dimension Design

**Critical Design Decision**: Flatten `(num_envs, num_agents)` into single batch dimension.

**Current single-agent**:
```python
batch_size = [num_envs]  # e.g., [2048]
obs["policy"] = Float[2048, obs_dim]
action = Float[2048, action_dim]
```

**Multi-agent flattened**:
```python
num_envs_per_agent = 2048
num_agents = 2
batch_size = [4096]  # num_envs_per_agent * num_agents

# Interleaved ordering: [env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]
obs["policy"] = Float[4096, obs_dim]
action = Float[4096, action_dim]
```

**Index mapping**:
- Flat index = `env_id * num_agents + agent_id`
- Extract agent's envs: `indices = torch.arange(agent_id, num_envs_total, num_agents)`

**Rationale**: TorchRL expects single batch dimension; flattening maintains full compatibility.

### High-Level Architecture Changes

#### 1. Configuration (`cfg/task/base/hdmi-base.yaml` or task configs)
```yaml
num_agents: 1  # Default single agent, override for multi-agent

command:
  data_path: data/motion/g1/omomo/sub1_suitcase_011_dual
```

#### 2. Environment Scene Setup ([active_adaptation/envs/locomotion.py](active_adaptation/envs/locomotion.py))

**Current**: Single robot + object per env
```python
scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
scene_cfg.suitcase.prim_path = "{ENV_REGEX_NS}/suitcase"
```

**Multi-agent**: Multiple robots + objects per env
```python
for agent_id in range(num_agents):
    robot_cfg.prim_path = f"{{ENV_REGEX_NS}}/Robot_{agent_id}"
    obj_cfg.prim_path = f"{{ENV_REGEX_NS}}/suitcase_{agent_id}"
    setattr(scene_cfg, f"robot_{agent_id}", robot_cfg)
    setattr(scene_cfg, f"suitcase_{agent_id}", obj_cfg)
```

Result: Each env contains N robots and N objects (e.g., `/World/envs/env_0/Robot_0`, `/World/envs/env_0/Robot_1`)

#### 3. Motion Data Loading ([active_adaptation/utils/motion.py](active_adaptation/utils/motion.py))

**Modification**: Load multiple datasets based on `num_agents`
```python
def create_from_path(data_path, num_agents=1, ...):
    if num_agents == 1:
        # Current behavior: load data_path/motion.npz
        return MotionDataset(data_path)
    else:
        # Multi-agent: load data_path/agent_0/, agent_1/, ...
        datasets = []
        for i in range(num_agents):
            agent_path = os.path.join(data_path, f"agent_{i}")
            datasets.append(MotionDataset(agent_path))
        return datasets
```

#### 4. Command Manager ([active_adaptation/envs/mdp/commands/hdmi/command.py](active_adaptation/envs/mdp/commands/hdmi/command.py))

**Modification**: Manage multiple datasets and tracking states
```python
class RobotTracking:
    def __init__(self, env, data_path, num_agents=1, ...):
        self.num_agents = num_agents
        self.datasets = load_datasets(data_path, num_agents)

        # Tracking buffers: [num_envs_per_agent, num_agents]
        self.motion_ids = torch.zeros(num_envs_per_agent, num_agents, dtype=torch.long)
        self.t = torch.zeros(num_envs_per_agent, num_agents, dtype=torch.long)
```

#### 5. MDP Components

**Observations** ([active_adaptation/envs/mdp/commands/hdmi/observations.py](active_adaptation/envs/mdp/commands/hdmi/observations.py)):
- Each agent only observes its own robot and object
- Reshape from `[num_envs_per_agent, num_agents, ...]` → `[num_envs_total, ...]` with interleaving

**Actions** ([active_adaptation/envs/mdp/action.py](active_adaptation/envs/mdp/action.py)):
- Distribute flattened actions to corresponding robots
- Agent i's actions: `action[i::num_agents]` → `robots[i].set_joint_position_target(...)`

**Rewards** ([active_adaptation/envs/mdp/commands/hdmi/rewards.py](active_adaptation/envs/mdp/commands/hdmi/rewards.py)):
- Compute independently per agent
- Flatten and interleave results

#### 6. iPPO Policy Networks ([active_adaptation/learning/ppo/ppo_roa.py](active_adaptation/learning/ppo/ppo_roa.py))

**Key Change**: Independent networks per agent using `nn.ModuleList`

```python
class PPOROA:
    def __init__(self, cfg, ..., num_agents=1):
        self.num_agents = num_agents

        # Create N independent networks
        self.actors = nn.ModuleList([build_actor(...) for _ in range(num_agents)])
        self.critics = nn.ModuleList([build_critic(...) for _ in range(num_agents)])
        self.adapters = nn.ModuleList([build_adapter(...) for _ in range(num_agents)])

        # Separate optimizers
        self.opt_policies = [Adam(self.actors[i].parameters()) for i in range(num_agents)]
        self.opt_critics = [Adam(self.critics[i].parameters()) for i in range(num_agents)]
```

**Rollout Policy**: Wrapper to distribute observations to agent-specific networks
```python
def get_rollout_policy(self):
    # Returns policy that:
    # 1. Extracts each agent's observations from flattened batch
    # 2. Passes through agent's network
    # 3. Interleaves actions back into flat batch
```

**Training**: Independent PPO updates per agent
```python
def train_op(self, tensordict):
    for agent_id in range(self.num_agents):
        # Extract agent's data: tensordict[agent_id::num_agents]
        agent_data = extract_agent_data(tensordict, agent_id)

        # Train with agent's networks
        loss = self.train_policy_single_agent(agent_data, agent_id)
        self.opt_policies[agent_id].step()
```

### Critical Files for Multi-Agent Implementation

**Must modify** (core logic changes):
1. [active_adaptation/envs/locomotion.py](active_adaptation/envs/locomotion.py) - `setup_scene()` for multiple robots/objects
2. [active_adaptation/envs/base.py](active_adaptation/envs/base.py) - Batch dimension from `[num_envs]` → `[num_envs * num_agents]`
3. [active_adaptation/envs/mdp/commands/hdmi/command.py](active_adaptation/envs/mdp/commands/hdmi/command.py) - Multi-dataset management
4. [active_adaptation/learning/ppo/ppo_roa.py](active_adaptation/learning/ppo/ppo_roa.py) - Independent networks (ModuleList)

**Moderate changes**:
5. [active_adaptation/envs/mdp/action.py](active_adaptation/envs/mdp/action.py) - Action distribution to multiple robots
6. [active_adaptation/envs/mdp/commands/hdmi/observations.py](active_adaptation/envs/mdp/commands/hdmi/observations.py) - Per-agent observation extraction
7. [active_adaptation/envs/mdp/commands/hdmi/rewards.py](active_adaptation/envs/mdp/commands/hdmi/rewards.py) - Per-agent reward computation
8. [active_adaptation/utils/motion.py](active_adaptation/utils/motion.py) - Multi-agent data loading

### Design Decisions (Finalized)

1. **Batch flattening**: ALWAYS use `[num_envs * num_agents]`, never nested `[num_envs, num_agents]`
2. **Index ordering**: Interleaved by agent within env: `[env0_ag0, env0_ag1, env1_ag0, env1_ag1, ...]`
3. **Network architecture**: `nn.ModuleList` with N completely independent networks (not shared with agent embeddings)
4. **Observation isolation**: Each agent observes ONLY its own robot and object (no global/shared information)
5. **Data offsets**: Spatial offsets are PRE-APPLIED in motion data; code should NOT add additional offsets
6. **Backward compatibility**: `num_agents=1` must maintain identical behavior to original codebase

### Multi-Agent Termination Strategy

**Default Behavior** (`agent0_only_termination: false`):
- Any agent's termination triggers reset for the entire physical environment
- All agents in the same environment reset together when any one fails

**Agent-0 Only Mode** (`agent0_only_termination: true`):
- **Only agent_0's termination can trigger environment reset**
- All other agents (agent_1, agent_2, ...) inherit agent_0's termination status
- If agent_1 fails but agent_0 is fine → no reset occurs
- If agent_0 fails → entire environment (all agents) resets
- Use case: Debugging, curriculum learning, or asymmetric training scenarios

**Implementation**: [base.py:459-487](active_adaptation/envs/base.py#L459-L487)
```python
def _compute_termination(self):
    # ... compute each agent's termination ...
    if self.num_agents > 1 and self.cfg.get("agent0_only_termination", False):
        # Extract agent_0's status: indices [0, num_agents, 2*num_agents, ...]
        agent0_terminated = terminated[0::self.num_agents]
        # Broadcast to all agents: [ag0_env0, ag0_env0, ag0_env1, ag0_env1, ...]
        terminated = agent0_terminated.repeat_interleave(self.num_agents, dim=0)
    return terminated
```

**Configuration**:
```yaml
# Normal multi-agent (any agent can trigger reset)
task: G1/hdmi/move_suitcase_dual
agent0_only_termination: false  # Default

# Agent-0 only termination (special mode)
task: G1/hdmi/move_suitcase_dual_agent0_only
agent0_only_termination: true
```

**Removing the feature**: Simply set `agent0_only_termination: false` in config, or delete the if-block in `_compute_termination()` (lines 472-483)

### Training Commands (Multi-Agent)

```bash
# Train 2-agent iPPO teacher
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase num_agents=2

# Train 2-agent iPPO student
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase num_agents=2 checkpoint_path=run:<teacher-run>
```

### Implementation Status

**Phase 1** (In Progress): Configuration and data loading
- [ ] Add `num_agents` parameter to YAML configs
- [ ] Modify `MotionDataset.create_from_path()` for multi-agent data loading
- [ ] Update `RobotTracking` command to handle multiple datasets

**Phase 2**: Environment scene setup
- [ ] Modify `setup_scene()` to create N robots and objects per env
- [ ] Update environment initialization to track multiple robot articulations

**Phase 3**: Batch dimension restructuring
- [ ] Change base env `batch_size` from `[num_envs]` → `[num_envs * num_agents]`
- [ ] Implement index mapping utilities

**Phase 4**: MDP component updates
- [ ] Modify action manager for multi-robot control
- [ ] Update observation builders for per-agent data extraction
- [ ] Update reward functions for per-agent computation

**Phase 5**: iPPO policy implementation
- [ ] Implement `nn.ModuleList` architecture in PPO-ROA
- [ ] Create rollout policy wrapper for multi-agent
- [ ] Modify `train_op()` for independent agent updates

**Phase 6**: Testing and validation
- [ ] Single-agent regression tests (`num_agents=1`)
- [ ] Dual-agent training validation (`num_agents=2`)
- [ ] Performance benchmarking
