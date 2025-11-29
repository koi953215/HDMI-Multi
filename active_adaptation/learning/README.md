# Active Adaptation Algorithms

We provide single-file PPO algorithms built on TensorDict.

## Files at a glance
- `ppo.py` — standard on-policy PPO with GAE, optional value norm, Adam optimizer, and optional `torch.compile` for the update/rollout policy.
- `ppo_roa.py` — PPO variant used for HDMI tasks: supports train/adapt/finetune phases (`algo=ppo_roa_*`), residual distillation, optional depth/object inputs, entropy scheduling, and optional GRU adapters.
- `ppo_amp.py` — PPO + Adversarial Motion Priors: adds a discriminator and AMP replay buffer on top of PPO.
- `common.py` — shared utilities (GAE, batching, MLP builders, normalization, key constants).
- `critics.py` — critic helpers and value-network utilities reused across variants.

## Common interface
- Rollout: `policy.get_rollout_policy(mode="train"|"eval")` returns the actor-only TensorDictModule used inside collectors/envs.
- Learning: `policy.train_op(tensordict)` consumes a rollout TensorDict to compute advantages/returns (GAE), PPO losses, and apply an optimizer step.
- State: `state_dict()/load_state_dict` cover actor, critic, and value norm (when enabled).

`ppo_roa` adds phase-specific behaviors on top of this interface:
- `phase=train` (`algo=ppo_roa_train`): encode privileged observations, train the main policy and the adaptation module; optionally enable **residual action distillation** (teacher = privileged policy, student = adapted policy).
- `phase=adapt` / `phase=finetune` (`algo=ppo_roa_adapt|finetune`): run the adaptation module (GRU/MLP) and then an adapted actor; `finetune` also trains the adapter together with the actor.
- `phase=train_est` / `phase=adapt_est`: attach a privileged estimator (e.g., depth/object features) and optionally a domain-randomization estimator.
- Recurrent adapters: when `adapt_module="gru"`, the rollout policy also reads/writes `("next", "adapt_hx")` via a `TensorDictPrimer(reset_key="done")`.

Typical TensorDict passed to `train_op` (structure from the collector):
```python
TensorDict(
  {
    "action":          Float[num_envs, T, act_dim],
    "sample_log_prob": Float[num_envs, T, 1],
    "state_value":     Float[num_envs, T, 1],
    "loc":             Float[num_envs, T, act_dim],   # actor mean (old policy)
    "scale":           Float[num_envs, T, act_dim],   # actor std  (old policy)
    OBS_KEY:           Float[..., obs_dim],           # e.g., "policy" group
    OBS_PRIV_KEY:      Float[..., priv_dim],          # privileged group
    REWARD_KEY:        Float[num_envs, T, reward_groups],
    "is_init":         Bool [num_envs, T, 1],         # episode-start mask
    "next": {
      "state_value":   Float[num_envs, T, 1],
      "discount":      Float[num_envs, T, 1],
      "done":          Bool [num_envs, T, 1],
      "terminated":    Bool [num_envs, T, 1],
      "truncated":     Bool [num_envs, T, 1],
    },
  },
  batch_size=[num_envs, T],
  device=cuda:0,
)
```

For `ppo_roa`, the same structure is extended with additional keys such as:
- command/object inputs (`CMD_KEY`, `OBJECT_KEY`, optional `DEPTH_KEY`)
- adaptation features/targets (`PRIV_FEATURE_KEY`, `PRIV_PRED_KEY`, optional `priv_est`, `dr_pred`)
- recurrent state when `adapt_module="gru"` (`("next", "adapt_hx")`).

## Typical usage
- Select the algorithm via Hydra: `algo=ppo` (vanilla), `algo=ppo_roa_train|adapt|finetune|*_est`, or `algo=ppo_amp`.
- The training entrypoint (`scripts/train.py`) calls `policy.get_rollout_policy(...)` for sampling and `policy.train_op(...)` for updates.
- Observation keys must match the variant’s `in_keys` (e.g., `ppo_roa` expects command + policy + object + priv keys; add depth if using `_est` configs).

## Extending
To create a new variant:
- Register a new Hydra `algo` config and (optionally) new observation groups in the task config.
- Add auxiliary heads / losses in `train_op` (or a helper like `train_policy`, `train_adapt`, `train_estimator`).
- Adjust MLP/RNN builders in `__init__` (or reuse `common.make_mlp`) for new architectures.

## Residual Action Distillation
`ppo_roa.py` implements residual action distillation (see Sec. 4.3 in the paper) so that an adapted “student” policy can match a privileged “teacher” policy in a residual action space:
- During `phase=train` with `enable_residual_distillation=True`, 
  - `train_policy` optimizes the privileged policy (teacher) with PPO.
  - `train_adapt` computes the teacher’s action (`actor` with a residual action module) and the student’s action (`actor_adapt` with a vanilla MLP) and supervises the student to match the teacher.
- During finetuning (`phase=finetune`)
  - `train_policy` the student (`actor_adapt`) is further optimized with PPO.
