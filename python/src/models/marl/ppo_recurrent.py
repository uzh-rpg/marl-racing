from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
import os
from src.utils.train_utils import load_teacher
from pathlib import Path
import warnings
from src.models.marl.on_policy_algorithm import OnPolicyAlgorithm
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0003,
        n_steps: int = 2048,
        use_tanh_act: bool = True,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        env_cfg: str = None,
        _init_setup_model: bool = True,
    ):
        if use_sde:
            raise ValueError("The paper does not use state-dependent exploration")
        super(RecurrentPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            use_tanh_act=use_tanh_act,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.env_cfg = env_cfg
        self.num_agents = self.env.num_agents
        self.num_selfplay_agents = self.env.num_selfplay_agents
        self.num_active_agents = self.num_agents - self.num_selfplay_agents
        self.active_agents_idx = [i for i in range(self.num_active_agents)]
        self.selfplay_agents_idx = [
            i for i in range(self.num_active_agents, self.num_agents)
        ]
        self.use_tanh_act = use_tanh_act
        self._last_lstm_states = [None] * self.num_agents
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        buffer_cls = RecurrentRolloutBuffer
        self.policy = [
            self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs,
            )
            for _ in range(self.num_agents)
        ]
        self.opponent_pool = []
        self.opponent_pool_size = 0
        self.opponent_assignment = None
        self.historic_opponent_cache = {}
        self.historic_opponent_cache_order = []
        flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
        pool = []
        if isinstance(self.env_cfg, dict):
            if "opponents_pool" in self.env_cfg:
                pool = self.env_cfg.get("opponents_pool", [])
            elif (
                "environment" in self.env_cfg
                and isinstance(self.env_cfg["environment"], dict)
                and ("opponents_pool" in self.env_cfg["environment"])
            ):
                pool = self.env_cfg["environment"].get("opponents_pool", [])
        if pool:
            self.opponent_pool_paths = []
            for p in pool:
                resolved = str(flightmare_path / Path(p))
                try:
                    opp_cfg = self._load_opponent_run_config(flightmare_path / Path(p))
                    policy_loaded = load_teacher(
                        flightmare_path / Path(p),
                        device=self.device,
                        config=opp_cfg or self.env_cfg,
                        env=self.env,
                    )
                    policy_loaded.set_training_mode(False)
                    self.opponent_pool.append(policy_loaded)
                    self.opponent_pool_paths.append(resolved)
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot load required opponent policy {p}"
                    ) from e
                    self.opponent_pool_paths.append(resolved)
            self.opponent_pool_size = len(self.opponent_pool)
            print(f"Loaded {self.opponent_pool_size} opponent policies for self-play.")
        self.opponent_pool_prob = 0.5
        if isinstance(self.env_cfg, dict):
            self.opponent_pool_prob = float(
                self.env_cfg.get(
                    "opponents_pool_prob",
                    self.env_cfg.get("environment", {}).get("opponents_pool_prob", 0.5),
                )
            )
        self._selfplay_index_map = {
            ag: i for (i, ag) in enumerate(self.selfplay_agents_idx)
        }
        try:
            if isinstance(self.env_cfg, dict):
                hist_heur = self.env_cfg.get(
                    "historic_sampling_heuristic",
                    self.env_cfg.get("environment", {}).get(
                        "historic_sampling_heuristic", "powerlaw"
                    ),
                )
            else:
                hist_heur = "powerlaw"
        except Exception:
            hist_heur = "powerlaw"
        if hist_heur != "powerlaw":
            raise ValueError("The paper uses powerlaw checkpoint sampling")
        self.sampler = self.sample_checkpoint_powerlaw
        for ag in range(self.num_agents):
            if self.use_tanh_act:
                self.policy[ag].action_net = th.nn.Sequential(
                    self.policy[ag].action_net, th.nn.Tanh()
                )
            self.policy[ag] = self.policy[ag].to(self.device)
        for ag in self.selfplay_agents_idx:
            if self.opponent_pool_size > 0:
                random_idx = np.random.randint(0, self.opponent_pool_size)
                self.policy[ag] = self.opponent_pool[random_idx]
                self.policy[ag].set_training_mode(False)
        lstm = self.policy[0].lstm_actor
        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = [
            RNNStates(
                (
                    th.zeros(single_hidden_state_shape, device=self.device),
                    th.zeros(single_hidden_state_shape, device=self.device),
                ),
                (
                    th.zeros(single_hidden_state_shape, device=self.device),
                    th.zeros(single_hidden_state_shape, device=self.device),
                ),
            )
            for _ in range(self.num_agents)
        ]
        hidden_state_buffer_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )
        self.rollout_buffer = [
            buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )
            for _ in range(self.num_active_agents)
        ]
        self.clip_range = get_schedule_fn(self.clip_range)
        self.curriculum_updated = 1

    def _load_opponent_run_config(
        self, weight_path: Union[str, Path]
    ) -> Optional[dict]:
        try:
            p = Path(weight_path)
            ancestors = [p.parent, p.parent.parent, p.parent.parent.parent]
            for anc in ancestors:
                if anc is None:
                    continue
                try:
                    if not anc.exists():
                        continue
                except Exception:
                    continue
                cfg_candidates = list(anc.glob("config.yaml"))
                if cfg_candidates:
                    cfg_path = cfg_candidates[0]
                    try:
                        import yaml

                        with open(cfg_path, "r") as f:
                            cfg = yaml.safe_load(f)
                        return cfg
                    except Exception as e_yaml:
                        warnings.warn(
                            f"Failed to parse opponent run config {cfg_path}: {e_yaml}"
                        )
                        return None
            return None
        except Exception:
            return None

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        for buffer in rollout_buffer:
            assert isinstance(
                buffer, RecurrentRolloutBuffer
            ), f"{buffer} doesn't support recurrent policy"
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        for ag in range(self.num_active_agents):
            self.rollout_buffer[ag].reset()
        callback.on_rollout_start()
        policy_dir = None
        try:
            policy_dir = self.logger.get_dir() + "/checkpoints/"
        except Exception:
            policy_dir = None
        checkpoints = []
        if policy_dir is not None and os.path.exists(policy_dir):
            checkpoints = sorted(
                [f for f in os.listdir(policy_dir) if f.endswith(".pth")],
                key=lambda name: int(Path(name).stem),
            )
        local_loaded = {}
        hist_size = len(checkpoints)
        rollout_subset = []
        subset_size = 8
        if hist_size > 0:
            actual_subset_size = min(subset_size, hist_size)
            selected_set = set()
            max_attempts = 5 * actual_subset_size
            attempts = 0
            while len(selected_set) < actual_subset_size and attempts < max_attempts:
                ck = self.sampler(checkpoints)
                if ck is None:
                    break
                selected_set.add(ck)
                attempts += 1
            if len(selected_set) < actual_subset_size:
                remaining = list(set(checkpoints) - selected_set)
                if remaining:
                    need = min(actual_subset_size - len(selected_set), len(remaining))
                    selected_set.update(
                        np.random.choice(remaining, size=need, replace=False).tolist()
                    )
            if not selected_set:
                selected_set = set(checkpoints[:actual_subset_size])
            rollout_subset = list(selected_set)
            checkpoints = rollout_subset
        cfg_size = getattr(self, "opponent_pool_size", 0)
        hist_size = len(checkpoints)
        subset_size_pool = 8
        selected_pool_indices = []
        if cfg_size > 0:
            actual_pool_subset_size = min(subset_size_pool, cfg_size)
            selected_pool_indices = np.random.choice(
                cfg_size, size=actual_pool_subset_size, replace=False
            ).tolist()
            cfg_size = len(selected_pool_indices)
        self.selected_pool_indices = selected_pool_indices
        self.opponent_assignment_source = np.zeros(
            (env.num_envs, self.num_selfplay_agents), dtype=bool
        )
        self.opponent_assignment_idx = np.zeros(
            (env.num_envs, self.num_selfplay_agents), dtype=int
        )
        if cfg_size == 0:
            self.opponent_assignment_source[:, :] = False
            for col in range(self.num_selfplay_agents):
                if hist_size > 0:
                    self.opponent_assignment_idx[:, col] = np.random.randint(
                        0, hist_size, size=env.num_envs
                    )
        elif hist_size == 0:
            self.opponent_assignment_source[:, :] = True
            for col in range(self.num_selfplay_agents):
                if cfg_size > 0:
                    self.opponent_assignment_idx[:, col] = np.random.randint(
                        0, cfg_size, size=env.num_envs
                    )
        else:
            pick_config = np.random.rand(
                env.num_envs, self.num_selfplay_agents
            ) < float(self.opponent_pool_prob)
            self.opponent_assignment_source = pick_config
            for col in range(self.num_selfplay_agents):
                cfg_idxs = np.nonzero(pick_config[:, col])[0]
                if cfg_size > 0 and cfg_idxs.size > 0:
                    self.opponent_assignment_idx[cfg_idxs, col] = np.random.randint(
                        0, cfg_size, size=cfg_idxs.size
                    )
                hist_idxs = np.nonzero(~pick_config[:, col])[0]
                if hist_size > 0 and hist_idxs.size > 0:
                    self.opponent_assignment_idx[hist_idxs, col] = np.random.randint(
                        0, hist_size, size=hist_idxs.size
                    )
        lstm_states = deepcopy(self._last_lstm_states)
        selfplay_lstm_states = {}
        self._last_dones = None
        agent_dones = None
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (actions, values, log_probs) = ([], [], [])
                for ag in range(self.num_agents):
                    if ag in self.active_agents_idx:
                        if self._last_dones is not None:
                            episode_starts = th.tensor(
                                self._last_dones[:, ag],
                                dtype=th.float32,
                                device=self.device,
                            )
                        else:
                            episode_starts = th.ones(
                                obs_tensor.shape[0],
                                dtype=th.float32,
                                device=self.device,
                            )
                        (action, value, log_prob, lstm_states[ag]) = self.policy[
                            ag
                        ].forward(obs_tensor[:, ag], lstm_states[ag], episode_starts)
                        action = np.array(action.cpu(), dtype=np.float64)
                        actions.append(action)
                        values.append(value)
                        log_probs.append(log_prob)
                    else:
                        if n_steps == 0:
                            selfplay_lstm_states.clear()
                        agent_actions = np.zeros((env.num_envs, 4), dtype=np.float64)
                        col = self._selfplay_index_map[ag]
                        src_mask = self.opponent_assignment_source[:, col]
                        assigned_idx = self.opponent_assignment_idx[:, col]
                        idxs_pool = np.arange(env.num_envs)
                        if cfg_size > 0:
                            cfg_env_idxs = idxs_pool[np.nonzero(src_mask)[0]]
                            if cfg_env_idxs.size > 0:
                                cfg_assigned = assigned_idx[cfg_env_idxs]
                                unique_cfg = np.unique(cfg_assigned)
                                for cfg_id in unique_cfg:
                                    local_idxs = cfg_env_idxs[
                                        np.nonzero(cfg_assigned == cfg_id)[0]
                                    ]
                                    if local_idxs.size == 0:
                                        continue
                                    actual_pool_idx = self.selected_pool_indices[
                                        int(cfg_id)
                                    ]
                                    opp_policy = self.opponent_pool[actual_pool_idx]
                                    opp_policy.set_training_mode(False)
                                    if hasattr(opp_policy, "lstm_actor"):
                                        state_key = (ag, "cfg", int(cfg_id))
                                        if state_key not in selfplay_lstm_states:
                                            lstm = opp_policy.lstm_actor
                                            full_shape = (
                                                lstm.num_layers,
                                                env.num_envs,
                                                lstm.hidden_size,
                                            )
                                            selfplay_lstm_states[state_key] = (
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                            )
                                        (pi_h, pi_c, vf_h, vf_c) = selfplay_lstm_states[
                                            state_key
                                        ]
                                        local_lstm_state = RNNStates(
                                            (
                                                pi_h[:, local_idxs, :],
                                                pi_c[:, local_idxs, :],
                                            ),
                                            (
                                                vf_h[:, local_idxs, :],
                                                vf_c[:, local_idxs, :],
                                            ),
                                        )
                                        episode_starts_local = th.zeros(
                                            local_idxs.size,
                                            dtype=th.float32,
                                            device=self.device,
                                        )
                                        (
                                            action_tensor,
                                            _,
                                            _,
                                            new_lstm_state,
                                        ) = opp_policy.forward(
                                            obs_tensor[local_idxs, ag],
                                            local_lstm_state,
                                            episode_starts_local,
                                            deterministic=True,
                                        )
                                        action = action_tensor.cpu().numpy()
                                        pi_h[:, local_idxs, :] = new_lstm_state.pi[0]
                                        pi_c[:, local_idxs, :] = new_lstm_state.pi[1]
                                        vf_h[:, local_idxs, :] = new_lstm_state.vf[0]
                                        vf_c[:, local_idxs, :] = new_lstm_state.vf[1]
                                    else:
                                        (action, _) = opp_policy.predict(
                                            self._last_obs[local_idxs, ag],
                                            deterministic=True,
                                        )
                                    agent_actions[local_idxs] = action
                        if hist_size > 0:
                            hist_env_idxs = idxs_pool[np.nonzero(~src_mask)[0]]
                            if hist_env_idxs.size > 0:
                                ckpt_assigned = assigned_idx[hist_env_idxs]
                                ckpt_names = [
                                    checkpoints[int(i)] for i in ckpt_assigned
                                ]
                                unique_ck = np.unique(ckpt_names)
                                for ck_name in unique_ck:
                                    sel = [
                                        i
                                        for (i, v) in enumerate(ckpt_names)
                                        if v == ck_name
                                    ]
                                    local_idxs = hist_env_idxs[sel]
                                    if local_idxs.size == 0:
                                        continue
                                    fullpath = (
                                        os.path.join(policy_dir, ck_name)
                                        if policy_dir is not None
                                        else ck_name
                                    )
                                    if fullpath in local_loaded:
                                        opp_policy = local_loaded[fullpath]
                                    else:
                                        try:
                                            opp_cfg = self._load_opponent_run_config(
                                                fullpath
                                            )
                                            pol = load_teacher(
                                                fullpath,
                                                device=self.device,
                                                config=opp_cfg or self.env_cfg,
                                                env=self.env,
                                            )
                                            pol = pol.to(self.device)
                                            local_loaded[fullpath] = pol
                                            opp_policy = pol
                                        except Exception as e:
                                            warnings.warn(
                                                f"Failed to load historic checkpoint {fullpath}: {e}"
                                            )
                                            continue
                                    opp_policy.set_training_mode(False)
                                    if hasattr(opp_policy, "lstm_actor"):
                                        state_key = (ag, "hist", ck_name)
                                        if state_key not in selfplay_lstm_states:
                                            lstm = opp_policy.lstm_actor
                                            full_shape = (
                                                lstm.num_layers,
                                                env.num_envs,
                                                lstm.hidden_size,
                                            )
                                            selfplay_lstm_states[state_key] = (
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                                th.zeros(
                                                    full_shape, device=self.device
                                                ),
                                            )
                                        (pi_h, pi_c, vf_h, vf_c) = selfplay_lstm_states[
                                            state_key
                                        ]
                                        local_lstm_state = RNNStates(
                                            (
                                                pi_h[:, local_idxs, :],
                                                pi_c[:, local_idxs, :],
                                            ),
                                            (
                                                vf_h[:, local_idxs, :],
                                                vf_c[:, local_idxs, :],
                                            ),
                                        )
                                        episode_starts_local = th.zeros(
                                            local_idxs.size,
                                            dtype=th.float32,
                                            device=self.device,
                                        )
                                        (
                                            action_tensor,
                                            _,
                                            _,
                                            new_lstm_state,
                                        ) = opp_policy.forward(
                                            obs_tensor[local_idxs, ag],
                                            local_lstm_state,
                                            episode_starts_local,
                                            deterministic=True,
                                        )
                                        action = action_tensor.cpu().numpy()
                                        pi_h[:, local_idxs, :] = new_lstm_state.pi[0]
                                        pi_c[:, local_idxs, :] = new_lstm_state.pi[1]
                                        vf_h[:, local_idxs, :] = new_lstm_state.vf[0]
                                        vf_c[:, local_idxs, :] = new_lstm_state.vf[1]
                                    else:
                                        (action, _) = opp_policy.predict(
                                            self._last_obs[local_idxs, ag],
                                            deterministic=True,
                                        )
                                    agent_actions[local_idxs] = action
                        actions.append(agent_actions)
            actions = np.array(actions).transpose(1, 0, 2)
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
            clipped_actions = clipped_actions.reshape((env.num_envs, -1))
            (new_obs, rewards, dones, infos) = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos)
            n_steps += 1
            for ag in range(self.num_active_agents):
                if self._last_dones is None:
                    agent_dones = np.array(episode_starts.cpu().numpy())
                else:
                    agent_dones = self._last_dones[:, ag]
                rollout_buffer[ag].add(
                    self._last_obs[:, ag],
                    actions[:, ag],
                    rewards[:, ag],
                    agent_dones,
                    values[ag],
                    log_probs[ag],
                    lstm_states=self._last_lstm_states[ag],
                )
            self._last_dones = dones
            self._last_obs = new_obs
            self._last_lstm_states = lstm_states
        final_values = []
        with th.no_grad():
            for ag in range(self.num_active_agents):
                episode_starts = th.tensor(
                    dones[:, ag], dtype=th.float32, device=self.device
                )
                agent_values = self.policy[ag].predict_values(
                    obs_as_tensor(new_obs[:, ag], self.device),
                    lstm_states[ag].vf,
                    episode_starts,
                )
                final_values.append(agent_values)
        for ag in range(self.num_active_agents):
            self.rollout_buffer[ag].compute_returns_and_advantage(
                last_values=final_values[ag], dones=dones[:, ag]
            )
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        for ag in range(self.num_agents):
            self.policy[ag].set_training_mode(True)
        for ag in range(self.num_active_agents):
            self._update_learning_rate(self.policy[ag].optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        for ag in range(self.num_active_agents):
            entropy_losses = []
            (pg_losses, value_losses) = ([], [])
            clip_fractions = []
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer[ag].get(self.batch_size):
                    actions = rollout_data.actions
                    mask = rollout_data.mask > 1e-08
                    (values, log_prob, entropy) = self.policy[ag].evaluate_actions(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (
                            advantages[mask].std() + 1e-08
                        )
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1) > clip_range).float()[mask]
                    ).item()
                    clip_fractions.append(clip_fraction)
                    values_pred = values
                    value_loss = th.mean(
                        ((rollout_data.returns - values_pred) ** 2)[mask]
                    )
                    value_losses.append(value_loss.item())
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[mask])
                    else:
                        entropy_loss = -th.mean(entropy[mask])
                    entropy_losses.append(entropy_loss.item())
                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                    )
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = (
                            th.mean((th.exp(log_ratio) - 1 - log_ratio)[mask])
                            .cpu()
                            .numpy()
                        )
                        approx_kl_divs.append(approx_kl_div)
                    self.policy[ag].optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        self.policy[ag].parameters(), self.max_grad_norm
                    )
                    self.policy[ag].optimizer.step()
                    approx_kl_divs.append(
                        th.mean(rollout_data.old_log_prob - log_prob)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            if ag == 0:
                self._n_updates += self.n_epochs
            explained_var = explained_variance(
                self.rollout_buffer[ag].values.flatten(),
                self.rollout_buffer[ag].returns.flatten(),
            )
            self.logger.record("Train/entropy_loss_" + str(ag), np.mean(entropy_losses))
            self.logger.record(
                "Train/policy_gradient_loss_" + str(ag), np.mean(pg_losses)
            )
            self.logger.record("Train/value_loss_" + str(ag), np.mean(value_losses))
            self.logger.record("Train/approx_kl_" + str(ag), np.mean(approx_kl_divs))
            self.logger.record(
                "Train/clip_fraction_" + str(ag), np.mean(clip_fractions)
            )
            self.logger.record("Train/loss_" + str(ag), loss.item())
            self.logger.record("Train/explained_variance_" + str(ag), explained_var)
            if hasattr(self.policy[ag], "log_std"):
                self.logger.record(
                    "Train/std_" + str(ag),
                    th.exp(self.policy[ag].log_std).mean().item(),
                )
            self.logger.record(
                "Train/n_updates_" + str(ag), self._n_updates, exclude="tensorboard"
            )
            self.logger.record("Train/clip_range", clip_range)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: tuple = (10, 100, 10),
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        env_cfg: str = None,
    ) -> OnPolicyAlgorithm:
        return super(RecurrentPPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            env_cfg=env_cfg,
        )
