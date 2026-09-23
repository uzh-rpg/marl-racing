import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union
import gym
import numpy as np
import torch
from flightgym import MarlNoCameraVecEnv_v0 as NoCameraVecEnv_v0
from ruamel.yaml import YAML
from src.models.marl.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean


class OnPolicyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[NoCameraVecEnv_v0, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        use_tanh_act: bool,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_tanh_act = use_tanh_act
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        if _init_setup_model:
            self._setup_model()

    def sample_checkpoint_powerlaw(self, checkpoints, alpha=0.9):
        if not checkpoints:
            return None
        n = len(checkpoints)
        weights = np.arange(1, n + 1) ** alpha
        probabilities = weights / weights.sum()
        return np.random.choice(checkpoints, p=probabilities)

    def train(self) -> None:
        raise NotImplementedError

    def update_statistics(self, iteration) -> None:
        time_elapsed = int((time.time_ns() - self.start_time) / 1000000000.0)
        frequency = int(self.num_timesteps / max(time_elapsed, 1))
        self.logger.record("Time/iterations", iteration, exclude="tensorboard")
        self.logger.record("Time/steps_per_second", frequency)
        self.logger.record(
            "Time/time_elapsed", f"{time_elapsed}s", exclude="tensorboard"
        )
        self.logger.record(
            "Time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if any((ep_info for ep_info in self.ep_info_buffer)):
            for ag in range(self.num_agents):
                self.logger.record(
                    f"Rollout/{ag}/ep_rew_mean",
                    safe_mean(
                        [
                            ep_info[ag]["r"]
                            for ep_info in self.ep_info_buffer
                            if ep_info and ag in ep_info
                        ]
                    ),
                )
                self.logger.record(
                    f"Rollout/{ag}/ep_len_mean",
                    safe_mean(
                        [
                            ep_info[ag]["l"]
                            for ep_info in self.ep_info_buffer
                            if ep_info and ag in ep_info
                        ]
                    ),
                )
        for ag in range(self.num_agents):
            for i in range(len(self.env.reward_names) - 1):
                reward = safe_mean(
                    [
                        ep_info[ag][self.env.reward_names[i]]
                        for ep_info in self.ep_info_buffer
                        if ep_info and ag in ep_info
                    ]
                )
                self.logger.record(f"Rewards/{ag}/{self.env.reward_names[i]}", reward)
        self.logger.dump(step=self.num_timesteps)

    def dump_setup_config(self, env_cfg):
        new_cfg_dir = self.logger.get_dir() + "/config.yaml"
        with open(new_cfg_dir, "w") as outfile:
            YAML().dump(env_cfg, outfile)
        flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
        track_cfg_path = flightmare_path / env_cfg["environment"]["tracks"]
        track_cfg = YAML().load(open(track_cfg_path, "r"))
        YAML().dump(track_cfg, open(self.logger.get_dir() + "/track.yaml", "w"))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 100),
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        env_cfg: dict = None,
    ) -> "OnPolicyAlgorithm":
        iteration = 0
        checkpoint_number = 0
        (total_timesteps, callback) = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name
        )
        self.dump_setup_config(env_cfg)
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            iteration += 1
            continue_training = True
            continue_training = continue_training and self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps
            )
            if not continue_training:
                break
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            if log_interval is not None and iteration % log_interval[0] == 0:
                self.update_statistics(iteration)
            self.train()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if iteration % 10 == 0:
                self.env.curriculum_update()
                self.curriculum_updated += 1
            if log_interval is not None and (
                iteration <= 1
                or (iteration % 100 == 0 and iteration < 500)
                or iteration % log_interval[1] == 0
            ):
                policy_path = self.logger.get_dir() + "/checkpoints/"
                if not os.path.exists(policy_path):
                    os.mkdir(policy_path)
                for ag in range(self.num_active_agents):
                    checkpoint_number += 1
                    self.policy[ag].save(policy_path + f"{checkpoint_number}.pth")
        callback.on_training_end()
        return self
