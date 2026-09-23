import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
from stable_baselines3.common import utils
from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import update_learning_rate


class BaseAlgorithm(SB3BaseAlgorithm):
    def _update_learning_rate(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        self.logger.record(
            "Train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        self.start_time = time.time_ns()
        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=1000)
            self.ep_success_buffer = deque(maxlen=1000)
        if self.action_noise is not None:
            self.action_noise.reset()
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )
        callback = self._init_callback(callback, progress_bar)
        return (total_timesteps, callback)

    def _update_info_buffer(
        self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        if dones is None:
            dones = np.array([False] * len(infos))
        for (idx, info) in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info and maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success and maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
