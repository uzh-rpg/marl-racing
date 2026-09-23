from typing import Callable, Tuple
from gym import spaces
import torch as th
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class MlpPolicySALSTM(MlpLstmPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        obs_dim_fixed: int,
        *args,
        **kwargs,
    ):
        self.obs_dim_fixed = obs_dim_fixed
        single_agent_obs_space = spaces.Box(
            low=observation_space.low[:obs_dim_fixed],
            high=observation_space.high[:obs_dim_fixed],
            shape=(obs_dim_fixed,),
            dtype=observation_space.dtype,
        )
        super().__init__(
            single_agent_obs_space, action_space, lr_schedule, *args, **kwargs
        )

    def _filter_observations(self, obs: th.Tensor) -> th.Tensor:
        return obs[..., : self.obs_dim_fixed]

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        filtered_obs = self._filter_observations(obs)
        return super().forward(filtered_obs, lstm_states, episode_starts, deterministic)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        filtered_obs = self._filter_observations(obs)
        return super().evaluate_actions(
            filtered_obs, actions, lstm_states, episode_starts
        )

    def predict_values(
        self, obs: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> th.Tensor:
        filtered_obs = self._filter_observations(obs)
        return super().predict_values(filtered_obs, lstm_states, episode_starts)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not isinstance(observation, th.Tensor):
            observation = th.as_tensor(observation, dtype=th.float32)
        filtered_obs = self._filter_observations(observation)
        return super().predict(
            filtered_obs.cpu().numpy(), state, episode_start, deterministic
        )
