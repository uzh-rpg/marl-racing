import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv


def reorder_and_reshape_extra_info(
    extra_info, extra_info_names, sorted_extra_info_names, n_envs, num_agents
):
    name_to_index = {name: idx for (idx, name) in enumerate(extra_info_names)}
    original_indices = [
        name_to_index[sorted_name] for sorted_name in sorted_extra_info_names
    ]
    reshaped_extra_info = (
        extra_info[:, original_indices]
        .reshape(n_envs, len(sorted_extra_info_names) // num_agents, num_agents)
        .transpose(0, 2, 1)
    )
    return reshaped_extra_info


class MarlFlightEnvVec(VecEnv):
    def __init__(self, impl, num_agents=1, num_selfplay_agents=0):
        self.wrapper = impl
        self.num_agents = num_agents
        self.num_selfplay_agents = num_selfplay_agents
        self.act_dim = self.wrapper.getActDim() // self.num_agents
        self.obs_dim_single = self.wrapper.getObsDim() // self.num_agents
        self.obs_dim = self.obs_dim_single
        self.rew_dim = self.wrapper.getRewDim() * self.num_agents
        self.max_num_opponents = self.wrapper.getMaxNumAgents()
        self.variable_feature_dim = self.wrapper.getOtherAgentObsDim()
        self.obs_dim_single_fixed = (
            self.obs_dim_single - self.max_num_opponents * self.variable_feature_dim
        )
        self.obs_dim_single_variable = (
            self.max_num_opponents * self.variable_feature_dim
        )
        self._observation_space = spaces.Box(
            np.ones(self.obs_dim) * -np.Inf,
            np.ones(self.obs_dim) * np.Inf,
            dtype=np.float64,
        )
        self._action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )
        self._observation = np.zeros(
            (self.num_envs, self.num_agents, self.obs_dim), dtype=np.float64
        )
        self._reward_components = np.zeros(
            (self.num_envs, self.num_agents, self.rew_dim // self.num_agents),
            dtype=np.float64,
        )
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._extra_info_names = self.wrapper.getExtraInfoNames()
        self._sorted_extra_info_names = sorted(self._extra_info_names, reverse=True)[
            ::-1
        ]
        self._sorted_extra_info_names_transpose = (
            np.array(self._sorted_extra_info_names)
            .reshape(-1, self.num_agents)
            .transpose()
        )
        self.reward_names = self.wrapper.getRewardNames()
        self._extra_info = np.zeros(
            [
                self.num_envs,
                self.num_agents,
                len(self._extra_info_names) // self.num_agents,
            ],
            dtype=np.float64,
        )
        self.rewards = [
            [[] for _ in range(self.num_agents)] for _ in range(self.num_envs)
        ]
        self.sum_reward_components = np.zeros(
            [self.num_envs, self.num_agents, self.rew_dim // self.num_agents - 1],
            dtype=np.float64,
        )
        self._quadstate = np.zeros(
            [self.num_envs, self.num_agents * 45], dtype=np.float64
        )

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def step(self, action):
        if action.ndim <= 3:
            action = action.reshape((self.num_envs, -1))
        next_observation = np.zeros(
            [self.num_envs, self.num_agents * self.obs_dim_single], dtype=float
        )
        self._reward_components = self._reward_components.reshape(self.num_envs, -1)
        self._extra_info = self._extra_info.reshape(
            (self.num_envs, len(self._extra_info_names))
        )
        action = np.ascontiguousarray(action, dtype=np.float64)
        next_observation = np.ascontiguousarray(next_observation, dtype=np.float64)
        self._reward_components = np.ascontiguousarray(
            self._reward_components, dtype=np.float64
        )
        self._done = np.ascontiguousarray(self._done, dtype=bool)
        self._extra_info = np.ascontiguousarray(self._extra_info, dtype=np.float64)
        self.wrapper.step(
            action,
            next_observation,
            self._reward_components,
            self._done,
            self._extra_info,
        )
        self._reward_components = self._reward_components.reshape(
            (self.num_envs, self.num_agents, self.rew_dim // self.num_agents)
        )
        self._extra_info = reorder_and_reshape_extra_info(
            self._extra_info,
            self._extra_info_names,
            self._sorted_extra_info_names,
            self.num_envs,
            self.num_agents,
        )
        next_observation = next_observation.reshape(
            (self.num_envs, self.num_agents, self.obs_dim_single)
        )
        self._observation = next_observation
        obs = self._observation
        if len(self._extra_info_names) != 0:
            info = []
            extra_info_reshaped = self._extra_info.reshape(
                self.num_envs, self.num_agents, -1
            )
            for i in range(self.num_envs):
                info_n_env = {
                    j: dict(
                        zip(
                            self._sorted_extra_info_names_transpose[j],
                            extra_info_reshaped[i, j],
                        )
                    )
                    for j in range(self.num_agents)
                }
                info.append(info_n_env)
            info = [{"extra_info": info[i]} for i in range(self.num_envs)]
        else:
            info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            info[i]["episode"] = {}
            epinfo = {}
            for agent_idx in range(self.num_agents):
                self.rewards[i][agent_idx].append(
                    self._reward_components[i, agent_idx, -1]
                )
                self.sum_reward_components[i][
                    agent_idx, : self.rew_dim // self.num_agents - 1
                ] += self._reward_components[
                    i, agent_idx, : self.rew_dim // self.num_agents - 1
                ]
                if info[i]["extra_info"][agent_idx]["done_" + str(agent_idx)]:
                    eprew = sum(self.rewards[i][agent_idx])
                    eplen = len(self.rewards[i][agent_idx])
                    epinfo[agent_idx] = {"r": eprew, "l": eplen}
                    for j in range(self.rew_dim // self.num_agents - 1):
                        epinfo[agent_idx][
                            self.reward_names[j]
                        ] = self.sum_reward_components[i][agent_idx, j]
                        self.sum_reward_components[i][agent_idx, j] = 0.0
                    info[i]["episode"][agent_idx] = epinfo[agent_idx]
                    self.rewards[i][agent_idx].clear()
        dones = []
        for i in range(self.num_envs):
            dones.append(
                [
                    info[i]["extra_info"][ag]["done_" + str(ag)]
                    for ag in range(self.num_agents)
                ]
            )
        dones_per_agent = np.array(dones)
        return (
            obs,
            self._reward_components[:, :, -1].copy(),
            dones_per_agent.copy(),
            info.copy(),
        )

    def reset(self, random_mode=True, init_pos_offset=None):
        self._reward_components = np.zeros(
            (self.num_envs, self.num_agents, self.rew_dim // self.num_agents),
            dtype=np.float64,
        )
        self.obs_dim = self.wrapper.getObsDim() // self.num_agents
        if init_pos_offset is None:
            init_pos_offset = np.zeros((self.num_envs, 3))
        else:
            init_pos_offset = np.asarray(init_pos_offset, dtype=np.float64)
            init_pos_offset = np.ascontiguousarray(init_pos_offset)
        if init_pos_offset is None:
            self.wrapper.setTestMode(not random_mode)
        elif init_pos_offset.ndim == 2:
            self.wrapper.setTestModeOffset(not random_mode, init_pos_offset)
        elif init_pos_offset.ndim == 3:
            for i in range(self.num_agents):
                self.wrapper.setTestModeOffsetAgent(
                    not random_mode, init_pos_offset[i], i
                )
        else:
            raise ValueError("Invalid init_pos_offset shape")
        next_observation = np.zeros(
            [self.num_envs, self.wrapper.getObsDim()], dtype=float
        )
        self.wrapper.reset(next_observation)
        next_observation = next_observation.reshape(
            (self.num_envs, self.num_agents, self.obs_dim_single)
        )
        self._observation = next_observation
        obs = self._observation
        return obs

    def get_quad_state(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate.astype(np.float64)

    def curriculum_update(self, iteration=None):
        if iteration is None:
            self.wrapper.curriculumUpdate()
        else:
            self.wrapper.curriculumUpdate(iteration)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        raise RuntimeError("This method is not implemented")

    def close(self):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("Use the batched environment methods directly")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
