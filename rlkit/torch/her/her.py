import numpy as np
import torch

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.her.her_replay_buffer import RelabelingReplayBuffer
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.sac.twin_sac import TwinSAC
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch.relational.relational_util import get_masks


class HER(TorchRLAlgorithm):
    """
    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            observation_key=None,
            desired_goal_key=None,
            exploration_masking=False,
    ):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.exploration_masking = exploration_masking

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
            mask=None
    ):
        assert mask is not None
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            mask=mask,
        )

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path, curr_num_blocks=self.env.unwrapped.num_blocks)
        self._exploration_paths.append(path)

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations'] # Np ndarray
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path, curr_num_blocks=self.training_env.unwrapped.num_blocks)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation, **kwargs):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        new_obs = np.hstack((
            observation[self.observation_key],
            observation[self.desired_goal_key],
        ))
        kwargs = dict()
        # if hasattr(self, "exploration_masking") and self.exploration_masking:
        mask = np.ones((1, self.training_env.unwrapped.num_blocks)) # Num_blocks is the MAX num_blocks
        # masks = np.pad(masks, ((0,0), (0, int(self.replay_buffer.max_num_blocks - self.env.unwrapped.num_blocks))), "constant", constant_values=((0,0), (0, 0)))
        kwargs['mask'] = mask
        return self.exploration_policy.get_action(new_obs, **kwargs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            from rlkit.envs.multi_env_wrapper import MultiEnvWrapperHerTwinSAC

            if isinstance(self, MultiEnvWrapperHerTwinSAC):
                self.env, env_name = self.get_new_env()
                print(f"Evaluating {env_name}")

            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        get_action_kwargs = dict()
        # if not hasattr(self, "exploration_masking") or self.exploration_masking:
        # masks = np.pad(masks, ((0,0), (0, int(self.replay_buffer.max_num_blocks - self.env.unwrapped.num_blocks))), "constant", constant_values=((0,0), (0, 0)))
        get_action_kwargs['mask'] = get_masks(self.env.unwrapped.num_blocks, self.replay_buffer.max_num_blocks, 1, keepdim=True)

        return multitask_rollout(
            self.env,
            self.policy,
            self.max_path_length,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
            get_action_kwargs=get_action_kwargs,
            max_num_blocks=self.replay_buffer.max_num_blocks,
            cur_num_blocks=self.env.unwrapped.num_blocks
        )


class HerTd3(HER, TD3):
    def __init__(
            self,
            *args,
            her_kwargs,
            td3_kwargs,
            **kwargs
    ):
        HER.__init__(self, **her_kwargs)
        TD3.__init__(self, *args, **kwargs, **td3_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )


class HerSac(HER, SoftActorCritic):
    def __init__(
            self,
            *args,
            her_kwargs,
            sac_kwargs,
            **kwargs
    ):
        HER.__init__(self, **her_kwargs)
        SoftActorCritic.__init__(self, *args, **kwargs, **sac_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)


class HerDdpg(HER, DDPG):
    def __init__(
            self,
            *args,
            her_kwargs,
            ddpg_kwargs,
            **kwargs
    ):
        HER.__init__(self, **her_kwargs)
        DDPG.__init__(self, *args, **kwargs, **ddpg_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )


class HerTwinSAC(HER, TwinSAC):
    def __init__(
            self,
            *args,
            her_kwargs,
            tsac_kwargs,
            **kwargs
    ):
        HER.__init__(self, **her_kwargs)
        TwinSAC.__init__(self, *args, **kwargs, **tsac_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)