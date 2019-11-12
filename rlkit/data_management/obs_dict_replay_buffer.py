import numpy as np
from gym.spaces import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.torch.relational.relational_util import get_masks, pad_obs


class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/


    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            internal_keys=None,
            observation_key='observation',
            achieved_goal_key='achieved_goal',
            desired_goal_key='desired_goal',
            num_relational=None,
            num_heads=1,
            max_num_blocks=None,
            demonstration_buffer=None,
            skip_future_obs_idx=False
    ):
        """

        :param max_size:
        :param env:
        :param fraction_goals_rollout_goals: Default, no her.
        :param fraction_goals_env_goals: What fraction of goals are sampled
        "from the environment" assuming that the environment has a "sample
        goal" method. The remaining resampled goals are resampled using the
        "future" strategy, described in Hindsight Experience Replay.
        :param internal_keys: Extra keys in the observation dictoary to save.
        Mostly for debugging.
        :param observation_key:
        :param desired_goal_key:
        :param achieved_goal_key:
        """
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        assert isinstance(env.observation_space, Dict)
        assert fraction_goals_env_goals >= 0
        assert fraction_goals_rollout_goals >= 0
        assert fraction_goals_env_goals + fraction_goals_rollout_goals <= 1.0
        self.max_size = max_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key

        self._action_dim = env.action_space.low.size
        self._actions = np.zeros((max_size, self._action_dim))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        ob_spaces = self.env.observation_space.spaces
        for key in self.ob_keys_to_save + internal_keys:
            assert key in ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, ob_spaces[key].low.size), dtype=type)

        self._top = 0 # Used as pointer for adding new samples
        self._size = 0 # Used to define maximum sample range

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

        # self._log_pi = np.zeros((max_size, 1))
        # self._log_pi_block2 = np.zeros((max_size, 1))
        #
        # self._logstd = np.zeros((max_size, 1))
        # self._stable_stacked = np.zeros((max_size, 1))

        self.num_heads = num_heads
        self.num_relational = num_relational
        self.max_num_blocks = max_num_blocks if max_num_blocks is not None else self.env.unwrapped.num_blocks # set num_blocks = the blocks of the first environment... which should have max blocks
        self._masks = np.zeros((max_size, self.max_num_blocks))
        # if self.num_relational:
        #     self._attn_softmax = np.zeros((max_size, self.num_blocks * self.num_heads * self.num_relational + 1, self.num_blocks))
        #
        # self._logstd_b1 = np.zeros((max_size, 1))  # Index represents the sample idx, value represents the entropy
        # self._logstd_b2 = np.zeros((max_size, 1))
        self.key_sizes = dict(observation=15,
                              desired_goal=3,
                              achieved_goal=3)
        self.demonstration_buffer = demonstration_buffer # A demonstration buffer of fixed size
        self.skip_future_obs_idx = skip_future_obs_idx

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def pad_all_obs(self, curr_num_blocks=None, max_num_blocks=None):
        for key in ['observation', 'desired_goal', 'achieved_goal']:
            self._obs[key] = pad_obs(self._obs[key], key=key, key_sizes=self.key_sizes, max_num_blocks=max_num_blocks, curr_num_blocks=curr_num_blocks)
            self._next_obs[key] = pad_obs(self._next_obs[key], key=key,key_sizes=self.key_sizes, max_num_blocks=max_num_blocks, curr_num_blocks=curr_num_blocks)
        # self._masks = (self._masks, max_num_blocks=max_num_blocks, curr_num_blocks=curr_num_blocks)
        self._masks = get_masks(curr_num_blocks=curr_num_blocks, max_num_blocks=max_num_blocks, path_len=self._obs['observation'].shape[0])


    def merge(self, other_replay_buffer):
        for key in ['observation', 'desired_goal', 'achieved_goal']:
            assert len(self._obs[key].shape) == len(other_replay_buffer._obs[key].shape)
            assert np.all(self._obs[key].shape[1] == other_replay_buffer._obs[key].shape[1])

            self._obs[key] = np.concatenate((self._obs[key], other_replay_buffer._obs[key]), axis=0)

            self._next_obs[key] = np.concatenate((self._next_obs[key], other_replay_buffer._next_obs[key]), axis=0)

        assert len(self._masks.shape) == len(other_replay_buffer._masks.shape)
        assert np.all(self._masks.shape[1] == other_replay_buffer._masks.shape[1])
        self._masks = np.concatenate((self._masks, other_replay_buffer._masks))
        self._actions = np.concatenate((self._actions, other_replay_buffer._actions))
        self._terminals = np.concatenate((self._terminals, other_replay_buffer._terminals))
        self._top = (self._top + other_replay_buffer._size) % self.max_size
        self._size = min(self._size + other_replay_buffer._size, self.max_size)

    def add_path(self, path, curr_num_blocks=None):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        mask = path['mask']
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs,
                                self.ob_keys_to_save + self.internal_keys)

        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]

                self._masks[buffer_slice] = mask[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = pad_obs(obs[key][path_slice], key,  key_sizes=self.key_sizes, max_num_blocks=self.max_num_blocks, curr_num_blocks=curr_num_blocks)
                    self._next_obs[key][buffer_slice] = pad_obs(next_obs[key][path_slice], key,  key_sizes=self.key_sizes, max_num_blocks=self.max_num_blocks, curr_num_blocks=curr_num_blocks)

                # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals

            if not hasattr(self, "_masks"):
                print("_masks not found, creating empty one...")

            self._masks[slc] = mask

            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = pad_obs(obs[key], key, key_sizes=self.key_sizes, max_num_blocks=self.max_num_blocks, curr_num_blocks=curr_num_blocks)
                self._next_obs[key][slc] = pad_obs(next_obs[key], key,  key_sizes=self.key_sizes, max_num_blocks=self.max_num_blocks, curr_num_blocks=curr_num_blocks)

            if not self.skip_future_obs_idx:
                for i in range(self._top, self._top + path_len):
                    self._idx_to_future_obs_idx[i] = np.arange(
                        i, self._top + path_len
                    )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)
        # return np.arange(0, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self.env.unwrapped.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            resampled_goals[num_rollout_goals:last_env_goal_idx] = (
                env_goals[self.desired_goal_key]
            )
        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice.
                # Makes you wonder what random.choice is doing...
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            resampled_goals[-num_future_goals:] = (
                self._next_obs[self.achieved_goal_key][future_obs_idxs]
            )

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]

        # if self.num_relational:
        #     new_attn_softmax = self._attn_softmax[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """
        if hasattr(self.env, 'compute_rewards'):
            new_rewards = self.env.compute_rewards(
                new_actions,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]

        new_masks = self._masks[indices]

        # Below: indices represent order, value represents index of sample
        # argsort in ascending order from smallest (most negative) to largest entropy
        # argsort_ = np.argsort(self._logstd_b1[:self._size], axis=0)
        # sample_idxs_2_rank = np.zeros_like(self._logstd_b1[:self._size])
        #
        # for i in range(self._size):
        #     sample_idxs_2_rank[argsort_[i]] = i
        #
        # logstd_block1_normalized_ranking = sample_idxs_2_rank[indices] / self._size
        # assert (logstd_block1_normalized_ranking <= 1).all() and (logstd_block1_normalized_ranking >= 0).all()

        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
            'masks': new_masks,
            # 'logstd': self._logstd[indices],
            # 'stable_stacked': self._stable_stacked[indices],
            # 'logstd_block1_normalized_ranking': logstd_block1_normalized_ranking,
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Input is list of dicts. This operation pulls out the key in each dict and combines the values into a new list mapped to the original key. A new dictionary is formed with these key -> list mappings.
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }


def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    # for obs_key, obs in obs_dict.items():
    #     if 'image' in obs_key and obs is not None:
    #         obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


# def normalize_image(image):
#     assert image.dtype == np.uint8
#     return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
