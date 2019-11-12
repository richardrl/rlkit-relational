"""
Wrap Algorithm such that we can swap between environments in multi-task setting
"""
import numpy as np
import gym
from rlkit.torch.her.her import HerTwinSAC


class MultiEnvWrapperHerTwinSAC(HerTwinSAC):
    def __init__(self, env_names, her_kwargs, tsac_kwargs, *args, env_probabilities=None, **kwargs):
        """
        :param env_names: List of environment names
        """
        HerTwinSAC.__init__(self,
            *args,
            her_kwargs=her_kwargs,
            tsac_kwargs=tsac_kwargs,
            **kwargs)
        self.env_names = env_names
        self.env_probabilities = env_probabilities
        assert [gym.make(self.env_names[i]).action_space == gym.make(self.env_names[0]).action_space for i in range(len(self.env_names))]

    def get_new_env(self):
        env_idx = np.random.choice(np.arange(len(self.env_names)), p=self.env_probabilities if self.env_probabilities else None)
        return gym.make(self.env_names[env_idx]), self.env_names[env_idx]

    def _handle_rollout_ending(self):
        super()._handle_rollout_ending()
        self.training_env, env_name = self.get_new_env()
        print(f"Loaded {env_name}")
        # self.training_env = pickle.loads(pickle.dumps(self.env))
        self.replay_buffer.env = self.training_env