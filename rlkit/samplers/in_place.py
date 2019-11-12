from rlkit.samplers.util import rollout
from rlkit.samplers.rollout_functions import multitask_rollout
import numpy as np


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length, randomize_env=False, alg=None):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        assert max_samples >= max_path_length, "Need max_samples >= max_path_length"
        self.randomize_env = randomize_env
        self.alg = alg

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, rollout_type="multitask"):
        paths = []
        n_steps_total = 0
        while n_steps_total + self.max_path_length <= self.max_samples:
            if self.randomize_env:
                self.env, env_name = self.alg.get_new_env()
                print(f"Evaluating {env_name}")
            if rollout_type == "multitask":
                path = multitask_rollout(
                    self.env,
                    self.policy,
                    max_path_length=self.max_path_length,
                    animated=False,
                    observation_key='observation',
                    desired_goal_key='desired_goal',
                    get_action_kwargs=dict(
                        return_stacked_softmax=False,
                        mask=np.ones((1, self.env.unwrapped.num_blocks)),
                        deterministic=True
                    )
                )
            else:
                path = rollout(
                    self.env, self.policy, max_path_length=self.max_path_length
                )
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
