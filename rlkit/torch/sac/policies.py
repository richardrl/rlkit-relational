import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp

from rlkit.torch.data_management.normalizer import CompositeNormalizer
from rlkit.torch.data_management.normalizer import TorchNormalizer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.relational.relational_util import fetch_preprocessing, invert_fetch_preprocessing
import gtimer as gt

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False, **kwargs):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=True,
            **kwargs
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        gt.stamp("Tanhnormal_forward")
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.layer_norm and i < len(self.fcs):
                h = self.layer_norms[i](h)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)

            gt.stamp("tanhnormal_pre")
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
            gt.stamp("tanhnormal_post")

        log_std = log_std.sum(dim=1, keepdim=True)

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def query_action_logpi(self,
                           obs,
                           action):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.layer_norm and i < len(self.fcs):
                h = self.layer_norms[i](h)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        pre_tanh_value = None
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
            pre_tanh_value=pre_tanh_value
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class NormalizedTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            *args,
            normalizer: TorchNormalizer = None,
            **kwargs
    ):
        assert normalizer is not None
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.normalizer = normalizer

    def forward(self, obs, **kwargs):
        obs = self.normalizer.normalize(obs)
        return super().forward(obs, **kwargs)


class CompositeNormalizedTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            *args,
            composite_normalizer: CompositeNormalizer = None,
            lop_state_dim=None,
            preprocessing_kwargs=None,
            num_blocks=None,
            demo_composite_normalizer: CompositeNormalizer = None,
            **kwargs
    ):
        assert composite_normalizer is not None
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.composite_normalizer = composite_normalizer
        self.demo_composite_normalizer = demo_composite_normalizer

        self.forward_activations = dict()
        self.lop_state_dim = lop_state_dim
        # ptu.register_forward_hooks(self, self.forward_activations)
        self.preprocessing_kwargs = preprocessing_kwargs
        self.num_blocks = num_blocks

    def forward(self, observations, blindfold_focus_block=None, demo_normalizer=False, **kwargs):
        # if self.lop_state_dim:
        #     obs = obs.narrow(1, 0,
        #                      obs.size(1) - self.lop_state_dim)  # Chop off the final 3 dimension of gripper position
        # obs, _ = self.composite_normalizer.normalize_all(obs, None)
        # return super().forward(obs, **kwargs)

        shared_state, object_goal_state = fetch_preprocessing(observations,
                                                              normalizer=self.demo_composite_normalizer if demo_normalizer else self.composite_normalizer,
                                                              **self.preprocessing_kwargs)

        if blindfold_focus_block is not None:
            assert isinstance(blindfold_focus_block, int)
            # assert blindfold_focus_block >= 0 and blindfold_focus_block < self.num_blocks
            object_goal_state = object_goal_state.narrow(1, blindfold_focus_block, 1)
            shared_state = shared_state.narrow(1, blindfold_focus_block, 1)
            assert object_goal_state.size(1) == 1, object_goal_state.size(1)
            assert shared_state.size(1) == 1, shared_state.size(1)

        flat_input = invert_fetch_preprocessing(shared_state, object_goal_state, num_blocks=self.num_blocks, **self.preprocessing_kwargs)

        if blindfold_focus_block is None:
            assert observations.size(-1) - self.lop_state_dim == flat_input.size(-1), (observations.size(-1) - self.lop_state_dim, flat_input.size(-1))
            assert len(observations.size()) == len(flat_input.size()) == 2

        return super().forward(flat_input, **kwargs)


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)

    def set_num_steps_total(self, n_env_steps_total):
        return self.stochastic_policy.set_num_steps_total(n_env_steps_total)

    def parameters(self):
        return self.stochastic_policy.parameters()

    def to(self, device):
        self.stochastic_policy.to(device)


class FlattenTanhGaussianPolicy(TanhGaussianPolicy):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)
