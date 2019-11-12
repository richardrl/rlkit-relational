import torch
import rlkit.torch.pytorch_util as ptu
import numpy as np

from rlkit.data_management.normalizer import Normalizer, FixedNormalizer
from rlkit.torch.relational.relational_util import fetch_preprocessing, invert_fetch_preprocessing
from rlkit.torch.core import PyTorchModule


class TorchNormalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std


class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v / std

    def denormalize(self, v):
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v * std


class CompositeNormalizer:
    """
    Useful for normalizing different data types e.g. when using the same normalizer for the Q function and the policy function
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 reshape_blocks=False,
                 fetch_kwargs=dict(),
                 **kwargs):
        # self.save_init_params(locals())
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.obs_normalizer = TorchNormalizer(self.observation_dim, **kwargs)
        self.action_normalizer = TorchNormalizer(self.action_dim)
        self.reshape_blocks = reshape_blocks
        self.kwargs = kwargs
        self.fetch_kwargs = fetch_kwargs

    def normalize_all(
            self,
            flat_obs,
            actions):
        """

        :param flat_obs:
        :param actions:
        :return:
        """
        if flat_obs is not None:
            # If reshape blocks
            # First, reshape
            # Then, flatten along batch
            # Normalize
            # Then bring back to original shape
            # if self.reshape_blocks:
            #     before_obs = flat_obs.clone()
            #     before_size = flat_obs.size()
            #     batched_robot_state, batched_objects_and_goals = fetch_preprocessing(flat_obs)
            #
            #     N, nB, nR = batched_robot_state.size()
            #     nOG = batched_objects_and_goals.size(-1)
            #
            #     # -> (N * nB, nR + nOG)
            #     single_objects = torch.cat((batched_robot_state, batched_objects_and_goals), dim=-1).view(N * nB, nR + nOG)
            #     normalized_batched = self.obs_normalizer.normalize(single_objects)
            #     flat_obs = invert_fetch_preprocessing(*torch.split(normalized_batched, [nR, nOG], dim=-1),
            #                                           nB=nB,
            #                                           N=N,
            #                                           nOG=nOG,
            #                                           nR=nR
            #                                           )
            #     assert flat_obs.size() == before_size, (flat_obs.size(), before_size)
            #
            #     """
            #     Test loop. If we don't normalize, batch_to_flat should invert the preprocessing
            #     """
            #     batched_rs, batched_og = torch.split(single_objects, [nR, nOG], dim=-1)
            #     assert torch.all(torch.eq(batched_rs, batched_robot_state)), (batched_rs, batched_robot_state)
            #     assert torch.all(torch.eq(batched_og, batched_objects_and_goals)), (batched_og, batched_objects_and_goals)
            #     test_obs = invert_fetch_preprocessing(batched_rs, batched_og,
            #                                           nB=nB,
            #                                           N=N,
            #                                           nOG=nOG,
            #                                           nR=nR
            #                                           )
            #     assert torch.all(torch.eq(before_obs[:,nR], test_obs[:,nR]))
            #     before_obj_goal = before_obs[:,nR:]
            #     test_obj_goal = test_obs[:,nR:]
            #     assert torch.all(torch.eq(before_obj_goal, test_obj_goal)), before_obj_goal-test_obj_goal
            #     assert torch.all(torch.eq(before_obs, test_obs)), before_obs - test_obs
            # else:
            flat_obs = self.obs_normalizer.normalize(flat_obs)
        if actions is not None:
            actions = self.action_normalizer.normalize(actions)
        return flat_obs, actions

    def update(self, data_type, v, mask=None):
        """
        Takes in tensor and updates numpy array
        :param data_type:
        :param v:
        :return:
        """
        if data_type == "obs":
            # Reshape_blocks: takes flat, turns batch, normalizes batch, updates the obs_normalizer...
            if self.reshape_blocks:
                batched_robot_state, batched_objects_and_goals = fetch_preprocessing(v, mask=mask, return_combined_state=False, **self.fetch_kwargs)
                N, nB, nR = batched_robot_state.size()
                v = torch.cat((batched_robot_state, batched_objects_and_goals), dim=-1).view(N * nB, -1)
                if mask is not None:
                    v = v[mask.view(N * nB).to(dtype=torch.bool)]

            # if self.lop_state_dim:
            #     v = v.narrow(-1, -3, 3)
            self.obs_normalizer.update(ptu.get_numpy(v))
        elif data_type == "actions":
            self.action_normalizer.update(ptu.get_numpy(v))
        else:
            raise("data_type not set")