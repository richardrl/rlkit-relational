import torch
import rlkit.torch.pytorch_util as ptu
import numpy as np


def fetch_preprocessing(obs,
                        actions=None,
                        normalizer=None,
                        robot_dim=10,
                        object_dim=15,
                        goal_dim=3,
                        zero_state_preprocessing_fnx=False,
                        lop_state_dim=3,
                        mask=None,
                        return_combined_state=True,
                        **kwargs):
    """
    For Fetch robotics gym environment. Takes a flattened state and processes it into batched, normalized objects.

    :param obs: N x (nR + nB * nFb)
    :param actions
    :param robot_dim:
    :param object_dim: nFb
    :param num_objects:
    :param goal_dim:
    :param zero_state_preprocessing_fnx: Zero out state for testing.
    :return: N x nB x (nFb + nR). If in QValueCPA, concats actions to the left of the shared return
    """
    if len(obs.shape) == 3:
        obs = obs.squeeze(1)
    if lop_state_dim:
        obs = obs.narrow(1, 0, obs.size(1)-lop_state_dim) # Chop off the final 3 dimension of gripper position

    batch_size, environment_state_length = obs.size()
    if actions is not None:
        action_dim = actions.size(-1)
    else:
        action_dim = 0

    if zero_state_preprocessing_fnx:
        obs = torch.zeros(batch_size, environment_state_length).to(ptu.device)

    nB = (environment_state_length - robot_dim) / (object_dim + goal_dim)

    assert nB.is_integer(), (nB, environment_state_length, robot_dim, object_dim, goal_dim) # TODO: this checks if the lopped state still breaks down into the right object dimensions. The only worry here is whether the obs was messed up at the start of the function, e.g. the samples from the replay buffer incorrectly put the lopped state somewwhere.

    nB = int(nB)
    if mask is None:
        mask = torch.ones(obs.shape[0], nB).to(ptu.get_device())

    kwargs_state_length = robot_dim + object_dim * nB + goal_dim * nB
    assert kwargs_state_length == environment_state_length, F"{kwargs_state_length} != {environment_state_length}"

    # N x nR. From index 0 to shared dim per sample, we have the robot_state
    robot_state_flat = obs.narrow(1, 0, robot_dim)

    # assert (state_length - shared_dim - goal_state_dim) % block_feature_dim == 0, state_length - shared_dim - goal_state_dim

    # N x (nB x nFb)
    flattened_objects = obs.narrow(1, robot_dim, object_dim * nB)

    # -> N x nB x nFb
    batched_objects = flattened_objects.view(batch_size, nB, object_dim)

    # N x (nB x nFg) # TODO: perhaps add lop state dim
    flattened_goals = obs.narrow(1, robot_dim + nB * object_dim, nB * goal_dim)

    # -> N x nB x nFg
    batched_goals = flattened_goals.view(batch_size, nB, goal_dim)

    assert torch.eq(torch.cat((
                         robot_state_flat.view(batch_size, -1),
                         batched_objects.view(batch_size, -1),
                         batched_goals.view(batch_size, -1)), dim=1),
        obs).all()

    # Broadcast robot_state
    # -> N x nB x nR
    batch_shared = robot_state_flat.unsqueeze(1).expand(-1, nB, -1)

    # Concatenate with block_state
    # N x nB x (nFb + nR)
    # output_state = torch.cat((block_state, robot_state), dim=2)
    # return output_state

    # We can just consider the goals to be part of the block state, so we concat them together

    batch_objgoals = torch.cat((batched_objects, batched_goals), dim=-1)

    batch_shared = batch_shared.clone() * mask.unsqueeze(-1).expand_as(batch_shared)
    batch_objgoals = batch_objgoals.clone() * mask.unsqueeze(-1).expand_as(batch_objgoals)
    # assert torch.unique(batch_shared, dim=1).shape == torch.Size([batch_size, 1, robot_dim]), (
    # torch.unique(batch_shared, dim=1).shape, torch.Size([batch_size, 1, robot_dim]))

    if normalizer is not None:
        robot_singleobj_singlegoal = torch.cat((batch_shared, batch_objgoals), dim=-1).view(batch_size * nB, robot_dim + object_dim + goal_dim)

        # Single objects means, we flatten the nB dimension
        norm_singlerobot_singleobj_singlegoal, norm_actions = normalizer.normalize_all(robot_singleobj_singlegoal, actions)

        # Set these two variables to be the normalized versions
        norm_singlerobot, norm_singleobj_singlegoal = torch.split(norm_singlerobot_singleobj_singlegoal, [robot_dim, object_dim + goal_dim], dim=-1)

        # Turn single objects back into batches of nB objects
        norm_batchobjgoals = norm_singleobj_singlegoal.contiguous().view(batch_size, nB,  object_dim + goal_dim)
        norm_batchshared = norm_singlerobot.contiguous().view(batch_size, nB, robot_dim)
        # assert torch.unique(norm_batchshared, dim=1).shape == torch.Size([batch_size, 1, robot_dim]), (torch.unique(norm_batchshared, dim=1).shape, torch.Size([batch_size, 1, robot_dim]))

        batch_shared = norm_batchshared
        batch_objgoals = norm_batchobjgoals
        actions = norm_actions

    if actions is not None:
        batch_shared = torch.cat((actions.unsqueeze(1).expand(-1, nB, -1), batch_shared), dim=-1)

    assert batch_shared.shape == torch.Size([batch_size, nB, robot_dim + action_dim]), (batch_shared.shape, torch.Size([batch_size, nB, robot_dim + action_dim]))

    if return_combined_state:
        batched_combined_state = torch.cat((batch_shared, batch_objgoals), dim=-1)
        return batched_combined_state
    else:
        return batch_shared, batch_objgoals


def invert_fetch_preprocessing(batched_shared,
                               batched_objects_and_goals,
                               robot_dim=10,
                               object_dim=15,
                               goal_dim=3,
                               num_blocks=None,
                               **kwargs):
    """

    :param batched: (N * nB) x nShared, (N * nB) x (nObject + nGoal)
    :return:
    """
    N = batched_shared.size(0)
    batched_shared = batched_shared.view(N, num_blocks, batched_shared.size(-1))

    # Reduce over nB dimension...

    assert (batched_shared[:, 0, :].unsqueeze(1) == batched_shared).all(), batched_shared
    shared_flat = batched_shared[:, 0, :]

    individual_flat = batched_objects_and_goals.contiguous().view(N, num_blocks * (object_dim + goal_dim))

    # Returns N x (nR + nB * nOG)
    return torch.cat((shared_flat, individual_flat), dim=-1)


def get_masks(curr_num_blocks, max_num_blocks, path_len, keepdim=False):
    assert curr_num_blocks <= max_num_blocks
    if path_len > 1:
        masks = np.ones((path_len, curr_num_blocks))  # Num_blocks is the MAX num_blocks
        masks = np.pad(masks, ((0, 0), (0, int(max_num_blocks - curr_num_blocks))), "constant",
                       constant_values=((0, 0), (0, 0)))

    else:
        masks = np.ones(curr_num_blocks)
        masks = np.pad(masks, ((0, int(max_num_blocks - curr_num_blocks))), "constant",
                   constant_values=((0, 0)))
        if keepdim:
            masks = np.expand_dims(masks, axis=0)
    return masks


def pad_obs(obs, key, key_sizes, max_num_blocks=None, curr_num_blocks=None):
    """
    Pads the -1 dimension to 'max_num_blocks'
    :param obs: ndarray
    :param key:
    :param key_sizes:
    :param max_num_blocks:
    :param curr_num_blocks:
    :return:
    """
    if len(obs.shape) == 1:
        obs = np.expand_dims(obs, 0)
    if "goal" in key:
        lop_state = obs[:, -3:].copy()
        padded_obs = obs[:, :-3].copy()

        assert int(max_num_blocks - curr_num_blocks) >=0, int(max_num_blocks - curr_num_blocks)
        padded_obs = np.pad(padded_obs,
                            ((0, 0), (0, int(max_num_blocks - curr_num_blocks) * key_sizes[key])),
                            "constant", constant_values=((0, 0), (0, -999)))
        padded_obs = np.concatenate((padded_obs, lop_state), axis=-1).copy() # If max_num_blocks == curr_num_blocks, lopping and concatenating shouldn't do anything
    else:
        assert int(max_num_blocks - curr_num_blocks) >=0, int(max_num_blocks - curr_num_blocks)
        # print("obs shape")
        # print(obs.shape)
        if len(obs.shape) == 2:
            padded_obs = np.pad(obs, ((0, 0), (0, int(max_num_blocks - curr_num_blocks) * key_sizes[key])),
                                "constant", constant_values=((0, 0), (0, -999)))
        elif len(obs.shape) == 1:
            padded_obs = np.pad(obs, ((0, int(max_num_blocks - curr_num_blocks) * key_sizes[key])),
                                "constant", constant_values=((0, -999)))
        else:
            raise NotImplementedError

    return padded_obs