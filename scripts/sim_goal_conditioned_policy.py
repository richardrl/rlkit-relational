import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
import numpy as np
from gym.wrappers.monitor import Monitor
import gym
import h5py
import os
from datetime import datetime
from rlkit.core.eval_util import get_generic_path_information



def is_solved(path, num_blocks):
    num_succeeded = 0
    goal_threshold = .05
    for block_id in range(num_blocks):
        if np.linalg.norm(path['full_observations'][-1]['achieved_goal'][block_id * 3:(block_id + 1) * 3] - path['full_observations'][-1]['desired_goal'][block_id * 3:(block_id + 1) * 3]) < goal_threshold:
            num_succeeded += 1
    return num_succeeded == num_blocks


def get_final_subgoaldist(env, path):
    if isinstance(env, Monitor):
        return sum(env.env.unwrapped.subgoal_distances(path['full_observations'][-1]['achieved_goal'], path['full_observations'][-1]['desired_goal']))
    else:
        return sum(env.unwrapped.subgoal_distances(path['full_observations'][-1]['achieved_goal'], path['full_observations'][-1]['desired_goal']))


def simulate_policy(args):
    # import torch
    # torch.manual_seed(6199)
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb"))
    policy = data['algorithm'].policy

    num_blocks = 2
    stack_only = True


    # env = data['env']
    # obs_type = "Dictimage" if args.save_image else "Dictstate"
    obs_type = "Dictstate"

    env = gym.make(F"FetchBlockConstruction_{num_blocks}Blocks_IncrementalReward_{obs_type}Obs_{args.image_size}Rendersize_{stack_only}Stackonly_SingletowerCase-v1")
    env.env.unwrapped.render_image_obs = args.save_image
    # env.env.unwrapped.render_image_size = args.image_size

    env = Monitor(env, force=True, directory="videos/", video_callable=lambda x:x)

    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    policy.train(False)

    if args.save_attn:
        # Create keys in h5py dictionary
        path_length = num_blocks * 50
        image_size = 128

        num_rel_blocks = 3
        num_selection_heads = 1

        scripts_dir = os.path.dirname(os.path.realpath(__file__))

        dt = datetime.now()

        f = h5py.File(scripts_dir + f"/data/trajectories{dt.strftime('%m_%d_%Y-%H_%M_%S_.%f')}.hdf5", "w")
        if args.save_image:
            image_dset = f.create_dataset("image_obs",
                                          shape=(0, path_length + 1, image_size, image_size, 3),
                                          dtype='i8',
                                          chunks=True,
                                          maxshape=(1000, path_length + 1, image_size, image_size, 3))

        obs_dset = f.create_dataset("obs",
                                    shape=(0, path_length, num_blocks*15 + 10),
                                    dtype='f8',
                                    chunks=True,
                                    maxshape=(1000, path_length, num_blocks*15 + 10))

        goals_dset = f.create_dataset("goals",
                                      shape=(0, path_length, num_blocks*3), # Chop off the GPFG 3
                                      dtype='f8',
                                      chunks=True,
                                      maxshape=(1000, path_length, num_blocks*3))

        actions_dset = f.create_dataset("actions",
                                        shape=(0, path_length, 4),
                                        dtype="f8",
                                        chunks=True,
                                        maxshape=(1000, path_length, 4))

        attn_dset = f.create_dataset("attn",
                                     shape=(0, path_length, num_blocks * num_rel_blocks + num_selection_heads, num_blocks),
                                     dtype="f8",
                                     chunks=True,
                                     maxshape=(1000, path_length, num_blocks * num_rel_blocks + num_selection_heads, num_blocks))

        rewards_dset = f.create_dataset("rewards",
                                        shape=(0, path_length, 1),
                                        dtype="f8",
                                        chunks=True,
                                        maxshape=(1000, path_length, 1))


    failures = []
    successes = []
    for path_idx in range(20):
        path = multitask_rollout(
            env,
            policy,
            max_path_length=num_blocks*50,
            animated=args.show,
            observation_key='observation',
            desired_goal_key='desired_goal',
            get_action_kwargs=dict(
                return_probs=True if args.save_attn else False,
                mask=np.ones((1, num_blocks)),
                deterministic=True
            ),
        )

        if not is_solved(path, num_blocks):
            failures.append(path)
            print(F"Failed {path_idx}")
        else:
            print(F"Succeeded {path_idx}")
            successes.append(path)

        if args.save_attn:
            # Save path to h5py file
            path_idx = obs_dset.shape[0]
            new_max_idx = path_idx + 1

            if args.save_image:
                image_dset.resize(new_max_idx, axis=0)
                image_dset[path_idx] = np.asarray([dic['image_observation'] for dic in path['full_observations']])

            attn_dset.resize(new_max_idx, axis=0)
            attn_dset[path_idx] = np.asarray([dic['attn_probs'] for dic in path['agent_infos']]).reshape(
                (path_length, num_blocks * num_rel_blocks + num_selection_heads, num_blocks))

            rewards_dset.resize(new_max_idx, axis=0)
            rewards_dset[path_idx] = np.asarray(path['rewards'])

            actions_dset.resize(new_max_idx, axis=0)
            actions_dset[path_idx] = np.asarray(path['actions'])

            obs_dset.resize(new_max_idx, axis=0)
            obs_dset[path_idx] = np.asarray([dic['observation'] for dic in path['observations']])

            goals_dset.resize(new_max_idx, axis=0)
            goals_dset[path_idx] = np.asarray(path['goals'][:, :-3])

        # if hasattr(env, "log_diagnostics"):
                #     env.log_diagnostics(paths)
                # if hasattr(env, "get_diagnostics"):
                #     for k, v in env.get_diagnostics(paths).items():
                #         logger.record_tabular(k, v)
                # logger.dump_tabular()
    if args.save_attn:
        f.close()

    print(f"Success rate {len(successes)/(len(successes) + len(failures))}")
    path_info = get_generic_path_information(successes + failures, num_blocks=num_blocks)
    print(path_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=np.inf,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save_attn', action='store_true')
    parser.add_argument('--image_size', type=int, default=128, choices=[42, 84, 128, 800])

    args = parser.parse_args()

    simulate_policy(args)
