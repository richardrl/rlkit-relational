import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
import numpy as np
from gym.wrappers.monitor import Monitor
import gym


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

    num_blocks = 6
    stack_only = True


    # env = data['env']
    env = gym.make(F"FetchBlockConstruction_{num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{stack_only}Stackonly_AllCase-v1")

    env = Monitor(env, force=True, directory="videos/", video_callable=lambda x:x)

    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    policy.train(False)
    failures = []
    successes = []
    for path_idx in range(100):
        path = multitask_rollout(
            env,
            policy,
            max_path_length=num_blocks*50,
            animated=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
            get_action_kwargs=dict(
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
        # if hasattr(env, "log_diagnostics"):
        #     env.log_diagnostics(paths)
        # if hasattr(env, "get_diagnostics"):
        #     for k, v in env.get_diagnostics(paths).items():
        #         logger.record_tabular(k, v)
        # logger.dump_tabular()
    print(f"Success rate {len(successes)/(len(successes) + len(failures))}")
    from rlkit.core.eval_util import get_generic_path_information
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
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
