import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
import numpy as np
from gym.wrappers.monitor import Monitor

def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb"))
    policy = data['algorithm'].policy
    env = data['env']
    env = Monitor(env, force=True, directory="videos/", video_callable=lambda x:x)

    num_blocks = 1
    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    policy.train(False)
    paths = []
    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
            get_action_kwargs=dict(
                mask=np.ones((1, num_blocks)),
                deterministic=True
            ),
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


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
