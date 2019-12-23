import argparse
import pickle
import joblib
import gym

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.launchers.launcher_util import run_experiment
import robotics_recorder
from rlkit.data_management.path_builder import PathBuilder
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import torch
import numpy as np
from rlkit.launchers.config import get_infra_settings

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def resume_training(variant):
    data = pickle.load(open(variant['trained_file'], "rb"))
    algorithm = data['algorithm']

    env = gym.make(variant['env_id'])
    algorithm.training_env = pickle.loads(pickle.dumps(env))
    algorithm.env = env

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    for key in algorithm.__dict__:
        if "optimizer" in key:
            getattr(algorithm, key).comm = MPI.COMM_WORLD
            network = key.split("_")[0]
            if network == "alpha":
                # getattr(algorithm, key).reconnect_params(getattr(algorithm, network).parameters())
                algorithm.alpha_optimizer.reconnect_params([algorithm.log_alpha])
            else:
                getattr(algorithm, key).reconnect_params(getattr(algorithm, network).parameters())

    if not hasattr(algorithm.replay_buffer, "_masks"):
        algorithm.replay_buffer._masks = np.zeros((algorithm.replay_buffer.max_size, num_blocks))
        algorithm.replay_buffer.max_num_blocks = num_blocks
        algorithm.replay_buffer.key_sizes = dict(observation=15,
                              desired_goal=3,
                              achieved_goal=3)
        algorithm.demonstration_policy = None
        algorithm.replay_buffer.demonstration_buffer = None
        algorithm._old_table_keys = None

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    docker_img = "latest"

    filename = "/home/richard/s3_files/11-27-sequentialtransfer-recurrentFalse-stack5-stack6-numrelblocks3-nqh1-dockimglatest-rewardIncremental-stackonlyTrue/11-27-sequentialtransfer_recurrentFalse_stack5_stack6_numrelblocks3_nqh1_dockimglatest_rewardIncremental_stackonlyTrue-1574894953750/11-27-sequentialtransfer_recurrentFalse_stack5_stack6_numrelblocks3_nqh1_dockimglatest_rewardIncremental_stackonlyTrue_2019_11_27_22_56_32_0000--s-23990/itr_250.pkl"

    num_blocks = int(input("Num blocks: "))
    # assert "relational_preloadstack1" in filename
    import re
    if "nqh" in filename:
        nqh = int(re.search("(?<=nqh)(\d+)(?=_)", filename).group(0))
    else:
        nqh = 0
    if "numrelblocks" in filename:
        num_relational_blocks = int(re.search("(?<=numrelblocks)(\d+)", filename).group(0))
    else:
        num_relational_blocks = 0

    prev_stackonly = bool(input("Prev stack only: "))

    variant = dict(
        trained_file=filename,
        env_id=F"FetchBlockConstruction_{num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{prev_stackonly}Stackonly_SingletowerCase-v1",
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
        new_replay_buffer=False,
        doodad_docker_image = F"richardrl/fbc:{docker_img}",
        gpu_doodad_docker_image = F"richardrl/fbc:{docker_img}",
    )

    mode="here_no_doodad"
    instance_type = "c5.18xlarge"
    num_parallel_processes = get_infra_settings(mode, instance_type)['num_parallel_processes']

    prefix = input("Prefix: ")

    run_experiment(
        resume_training,
        exp_prefix=f"resume-{prefix}-numrelblocks{num_relational_blocks}-nqh{nqh}-numblocks{num_blocks}-stackonly{prev_stackonly}_dockimg{docker_img}",  # Make sure no spaces..
        region="us-west-2",
        mode=mode,
        variant=variant,
        gpu_mode=False,
        spot_price=5,
        snapshot_mode='gap_and_last',
        snapshot_gap=100,
        num_exps_per_instance=1,
        instance_type=instance_type,
        python_cmd=F"mpirun --allow-run-as-root -np {num_parallel_processes} python"
    )
