"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.torch.her.her import HerTwinSAC
from rlkit.torch.optim.mpi_adam import MpiAdam
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.relational.networks import *
import pickle
from mpi4py import MPI
from rlkit.launchers.config import get_infra_settings
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
import gym


def experiment(variant):
    import fetch_block_construction

    env = gym.make(variant['env_id'])
    data = pickle.load(open(
        filename,
        "rb"))

    algorithm = data['algorithm']
    policy = algorithm.policy #TODO: make sure output goes through tanh. Apparently it is TANH but then.. why explosion
    qf1 = algorithm.qf1
    qf2 = algorithm.qf2
    vf = algorithm.vf
    log_alpha = data['algorithm'].log_alpha

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    es = EpsilonGreedy(env.action_space, prob_random_action=prob_action)
    exp_policy = PolicyWrappedWithExplorationStrategy(exploration_strategy=es,
                                                      policy=policy)

    algorithm = HerTwinSAC(
        her_kwargs=dict(
            observation_key='observation',
            desired_goal_key='desired_goal',
            **variant['her_kwargs']
        ),
        tsac_kwargs=dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            policy=policy,
            optimizer_class=MpiAdam,
            exploration_policy=exp_policy,
        ),
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )

    algorithm.log_alpha = torch.tensor(ptu.get_numpy(log_alpha), dtype=torch.float32, requires_grad=True, device=ptu.device)

    #TODO: we may need to reload the alpha

    algorithm.to(ptu.device) # Convert all preloaded weights to GPU here
    algorithm.optim_to(ptu.device)

    algorithm.alpha_optimizer.comm = MPI.COMM_WORLD
    algorithm.alpha_optimizer.reconnect_params([algorithm.log_alpha])
    assert algorithm.alpha_optimizer.param_groups[0]['params'][0] is algorithm.log_alpha
    algorithm.qf1_optimizer.comm = MPI.COMM_WORLD
    algorithm.qf1_optimizer.reconnect_params(algorithm.qf1.parameters())

    algorithm.qf2_optimizer.comm = MPI.COMM_WORLD
    algorithm.qf2_optimizer.reconnect_params(algorithm.qf2.parameters())

    algorithm.vf_optimizer.comm = MPI.COMM_WORLD
    algorithm.vf_optimizer.reconnect_params(algorithm.vf.parameters())

    algorithm.policy_optimizer.comm = MPI.COMM_WORLD
    algorithm.policy_optimizer.reconnect_params(algorithm.policy.parameters())

    algorithm.train()


if __name__ == "__main__":
    action_dim = 4
    object_dim = 15
    shared_dim = 10

    modes = ["ec2", "here_no_doodad", "local_docker"]
    mode = modes[int(input(f"\nMode: {modes}\n"))]
    print(F"Mode selected {mode}. ")

    docker_img = "latest"

    num_blocks = int(input("\nNum blocks: "))
    print(f'\n{num_blocks} selected.')
    num_epochs_per_eval = 10

    instance_type = "c5.18xlarge"
    ec2_settings = get_infra_settings(mode, instance_type)
    num_gpus = ec2_settings['num_gpus']
    num_parallel_processes = ec2_settings['num_parallel_processes']
    gpu_mode = ec2_settings['gpu_mode']

    prob_action = .1
    stackonly = bool(int(input("\nStack only: ")))
    print(f'\n{stackonly} selected.')

    filename = "/home/richard/rlkit-relational/examples/relationalrl/pkls/stack1/pickandplace1/11-23-pickandplace1-stack1-numrelblocks3-nqh1-dockimglatest-Falsestackonly-recurrentTrue/11-23-pickandplace1_stack1_numrelblocks3_nqh1_dockimglatest_Falsestackonly_recurrentTrue-1574500092573/11-23-pickandplace1_stack1_numrelblocks3_nqh1_dockimglatest_Falsestackonly_recurrentTrue_2019_11_23_09_14_11_0000--s-17380/itr_4800.pkl"

    print(F"\nFile name: {filename}")

    # assert "relational_preloadstack1" in filename
    import re
    if "nqh" in filename:
        nqh = int(re.search("(?<=nqh)(\d+)(?=_)", filename).group(0))
    else:
        nqh = 1
    if "numrelblocks" in filename:
        num_relational_blocks = int(re.search("(?<=numrelblocks)(\d+)", filename).group(0))

    else:
        num_relational_blocks = 3

    print(F"\nNum_rel_blocks: {num_relational_blocks}")

    if "stackonly" in filename:
        so = re.search("(?<=stackonly)(True|False)(_|-)", filename)
        if so is None:
            so = re.search("(?<=[_,-])(True|False)stackonly", filename)
        prev_stackonly = so.group(1)

    variant = dict(
        algo_kwargs=dict(
            num_epochs=3000 * 10,
            max_path_length=50*num_blocks,
            batch_size=256,
            discount=0.98,
            save_algorithm=True,
            # collection_mode="online",
            # num_updates_per_env_step=1,
            collection_mode='batch', # TODO: set these settings from now on
            num_updates_per_epoch=50*num_blocks if num_blocks <= 6 else 300 + 15*(num_blocks - 6),
            num_steps_per_epoch=50*num_blocks if num_blocks <= 6 else 300 + 15*(num_blocks - 6), # Do one episode per block
            num_steps_per_eval=50*num_blocks * 10 if num_blocks <= 6 else 300 + 15*(num_blocks - 6) * 10, # Do ten episodes per eval
            num_epochs_per_eval=num_epochs_per_eval, # One episode per epoch, so this is roughly 10 episodes per eval * number of parallel episodes...
            num_epochs_per_param_save=num_epochs_per_eval * 5,
            num_gpus=num_gpus,

            #SAC args start
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            # target_entropy=4,
            min_num_steps_before_training=50*num_blocks*10 if num_blocks <= 6 else (300 + 15*(num_blocks - 6))*10
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e5),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
            num_relational=num_relational_blocks,
            num_heads=nqh
        ),
        render=False,
        env_id=F"FetchBlockConstruction_{num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{stackonly}Stackonly_SingletowerCase-v1",
        doodad_docker_image=F"richardrl/fbc:{docker_img}",
        gpu_doodad_docker_image=F"richardrl/fbc:{docker_img}",
        save_video=False,
        save_video_period=50,
        num_relational_blocks=num_relational_blocks,
        num_query_heads=nqh,
        grad_clip_max=1000,
        exploration_policy="default",
        prob_action=prob_action,
        filename=filename,
        instance_type=instance_type,
        her_kwargs=dict(
            exploration_masking=True
        ),
    )

    test_prefix = "test_sequentialtransfer" if mode == "here_no_doodad" else "sequentialtransfer"
    print(f"\nTest prefix: {test_prefix}")

    prev_block_num = input("\nPrev number of blocks: ")
    if not prev_block_num:
        prev_block_num = num_blocks if (prev_stackonly == "False" and stackonly) else num_blocks - 1

    run_experiment(
        experiment,
        exp_prefix=F"{test_prefix}_stack{prev_block_num}"
        F"_stack{num_blocks}_numrelblocks{num_relational_blocks}_nqh{nqh}_dockimg{docker_img}_probaction{prob_action}_stackonly{stackonly}",  # Make sure no spaces..
        region="us-west-2",
        mode=mode,
        variant=variant,
        gpu_mode=gpu_mode,
        spot_price=5,
        snapshot_mode='gap_and_last',
        snapshot_gap=num_epochs_per_eval,
        num_exps_per_instance=1,
        instance_type=instance_type,
        python_cmd=F"mpirun --allow-run-as-root -np {num_parallel_processes} python"
    )