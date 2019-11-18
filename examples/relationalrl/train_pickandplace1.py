import gym
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.torch.her.her import HerTwinSAC
from rlkit.torch.data_management.normalizer import CompositeNormalizer
from rlkit.torch.optim.mpi_adam import MpiAdam
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.relational.networks import *
from rlkit.torch.relational.modules import *
from rlkit.torch.networks import Mlp
from rlkit.launchers.config import get_ec2_settings


def experiment(variant):
    import fetch_block_construction

    env = gym.make(variant['env_id'])
    env.unwrapped.render_image_obs = False
    if variant['set_max_episode_steps']:
        env.env._max_episode_steps = variant['set_max_episode_steps']

    action_dim = env.action_space.low.size
    robot_dim = 10
    object_dim = 15
    goal_dim = 3

    shared_normalizer = CompositeNormalizer(object_dim + shared_dim + goal_dim,
                                            action_dim,
                                            default_clip_range=5,
                                            reshape_blocks=True,
                                            fetch_kwargs=dict(
                                                lop_state_dim=3,
                                                object_dim=object_dim,
                                                goal_dim=goal_dim
                                            ))

    object_total_dim = robot_dim + object_dim + goal_dim

    policy = ReNNPolicy(
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_total_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        graph_module_kwargs=dict(
            object_total_dim=object_total_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        readout_module_kwargs=dict(
            embedding_dim=64,
            layer_norm=layer_norm,
            num_heads=variant['pooling_heads']
        ),
        proj_kwargs=dict(
            hidden_sizes=mlp_hidden_sizes,
            obs_dim=embedding_dim,
            action_dim=action_dim,
            output_activation=torch.tanh,
            layer_norm=layer_norm,
        ),
        num_graph_modules=num_graph_modules
    )

    qf1 = ReNN(
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_total_dim + action_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        graph_module_kwargs=dict(
            object_total_dim=object_total_dim + action_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        readout_module_kwargs=dict(
            embedding_dim=64,
            layer_norm=layer_norm,
            num_heads=variant['pooling_heads']
        ),
        proj_class=Mlp,
        proj_kwargs=dict(
            hidden_sizes=mlp_hidden_sizes,
            output_size=1,
            input_size=embedding_dim,
            layer_norm=layer_norm
        ),
        num_graph_modules=num_graph_modules
    )

    qf2 = ReNN(
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_total_dim + action_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        graph_module_kwargs=dict(
            object_total_dim=object_total_dim + action_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        readout_module_kwargs=dict(
            embedding_dim=64,
            layer_norm=layer_norm,
            num_heads=variant['pooling_heads']
        ),
        proj_class=Mlp,
        proj_kwargs=dict(
            hidden_sizes=mlp_hidden_sizes,
            output_size=1,
            input_size=embedding_dim,
            layer_norm=layer_norm
        ),
        num_graph_modules=num_graph_modules
    )

    vf = ReNN(
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_total_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        graph_module_kwargs=dict(
            object_total_dim=object_total_dim,
            embedding_dim=64,
            layer_norm=layer_norm
        ),
        readout_module_kwargs=dict(
            embedding_dim=64,
            layer_norm=layer_norm,
            num_heads=variant['pooling_heads']
        ),
        proj_class=Mlp,
        proj_kwargs=dict(
            hidden_sizes=mlp_hidden_sizes,
            output_size=1,
            input_size=embedding_dim,
            layer_norm=layer_norm
        ),
        num_graph_modules=num_graph_modules
    )

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
        ),
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    docker_img = "latest"
    # docker_img = "negativereward_gpfg_uniformxy_rotctrl"

    if "rotctrl" in docker_img:
        action_dim = 8
        object_dim = 16
        goal_dim = 7
    else:
        action_dim = 4
        object_dim = 15
        goal_dim = 3

    shared_dim = 10
    num_graph_modules = 3
    num_query_heads = 1

    embedding_dim = 64
    layer_norm = True
    num_blocks = 1
    stackonly = False
    num_epochs_per_eval = 10

    max_path_len = 50*num_blocks

    max_episode_steps = 50*num_blocks

    mlp_hidden_sizes=[64, 64, 64]

    module = __import__("rlkit.torch.relational.modules", fromlist=['modules'])

    mode = "here_no_doodad"

    instance_type = "c5.18xlarge"
    settings_dict = get_ec2_settings(mode, instance_type)

    variant = dict(
        algo_kwargs=dict(
            num_epochs=3000 * 10,
            max_path_length=max_path_len,
            batch_size=256,
            discount=0.98,
            save_algorithm=True,
            # collection_mode="online",
            # num_updates_per_env_step=1,
            collection_mode='batch',
            num_updates_per_epoch=50*num_blocks,
            num_steps_per_epoch=50*num_blocks, # Do one episode per block
            num_steps_per_eval=50*num_blocks * 10, # Do ten episodes per eval
            num_epochs_per_eval=10, # One episode per epoch, so this is roughly 10 episodes per eval * number of parallel episodes...
            num_epochs_per_param_save=10 * 5,
            num_gpus=settings_dict['num_gpus'],
            # min_num_steps_before_training=10000,

            #SAC args start
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            grad_clip_max=1000
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e5),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
            num_relational=num_graph_modules,
            num_heads=num_query_heads
        ),
        render=False,
        env_id=F"FetchBlockConstruction_{num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{stackonly}Stackonly_SingletowerCase-v1", # TODO: make sure FalseStackonly so it goes in the air
        doodad_docker_image=F"richardrl/fbc:{docker_img}",
        gpu_doodad_docker_image=F"richardrl/fbc:{docker_img}",
        save_video=False,
        save_video_period=50,
        num_relational_blocks=num_graph_modules,
        set_max_episode_steps=max_episode_steps,
        mlp_hidden_sizes=mlp_hidden_sizes,
        num_query_heads=num_query_heads,
        action_dim=action_dim,
        goal_dim=goal_dim,
        embedding_dim=embedding_dim,
        her_kwargs=dict(
            exploration_masking=True
        ),
        pooling_heads=2
    )

    test_prefix = "test" if mode == "here_no_doodad" else "pickandplace1"
    print(f"\nprefix: {test_prefix}")

    run_experiment(
        experiment,
        exp_prefix=F"{test_prefix}_stack{num_blocks}_numgraphmodules{num_graph_modules}_nqh{num_query_heads}_dockimg{docker_img}_{stackonly}stackonly",  # Make sure no spaces..
        region="us-west-2",
        mode=mode,
        variant=variant,
        gpu_mode=settings_dict['gpu_mode'],
        spot_price=10,
        snapshot_mode='gap_and_last',
        snapshot_gap=100,
        num_exps_per_instance=1,
        instance_type=instance_type,
        python_cmd=F"mpirun --allow-run-as-root -np {settings_dict['num_parallel_processes']} python"
    )
