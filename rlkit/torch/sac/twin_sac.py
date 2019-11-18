import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

from rlkit.data_management.path_builder import PathBuilder


try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import inspect
from torch import distributions
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import gtimer as gt
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle
from rlkit.torch.optim.mpi_adam import MpiAdam


class TwinSAC(TorchRLAlgorithm):
    """
    SAC with the twin architecture from TD3.
    """
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=True,
            soft_target_tau=1e-2,
            policy_update_period=1,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,

            eval_policy=None,
            exploration_policy=None,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            observation_key=None,
            alpha_lr=1e-3,
            zero_observation_in_take_step=False,
            alpha=0,
            gpu_id=None,
            entropy_penalty=None,
            entropy_penalty_coeff=0,
            grad_clip_max=100,
            log_pi_t_coeff=10,
            policy_bc_weightings=None,
            demonstration_policy_pkl=None,
            **kwargs
    ):
        if eval_policy is None:
            if eval_deterministic:
                eval_policy = MakeDeterministic(policy)
            else:
                eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=exploration_policy or policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf

        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            if MPI:
                if ptu.get_mode() == "gpu_opt":
                    self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=F"cuda:{self.gpu_id}")
                else:
                    self.log_alpha = ptu.zeros(1, requires_grad=True)

                self.alpha_optimizer = optimizer_class(
                    [self.log_alpha],
                    lr=alpha_lr,
                    gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
                )
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = vf.copy()
        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()

        # if MPI:
        #     assert optimizer_class is MpiAdam
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        )

        if self.observation_key is None:
            self.observation_key = observation_key

        self.zero_observation_in_take_step = zero_observation_in_take_step
        self.alpha = alpha

        if optimizer_class is MpiAdam:
            self._sync_optimizers()

        self.entropy_penalty = entropy_penalty
        self.entropy_penalty_coeff = entropy_penalty_coeff
        self.grad_clip_max = grad_clip_max

        self.logpi_coeff = log_pi_t_coeff
        self.digraphs_for_debugging = []
        # self.optimizers = [self.qf1_optimizer, self.qf2_optimizer, self.vf_optimizer, self.policy_optimizer, self.alpha_optimizer]
        if "demo_batch_size" in kwargs:
            self.demo_batch_size = kwargs['demo_batch_size']

        if self.replay_buffer.demonstration_buffer is not None:
            assert policy_bc_weightings is not None
        self.policy_bc_weightings = policy_bc_weightings

        self.demonstration_policy = None
        if demonstration_policy_pkl is not None:
            pkl = pickle.load(open(demonstration_policy_pkl, 'rb'))
            if "policy" in pkl:
                self.demonstration_policy = pkl['policy']
            elif "algorithm" in pkl:
                self.demonstration_policy = pkl['algorithm'].policy
            else:
                raise NotImplementedError

    def _take_step_in_env(self, observation):
        # Enable zero'ing out observation for debugging purposes
        # TODO: zero goals too?
        if self.zero_observation_in_take_step:
            observation[self.observation_key] = np.zeros_like(observation[self.observation_key])
        return super()._take_step_in_env(observation)

    def _do_training(self):
        batch = self.get_batch()
        gt.stamp('get_batch')

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations'].detach()
        actions = batch['actions']
        next_obs = batch['next_observations']

        mask = batch['masks'].detach()
        kwargs = dict()
        kwargs['mask'] = mask

        if self.policy.input_module is not None:
            self.policy.input_module.normalizer.update("obs", obs, mask=mask)
            self.policy.input_module.normalizer.update("actions", actions, mask=mask)
        gt.stamp('update_normalizer')

        q1_pred = self.qf1(obs, actions=actions, **kwargs)
        q2_pred = self.qf2(obs, actions=actions, **kwargs)
        v_pred = self.vf(obs, **kwargs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True,
                                     **kwargs)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        gt.stamp('forward')

        """
        Alpha Loss (if applicable)
        """
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            # alpha = 1
            alpha_loss = 0
            alpha = self.alpha
        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs, **kwargs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        assert q1_pred.size() == q_target.detach().size()
        assert q2_pred.size() == q_target.detach().size()

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Prioritized Experience Replay Stuff
        """
        td_error = q_target - q1_pred
        abs_td_error = np.abs(ptu.get_numpy(td_error))
        self.eval_statistics['Absolute Td Error'] = abs_td_error.mean()

        """
        VF Loss
        """
        q_new_actions = torch.min(
            self.qf1(obs, actions=new_actions, **kwargs),
            self.qf2(obs, actions=new_actions, **kwargs),
        )
        v_target = q_new_actions - alpha*log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        gt.stamp('compute_losses')

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        clip_grad_norm_(self.qf1.parameters(), self.grad_clip_max)
        self.qf1_optimizer.step()

        gt.stamp('qf1_loop')

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        clip_grad_norm_(self.qf2.parameters(), self.grad_clip_max)
        self.qf2_optimizer.step()

        gt.stamp('qf2_loop')

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        clip_grad_norm_(self.vf.parameters(), self.grad_clip_max)
        self.vf_optimizer.step()

        gt.stamp('vf_loop')

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (alpha*log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                    log_pi * (alpha*log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

            policy_loss = policy_loss + policy_reg_loss

            """
            BC Cloning Loss
            """
            if self.demonstration_policy is not None:
                assert self.replay_buffer.demonstration_buffer is None
                demo_actions = self.demonstration_policy(obs,
                                                         reparameterize=self.train_policy_with_reparameterization,
                                                         return_log_prob=True,
                                                         demo_normalizer=False,
                                                         deterministic=False,
                                                         **kwargs)[0]
                bc_loss = self.qf_criterion(new_actions, demo_actions.detach())

            if self.replay_buffer.demonstration_buffer is not None:
                demos = self.replay_buffer.demonstration_buffer.random_batch(self.demo_batch_size)
                demos = {f'demo_{k}': v for k, v in demos.items()}
                demo_obs = ptu.from_numpy(demos['demo_observations'])
                demo_goals = ptu.from_numpy(demos['demo_resampled_goals'])

                demo_obsgoals = torch.cat((demo_obs, demo_goals), dim=1)

                demo_actions = ptu.from_numpy(demos['demo_actions'])
                demo_predicted_actions = self.policy(demo_obsgoals,
                                             reparameterize=self.train_policy_with_reparameterization,
                                             return_log_prob=True,
                                                     demo_normalizer=True,
                                             **kwargs)[0]
                bc_loss = self.qf_criterion(demo_predicted_actions, demo_actions.detach())
                policy_loss = self.policy_bc_weightings['rl_lambda']* policy_loss + self.policy_bc_weightings['bc_lambda']* bc_loss
            gt.stamp("policy_loss_forward")

            self.policy_optimizer.zero_grad()
            clip_grad_norm_(self.policy.parameters(), self.grad_clip_max)
            policy_loss.backward()
            self.policy_optimizer.step()

            # DEBUG LOOP below
            if self._n_train_steps_total % (self.num_updates_per_train_call * 10) == 0:
                try:
                    from torchviz import make_dot, make_dot_from_trace
                    tmp = {f'policy_{k}': v for k, v in dict(self.policy.named_parameters()).items()}
                    tmp.update({f'qf1_{k}': v for k, v in dict(self.qf1.named_parameters()).items()})
                    tmp.update({f'qf2_{k}': v for k, v in dict(self.qf2.named_parameters()).items()})
                    tmp.update({f'vf_{k}': v for k, v in dict(self.vf.named_parameters()).items()})
                    tmp.update(dict(log_alpha=self.log_alpha))

                    dot = make_dot(policy_loss, params=tmp)
                    dot.format = "svg"
                    from pathlib import Path
                    path = Path("/home/richard/Pictures/dots/")
                    # path.mkdir(parents=True)
                    dot.render(f"{path.absolute()}/dot{self._n_train_steps_total}_numnodes{len(dot.body)}")
                    import re

                    # self.digraphs_for_debugging.append([_ for _ in dot.body])
                    if len(self.digraphs_for_debugging) > 1:
                        with open(f"{path.absolute()}/dot{self._n_train_steps_total}_numnodes{len(dot.body)}_diag.txt", mode='w') as f:
                            diff = np.setdiff1d(self.digraphs_for_debugging[-1], self.digraphs_for_debugging[-2])
                            f.write(diff)
                except ModuleNotFoundError as e:
                    print(e)


            gt.stamp('policy_loop')
            # assert gcc.all_params_unchanged(self.policy.block2_evaluator)

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.replay_buffer.demonstration_buffer is not None or self.demonstration_policy is not None:
                self.eval_statistics['BC Loss'] = ptu.get_numpy(bc_loss)

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            # Gradient norms
            network_names = ['QF1', "VF", "Policy"]
            networks = [self.qf1, self.vf, self.policy]
            for (name, network) in zip(network_names, networks):
                for key, value in ptu.layerwise_model_gradient_norm(network).items():
                    if "_evaluator" not in key:
                        self.eval_statistics[F"{name}.{key} Gradient Norm"] = value if type(value) == float else np.mean(value)


    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]

    @property
    def networks_names(self):
        return [
            "policy",
            "qf1",
            "qf2",
            "vf",
            "target_vf"
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        # snapshot['qf1'] = self.qf1
        # snapshot['qf2'] = self.qf2
        # snapshot['policy'] = self.policy
        # snapshot['vf'] = self.vf
        # snapshot['target_vf'] = self.target_vf
        snapshot['algorithm'] = self
        return snapshot

    def _sync_optimizers(self):
        self.qf1_optimizer.sync()
        self.qf2_optimizer.sync()
        self.vf_optimizer.sync()
        self.alpha_optimizer.sync()
        self.policy_optimizer.sync()

    def optim_to(self, device=None):
        if device is None:
            device = ptu.device
        self.log_alpha = self.log_alpha.to(device=device)
        self.alpha_optimizer.to(device=device)
        self.qf1_optimizer.to(device=device)
        self.qf2_optimizer.to(device=device)
        self.vf_optimizer.to(device=device)
        self.policy_optimizer.to(device=device)