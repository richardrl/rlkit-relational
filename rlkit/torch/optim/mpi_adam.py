# Adapted from https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam.py

import rlkit.torch.optim.util as U
import torch
from torch.optim.optimizer import Optimizer
import math
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.core.serializable import Serializable
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MpiAdam(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 scale_grad_by_procs=True,
                 comm=None,
                 gpu_id=0):
        # Serializable.quick_init(self,
        #                         locals())
        super().__init__(params, dict())
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        total_params = sum([U.num_elements(param) for param in U.get_flat_params(self.param_groups)])
        if ptu.get_mode() == "gpu_opt":
            assert gpu_id is not None
            self.m = torch.zeros(total_params, dtype=torch.float32).to(device=F"cuda:{gpu_id}")
            self.v = torch.zeros(total_params, dtype=torch.float32).to(device=F"cuda:{gpu_id}")
        elif not ptu.get_mode(): #CPU is false
            self.m = torch.zeros(total_params, dtype=torch.float32)
            self.v = torch.zeros(total_params, dtype=torch.float32)
        else:
            print(ptu.get_mode())
            raise NotImplementedError
        self.t = 0
        self.set_params_from_flat = U.SetFromFlat(self.param_groups)
        self.get_params_as_flat = U.GetParamsFlat(self.param_groups)
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm

    def __getstate__(self):
        # d = Serializable.__getstate__(self)
        # d = dict()
        d = super().__getstate__()
        d['lr'] = self.lr
        d['beta1'] = self.beta1
        d['beta2'] = self.beta2
        d['epsilon'] = self.epsilon
        d['scale_grad_by_procs'] = self.scale_grad_by_procs
        d["m"] = self.m.clone()
        d["v"] = self.v.clone()
        d["t"] = self.t
        return d

    def __setstate__(self, d):
        # Serializable.__setstate__(self, d)
        super().__setstate__(d)
        if "lr" in d.keys():
            self.lr = d['lr']
        else:
            self.lr = 3E-4
        self.beta1 = d['beta1']
        self.beta2 = d['beta2']
        self.epsilon = d['epsilon']
        self.scale_grad_by_procs = d['scale_grad_by_procs']
        self.m = d["m"]
        self.v = d["v"]
        self.t = d["t"]

    def reset_state(self, gpu_id=0):
        self.m = torch.zeros_like(self.m, dtype=torch.float32).to(device=F"cuda:{gpu_id}")
        self.v = torch.zeros_like(self.v, dtype=torch.float32).to(device=F"cuda:{gpu_id}")

    def reconnect_params(self, params):
        super().__init__(params, dict()) # This does not alter the optimizer state m or v
        self.reinit_flat_operators()

    def reinit_flat_operators(self):
        self.set_params_from_flat = U.SetFromFlat(self.param_groups)
        self.get_params_as_flat = U.GetParamsFlat(self.param_groups)

    def step(self, closure=None):
        """
        Aggregate and reduce gradients across all threads
        :param closure:
        :return:
        """
        # self.param_groups updated on the GPU, stepped, then moved back to its own thread
        localg = U.get_flattened_grads(self.param_groups)
        if self.t % 100 == 0:
            self.check_synced()
        if localg.device.type == "cpu":
            localg = localg.detach().numpy()
        else:
            localg = localg.cpu().detach().numpy()
        if self.comm is not None:
            globalg = np.zeros_like(localg)
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
            if localg.shape[0] > 1 and self.comm.Get_size() > 1:
                assert not (localg == globalg).all()
            globalg = ptu.from_numpy(globalg, device=torch.device(ptu.get_device()))
        else:
            globalg = ptu.from_numpy(localg, device=torch.device(ptu.get_device()))

        self.t += 1
        a = self.lr * math.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step_update = (- a) * self.m / (torch.sqrt(self.v) + self.epsilon)
        # print("before: ")
        # print(self.get_params_as_flat())
        self.set_params_from_flat((self.get_flat_params() + step_update).to(device=torch.device("cpu")))
        # print("after, in mpi adam: ")
        # print(self.get_params_as_flat())

    def sync(self):
        if self.comm is None:
            return
        theta = ptu.get_numpy(self.get_params_as_flat())
        self.comm.Bcast(theta, root=0)
        self.set_params_from_flat(ptu.from_numpy(theta))

    def check_synced(self):
        # If this fails on iteration 0, remember to call SYNC for each optimizer!!!
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0: # this is root
            theta = ptu.get_numpy(self.get_params_as_flat())
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = ptu.get_numpy(self.get_params_as_flat())
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

    def to(self, device=None):
        if device is None:
            device = ptu.device
        self.m = self.m.to(device=device)
        self.v = self.v.to(device=device)

    def get_flat_params(self):
        """
        Get params from a CPU thread
        :return:
        """
        return torch.cat([param.view([U.num_elements(param)]) for param in U.get_flat_params(self.param_groups)], dim=0)