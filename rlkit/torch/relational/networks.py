import torch.nn as nn
import torch.nn.functional as F
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import PyTorchModule
from rlkit.torch.relational.modules import AttentiveGraphToGraph, FetchInputPreprocessing, AttentiveGraphPooling
from rlkit.torch.sac.policies import FlattenTanhGaussianPolicy
import numpy as np
import gtimer as gt


class ReNN(PyTorchModule):
    def __init__(self,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 graph_module_class=AttentiveGraphToGraph,
                 graph_module_kwargs=None,
                 readout_module_class=AttentiveGraphPooling,
                 readout_module_kwargs=None,
                 proj_class=FlattenTanhGaussianPolicy,
                 proj_kwargs=None,
                 activation_fnx=F.leaky_relu,
                 layer_norm=True,
                 num_graph_modules=1
                 ):
        self.save_init_params(locals())
        super().__init__()
        self.input_module = input_module(**input_module_kwargs)
        self.graph_modules = nn.ModuleList([graph_module_class(**graph_module_kwargs) for i in range(num_graph_modules)])
        self.readout_module = readout_module_class(**readout_module_kwargs)
        self.proj = proj_class(**proj_kwargs)
        self.activation_fnx = activation_fnx

    def forward(self,
                obs,
                actions=None,
                mask=None,
                **proj_kwargs):
        vertices = self.input_module(obs, actions=actions, mask=mask)

        assert len(vertices.size()) == 3

        gt.stamp("Forward_input_module")
        for i in range(len(self.graph_modules)):
            graph_output = self.graph_modules[i](vertices, mask)
            vertices = self.activation_fnx(graph_output) + vertices # Residual connection
            vertices = self.activation_fnx(vertices)

        gt.stamp("Forward_graph_modules")

        pooled_embedding = self.readout_module(vertices, mask)

        gt.stamp("Forward_readout_module")
        return self.proj(pooled_embedding, **proj_kwargs)


class ReNNPolicy(ReNN, ExplorationPolicy):
    def __init__(self, **kwargs):
        self.save_init_params(locals())
        super().__init__(**kwargs)

    def get_action(self,
                   obs_np,
                   **kwargs):
        assert len(obs_np.shape) == 1
        actions, agent_info = self.get_actions(obs_np[None], **kwargs)
        assert isinstance(actions, np.ndarray)
        return actions[0, :], agent_info

    def get_actions(self,
                    obs_np,
                    **kwargs):
        mlp_outputs = self.eval_np(obs_np, **kwargs)
        actions = mlp_outputs[0]

        assert len(actions.shape) == 2
        assert isinstance(actions, np.ndarray)

        agent_info = dict()
        return actions, agent_info