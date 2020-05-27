import torch.nn as nn
import torch.nn.functional as F
from rlkit.policies.base import ExplorationPolicy
import torch
from rlkit.torch.networks import Mlp
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies import FlattenTanhGaussianPolicy, CompositeNormalizedTanhGaussianPolicy
from rlkit.torch.relational.relational_util import fetch_preprocessing
import numpy as np
import numpy
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rlkit.torch.pytorch_util import shuffle_and_mask
from rlkit.torch.networks import Mlp
from rlkit.torch.relational.modules import *
import rlkit.torch.pytorch_util as ptu


class GraphPropagation(PyTorchModule):
    """
    Input: state
    Output: context vector
    """

    def __init__(self,
                 num_relational_blocks=1,
                 num_query_heads=1,
                 graph_module_kwargs=None,
                 layer_norm=False,
                 activation_fnx=F.leaky_relu,
                 graph_module=AttentiveGraphToGraph,
                 post_residual_activation=True,
                 recurrent_graph=False,
                 **kwargs
                 ):
        """

        :param embedding_dim:
        :param lstm_cell_class:
        :param lstm_num_layers:
        :param graph_module_kwargs:
        :param style: OSIL or relational inductive bias.
        """
        self.save_init_params(locals())
        super().__init__()

        # Instance settings

        self.num_query_heads = num_query_heads
        self.num_relational_blocks = num_relational_blocks
        assert graph_module_kwargs, graph_module_kwargs
        self.embedding_dim = graph_module_kwargs['embedding_dim']

        if recurrent_graph:
            rg = graph_module(**graph_module_kwargs)
            self.graph_module_list = nn.ModuleList(
                [rg for i in range(num_relational_blocks)])
        else:
            self.graph_module_list = nn.ModuleList(
                [graph_module(**graph_module_kwargs) for i in range(num_relational_blocks)])

        # Layer norm takes in N x nB x nE and normalizes
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for i in range(num_relational_blocks)])

        # What's key here is we never use the num_objects in the init,
        # which means we can change it as we like for later.

        """
        ReNN Arguments
        """
        self.layer_norm = layer_norm
        self.activation_fnx = activation_fnx

    def forward(self, vertices, mask=None, **kwargs):
        """

        :param shared_state: state that should be broadcasted along nB dimension. N * (nR + nB * nF)
        :param object_and_goal_state: individual objects
        :return:
        """
        graph = vertices

        for i in range(self.num_relational_blocks):
            graph_module_output = self.graph_module_list[i](graph, mask, **kwargs)

            new_graph = graph_module_output[0]

            if "return_probs" in kwargs.keys() and kwargs["return_probs"]:
                # Get probs as np array
                probs = graph_module_output[1]
                if i == 0:
                    N, nQ, nH, nB = probs.shape
                    stacked_probs = probs.reshape(N, nQ * nH, nB)
                else:
                    # stacked_probs -> N * num_relational_blocks * nQ * nH * nB
                    N, nQ, nH, nB = probs.shape
                    stacked_probs = np.concatenate((stacked_probs, probs.reshape(N, nQ * nH, nB)),
                                                axis=1)


            new_graph = graph + new_graph

            graph = self.activation_fnx(new_graph) # Diff from 7/22
            # Apply layer normalization
            if self.layer_norm:
                graph = self.layer_norms[i](graph)

        outlist = [graph]

        if "return_probs" in kwargs.keys() and kwargs["return_probs"]:
           outlist.append(stacked_probs)
        return outlist


class ValueReNN(PyTorchModule):
    def __init__(self,
                 graph_propagation,
                 readout,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 state_preprocessing_fnx=fetch_preprocessing,
                 *args,
                 value_mlp_kwargs=None,
                 composite_normalizer=None,
                 **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_module = input_module(**input_module_kwargs)
        self.graph_propagation = graph_propagation
        self.readout = readout
        self.composite_normalizer = composite_normalizer

    def forward(self,
                obs,
                mask=None,
                return_stacked_softmax=False):
        vertices = self.input_module(obs, mask=mask)
        new_vertices = self.graph_propagation.forward(vertices, mask=mask)
        pooled_output = self.readout(new_vertices, mask=mask)
        return pooled_output


class QValueReNN(PyTorchModule):
    """
    Used for q-value network
    """

    def __init__(self,
                 graph_propagation,
                 readout,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 state_preprocessing_fnx=fetch_preprocessing,
                 *args,
                 composite_normalizer=None,
                 **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.graph_propagation = graph_propagation
        self.state_preprocessing_fnx = state_preprocessing_fnx
        self.readout = readout
        self.composite_normalizer = composite_normalizer
        self.input_module = input_module(**input_module_kwargs)

    def forward(self, obs, actions, mask=None, return_stacked_softmax=False):
        assert mask is not None
        vertices = self.input_module(obs, actions=actions, mask=mask)
        relational_block_embeddings = self.graph_propagation.forward(vertices, mask=mask)
        pooled_output = self.readout(relational_block_embeddings, mask=mask)
        assert pooled_output.size(-1) == 1
        return pooled_output


class PolicyReNN(PyTorchModule, ExplorationPolicy):
    """
    Used for policy network

    The function output is typically np_ified in eval_np
    """

    def __init__(self,
                 graph_propagation,
                 readout,
                 *args,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 mlp_class=FlattenTanhGaussianPolicy,
                 composite_normalizer=None,
                 batch_size=None,
                 **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.composite_normalizer = composite_normalizer

        # Internal modules
        self.graph_propagation = graph_propagation
        self.selection_attention = readout

        self.mlp = mlp_class(**kwargs['mlp_kwargs'])
        self.input_module = input_module(**input_module_kwargs)

    def forward(self,
                obs,
                mask=None,
                demo_normalizer=False,
                return_probs=False,
                **mlp_kwargs):
        assert mask is not None
        vertices = self.input_module(obs, mask=mask)
        graphpropagation_output = self.graph_propagation.forward(vertices, mask=mask, return_probs=return_probs)

        response_embeddings = graphpropagation_output[0]
        assert len(response_embeddings.shape) == 3

        # import pdb
        # pdb.set_trace()
        selection_out = self.selection_attention(
            vertices=response_embeddings,
            mask=mask,
            return_probs=return_probs
        )

        selected_objects = selection_out[0]
        selected_objects = selected_objects.squeeze(1)

        if return_probs:
            selection_probs = selection_out[1]
            N, nQ, nH, nB = selection_probs.shape

            # Concatenate the graph propagation probabilities with the selection probabilities
            stacked_probs = np.concatenate((graphpropagation_output[1],
                                            selection_probs.reshape(N, nQ * nH, nB)),
                                           axis=1)
            return [self.mlp(selected_objects, **mlp_kwargs),
                    stacked_probs]
        else:
            return [self.mlp(selected_objects, **mlp_kwargs)]

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

        policy_outputs = self.eval_np(obs_np, **kwargs)

        mlp_outputs = policy_outputs[0]

        assert len(mlp_outputs) == 8
        actions = mlp_outputs[0]

        agent_info = dict()

        if "return_probs" in kwargs and kwargs['return_probs']:
            agent_info["attn_probs"] = policy_outputs[1]
        return actions, agent_info