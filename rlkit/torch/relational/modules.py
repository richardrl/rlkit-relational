from torch import nn as nn
from torch.nn import Parameter, functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp
import torch
from rlkit.torch.relational.relational_util import fetch_preprocessing
import rlkit.torch.pytorch_util as ptu
import gtimer as gt


class FetchInputPreprocessing(PyTorchModule):
    """
    Used for the Q-value and value function

    Takes in either obs or (obs, actions) in the forward function and returns the same sized embedding for both

    Make sure actions are being passed in!!
    """
    def __init__(self,
                 normalizer,
                 object_total_dim,
                 embedding_dim,
                 layer_norm=True):
        self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.fc_embed = nn.Linear(object_total_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, obs, actions=None, mask=None):
        vertices = fetch_preprocessing(obs, actions=actions, normalizer=self.normalizer, mask=mask)

        if self.layer_norm is not None:
            return self.layer_norm(self.fc_embed(vertices))
        else:
            return self.fc_embed(vertices)


class Attention(PyTorchModule):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 softmax_temperature=1.0):
        self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))

        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory, mask):
        """
        N, nV, nE memory -> N, nV, nE updated memory

        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = query.size()
        # assert len(query.size()) == 3

        # assert self.fc_createheads.out_features % nE == 0
        nH = int(self.fc_createheads.out_features / nE)

        nV = memory.size(1)

        # assert len(mask.size()) == 2

        # N, nQ, nE -> N, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nQ, nH, nE)
        # else:
        #     query = query.view(N, nQ, nH, nE)

        # if self.layer_norms is not None:
        #     query = self.layer_norms[0](query)
        # N, nQ, nH, nE -> N, nQ, nV, nH, nE
        query = query.unsqueeze(2).expand(-1, -1, nV, -1, -1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)

        # -> N, nQ, nV, nH, 1
        qc_logits = self.fc_logit(torch.tanh(query + context))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)

        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(2).squeeze(2)

        # N, nQ, nH, nE -> N, nQ, nE
        # if nQ > 1:
        attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nH*nE))
        # else:
        #     attention_result = attention_heads.view(N, nQ, nE)

        # attention_result = self.activation_fnx(attention_result)
        #TODO: add nonlinearity here...

        if self.layer_norms is not None:
            attention_result = self.layer_norms[2](attention_result)

        # assert len(attention_result.size()) == 3
        return attention_result


class AttentiveGraphToGraph(PyTorchModule):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 object_total_dim,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True):
        self.save_init_params(locals())
        super().__init__()
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)
        self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None

    def forward(self, vertices, mask):
        """

        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        assert len(vertices.size()) == 3
        N, nV, nE = vertices.size()
        assert mask.size() == torch.Size([N, nV])

        # -> (N, nQ, nE), (N, nV, nE), (N, nV, nE)

        # if self.layer_norm is not None:
        #     qcm_block = self.layer_norm(self.fc_qcm(vertices))
        # else:
        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return self.attention(query, context, memory, mask)


class AttentiveGraphPooling(PyTorchModule):
    """
    Pools nV vertices to a single vertex embedding

    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True):
        self.save_init_params(locals())
        super().__init__()
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        self.num_heads = num_heads
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)

    def forward(self, vertices, mask):
        """
        N, nV, nE -> N, nE
        :param vertices:
        :param mask:
        :return:
        """
        N, nV, nE = vertices.size()

        # nE -> N, nQ, nE where nQ == self.num_heads
        query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, self.num_heads, -1)

        # if self.layer_norm is not None:
        #     cm_block = self.layer_norm(self.fc_cm(vertices))
        # else:
        cm_block = self.fc_cm(vertices)
        context, memory = cm_block.chunk(2, dim=-1)

        gt.stamp("Readout_preattention")
        attention_result = self.attention(query, context, memory, mask)

        gt.stamp("Readout_postattention")
        return attention_result.sum(dim=1) # Squeeze nV dimension so that subsequent projection function does not have a useless 1 dimension