import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu


def get_flattened_grads(parameter_groups):
    """Flattens a variables and their gradients.
    """
    parameter_list = get_flat_params(parameter_groups)
    grads = [param.grad for param in parameter_list]
    # return torch.cat([grad.view([num_elements(grad)]) if grad is not None else 0 for grad in grads], dim=0)
    return torch.cat([grads[i].view([num_elements(grads[i])]) if grads[i] is not None else ptu.zeros([num_elements(parameter_list[i])]) for i in range(len(grads))], dim=0)


def var_shape(x):
    return x.shape


def num_elements(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def get_flat_params(parameter_groups):
    """

    :param parameter_groups:
    :return: List of parameters pulled out of parameter groups
    """
    parameter_list = []
    for parameter_group in parameter_groups:
        parameter_list += parameter_group['params']

    return parameter_list


class SetFromFlat(object):
    def __init__(self,
                 parameter_groups):
        self.parameter_list = get_flat_params(parameter_groups)
        self.shapes = list(map(var_shape, self.parameter_list))
        self.total_size = np.sum([intprod(shape) for shape in self.shapes])

    def __call__(self, flattened_parameters):
        """

        :param flattened_parameters: type -> Torch Tensor
        :return:
        """
        # before = flattened_parameters.detach().clone()
        # Update worker parameters with flattened_parameters weights broadcasted from root process
        start = 0
        for (shape, param) in zip(self.shapes, self.parameter_list):
            size = intprod(shape)
            param.data.copy_(flattened_parameters[start:start+size].view(shape))
            start += size

        # assert not (before == self.parameter_list).all()
        assert start == self.total_size, (start, self.total_size)


class GetParamsFlat(object):
    def __init__(self,
                 parameter_groups):
        self.parameter_list = get_flat_params(parameter_groups)

    def __call__(self):
        return torch.cat([param.view([num_elements(param)]) for param in self.parameter_list], dim=0)