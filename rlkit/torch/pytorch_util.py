import torch
import numpy as np
import torch.nn as nn
import functools
from collections import OrderedDict


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = False
device = None
_multigpu = False
_mode = None


def get_mode():
    return _mode


def get_device():
    if _use_gpu:
        if _multigpu:
            return torch.cuda.current_device()
        else:
            return device
    else:
        return device


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    global _mode
    # assert mode in ["gpu_opt", "gpu", "cpu", True, False]
    assert mode in ["gpu_opt", "gpu", False], mode
    _mode = mode
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if (isinstance(mode, str) and "gpu" in mode) else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(device_id=0, device_type="gpu"):
    global device
    if device_type == "gpu":
        torch.cuda.set_device(device_id)
        device = torch.device(F"cuda:{device_id}")
    elif device_type == "cpu":
        device = torch.device(F"cpu:{device_id}")


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(get_device())


def from_numpy(*args, device=None, **kwargs):
    if device is None:
        return torch.from_numpy(*args, **kwargs).float().to(get_device())
    else:
        return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    # print("torch_device")
    # print(torch_device)
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def model_gradient_norm(model):
    total_norm = 0

    num_no_grads = 0

    # if model.parameters():
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        else:
            # print(F"No grad: {name}, Count: {num_no_grads} ")
            num_no_grads += 1
    total_norm = total_norm ** (1. / 2)
    return total_norm


def layerwise_model_gradient_norm(model):
    """
    "": corresponds to the total model gradient norm. Currently not doing anything with it, because a dict can't have "" key

    :param model:
    :return: Dict of each layer of the model and corresponding gradient norm as NUMPY array
    """
    module_gradients_dict = dict()

    for (name, module) in model.named_modules():
        if isinstance(module, nn.ModuleList):
            continue
        module_norm = 0
        for p in module.parameters():
            # print("name" + str(name))
            # print("p" + str(p))
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                module_norm += param_norm.item() ** 2
        module_norm = module_norm ** (1. / 2)
        # if module_norm != 0: # TODO: this ignores the softmax 0 grad
        module_gradients_dict[name] = module_norm
    for (name, parameter) in model.named_parameters():
        if parameter.grad is not None and ".bias" not in name and ".weight" not in name:
            param_norm = parameter.grad.data.norm(2)
            module_gradients_dict[name] = get_numpy(param_norm)
    return module_gradients_dict


def get_activation(name, dic):
    def hook(model, input, output):
        recursive_np_ify_to_dict(output, name, dic)
    return hook


def recursive_np_ify_to_dict(tuple_or_tensor, prefix, dic):
    if isinstance(tuple_or_tensor, torch.Tensor):
        dic[prefix] = get_numpy(tuple_or_tensor)
    elif isinstance(tuple_or_tensor, tuple):
        for el, idx in enumerate(tuple_or_tensor):
            recursive_np_ify_to_dict(el, F"{prefix}.{idx}", dic)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def register_forward_hooks(model, dic):
    for (name, module) in model.named_modules():
        if isinstance(module, nn.ModuleList):
            continue
        filtered_names = ["mlp"]
        if name not in filtered_names:
            module.register_forward_hook(get_activation(name, dic))
    # for (name, parameter) in model.named_parameters():
    #     parameter.register_forward_hook(get_activation(name, dic))


def dfs_freeze(model):
    """
    In place freeze of model weights
    :param model:
    :return:
    """
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_set_attr(attr, type, model, value):
    for name, child in model.named_children():
        if isinstance(child, type):
            setattr(child, attr, value)
        dfs_set_attr(attr, type, child, value)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_freeze(child)


class GradientChangeChecker:
    """
    Checks if params cloned before gradient update loop are changed after the optimizer step.
    Usage: clone params before gradient update
    Run assert param changed after step
    """
    def __init__(self):
        self.before_params = None

    def clone_params(self, network):
        self.before_params = [(name, p.clone()) for (name, p) in network.named_parameters()]

    def all_params_unchanged(self, network):
        """

        :param network:
        :return: Boolean True if ALL parameters were changed
        """
        assert self.before_params, "No params cloned before gradient update loop"
        unchanged_params = []
        for (_, p0), (name, p1) in zip(self.before_params, network.named_parameters()):
            # if p0.grad is not None and p1.grad is not None:
                # print(F"Yes grad: {name}")
            if torch.equal(p0, p1):
                unchanged_params.append(name)
            else:
                # print(F"No grad: {name}")
                pass
        # assert not unchanged_params, unchanged_params
        # return not unchanged_params
        return len(unchanged_params) == len(list(self.before_params))

    def any_param_unchanged(self, network):
        assert self.before_params, "No params cloned before gradient update loop"
        unchanged_params = []
        for (_, p0), (name, p1) in zip(self.before_params, network.named_parameters()):
            # if p0.grad is not None and p1.grad is not None:
            # print(F"Yes grad: {name}")
            if torch.equal(p0, p1):
                unchanged_params.append(name)
            else:
                # print(F"No grad: {name}")
                pass
        # assert not unchanged_params, unchanged_params
        # return not unchanged_params
        return len(unchanged_params) > 0

gcc = GradientChangeChecker()


def convert_lstm_to_lstm_cells(lstm):
    lstm_cells = nn.ModuleList([nn.LSTMCell(lstm.input_size, lstm.hidden_size)] +
                               ([nn.LSTMCell(lstm.hidden_size, lstm.hidden_size)] * (lstm.num_layers - 1)))

    key_names = lstm_cells[0].state_dict().keys()
    source = lstm.state_dict()
    for i in range(lstm.num_layers):
        new_dict = OrderedDict([(k, source["%s_l%d" % (k, i)]) for k in key_names])
        lstm_cells[i].load_state_dict(new_dict)

    return lstm_cells


def convert_lstm_cells_to_lstm(lstm_cells):
    lstm = nn.LSTM(lstm_cells[0].input_size, lstm_cells[0].hidden_size, len(lstm_cells))

    key_names = lstm_cells[0].state_dict().keys()
    lstm_dict = OrderedDict()
    for i, lstm_cell in enumerate(lstm_cells):
        source = lstm_cell.state_dict()
        new_dict = OrderedDict([("%s_l%d" % (k, i), source[k]) for k in key_names])
        lstm_dict = OrderedDict(list(lstm_dict.items()) + list(new_dict.items()))
    lstm.load_state_dict(lstm_dict)

    return lstm


def shuffle_and_mask(tensor, mask=None):
    """
    Shuffles along column dimension (shape 1) for every row (batch_idx)
    :param tensor: 3d tensor
    :return:
    """
    for batch_idx in range(tensor.shape[0]):
        num_blocks = int(mask[batch_idx].sum())
        tensor[batch_idx, :num_blocks, :] = tensor[batch_idx, :num_blocks, :][torch.randperm(num_blocks), :].clone()
        tensor[batch_idx, num_blocks:, :] = torch.zeros_like(tensor[batch_idx, num_blocks:, :])
    return tensor