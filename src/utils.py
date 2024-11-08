import datetime
import os
import torch

import torch.nn as nn
import oyaml as yaml

def assert_no_nans(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaNs!"
    assert not torch.isinf(tensor).any(), f"{name} contains Infs!"
    
def assert_normalized(tensor, name):
    assert tensor.min() == 0.0, f"{name}: min is not 0!"
    assert tensor.max() == 1.0, f"{name}: max is not 1! it is {tensor.max()}"

def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path, '--port=24682', '--host=0.0.0.0'])
    url = tb.launch()
    print("[Inpainting Network] Tensorboard session created: "+url)
    webbrowser.open_new(url)

def create_ckpt_dir():
    now = datetime.datetime.today()
    ckpt_dir = "ckpt/{0:%m%d_%H%M_%S}".format(now)
    os.makedirs(ckpt_dir)
    os.makedirs(os.path.join(ckpt_dir, "val_vis"))
    os.makedirs(os.path.join(ckpt_dir, "models"))
    return ckpt_dir

def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()


class Config(dict):
    def __init__(self, conf_file):
        with open(conf_file, "r") as f:
            config = yaml.safe_load(f)
        self._conf = config

    def __getattr__(self, name):
        if self._conf.get(name) is None:
            return None

        return self._conf[name]
    
    def make_copy(self):
        target_path = os.path.join(self.ckpt, 'config_copy.yaml')
        with open(target_path, 'w') as file:
            yaml.safe_dump(self._conf, file)


def conf_to_param(config: dict) -> dict:
    dind_keys = []
    rm_keys = []
    for key, val in config.items():
        if isinstance(val, dict):
            dind_keys.append(key)
        elif not type(val) in [float, int, bool, str]:
            rm_keys.pop(key)

    for target in dind_keys:
        val = config.pop(target)
        config.update(val)
    for target in rm_keys:
        del config[target]

    return config


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device("cpu")
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {"n_iter": n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict["n_iter"]
