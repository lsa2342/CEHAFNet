import torch
import copy
import os
from model.SOLNet import SOLNet
from utils import load_config

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__=="__main__":
    config = load_config('../config/SOLNet.yaml')
    model = SOLNet()
    path_checkpoint = config['checkpoint']
    convert_path = config['model_path']
    if not os.path.exists(convert_path):
        os.makedirs(convert_path)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint)

    repvgg_model_convert(model, os.path.join(convert_path, 'model.pth'))