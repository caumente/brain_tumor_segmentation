import os
import torch
import logging
from pathlib import Path


def count_parameters(model: torch.nn.Module) -> int:
    """
    This function counts the trainable parameters in a model.

    Params:
    *******
        - model (torch.nn.Module): Torch model

    Return:
    *******
        - Int: Number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, checkpoint_path: Path):
    """
    This function saves a model state.

    Params:
    *******
        state (dict): Dict with information such as {epoch:value, optimizer:value, scheduler:value, ..}
        save_folder (Path): Path where the model state is saved

    """

    filename = f'{str(checkpoint_path)}/model_best.pth.tar'
    torch.save(state, filename)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module):
    """
    This function reloads a checkpoint previously generated.

    Params:
    *******
        - checkpoint_path (str): path where checkpoint is located
        - model (torch.nn.Module): model used to compute the segmentation

    """

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"\n\t-> Loaded checkpoint '{checkpoint_path}'. Last epoch: {epoch}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{checkpoint_path}'")


def get_lr(optimizer):
    """
    This function takes a optimizer a input and gets the learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
