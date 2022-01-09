import os
import sys

import torch
import logging
from pathlib import Path
from typing import List, Tuple
from src.models.DeepUNet import DeepUNet
from src.models.ResidualUNet import resunet_3d
from src.models.ShallowUNet import ShallowUNet
from src.models.Unet3D import UNet3D
from src.models.VNet import VNet
#from ranger import Ranger
from monai.losses import DiceLoss
from src.loss import EDiceLoss


def create_model(
        architecture: str,
        sequences: List[str],
        regions: Tuple[str],
        width: int = 48,
        save_folder: Path = None
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    Params:
    *******
        - architecture: architecture chosen

    Return:
    *******
        - et_present: it is true if the segmentation possess the ET region
        - img_segmentation: stack of images of the regions
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"Sequences for feeding the network: {len(sequences)} ({sequences})")

    if architecture == '3DUNet':
        model = UNet3D(sequences=len(sequences), regions=len(regions))
    elif architecture == 'VNet':
        model = VNet(sequences=len(sequences), regions=len(regions))
    elif architecture == 'ResidualUNet':
        model = resunet_3d(sequences=len(sequences), regions=len(regions), witdh=width)
    elif architecture == 'ShallowUNet':
        model = ShallowUNet(sequences=len(sequences), regions=len(regions), width=width)
    elif architecture == 'DeepUNet':
        model = DeepUNet(sequences=len(sequences), regions=len(regions), width=width)
    else:
        model = torch.nn.Module()
        assert "The model selected does not exist. " \
               "Please, chose some of the following architectures: 3DUNet, VNet, ResidualUNet, ShallowUNet, DeepUNet"

    # Saving the model scheme in a .txt file
    if save_folder is not None:
        model_file = save_folder / "model.txt"
        with model_file.open("w") as f:
            print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


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
        logging.info(f"Loaded checkpoint '{checkpoint_path}'. Last epoch: {epoch}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{checkpoint_path}'")


def get_lr(optimizer):
    """
    This function takes a optimizer a input and gets the learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']




def optimizer_loading(
        model: torch.nn.Module,
        optimizer: str,
        learning_rate: float
) -> torch.optim:

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "ranger":
        optimizer = Ranger(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    return optimizer



def loss_function_loading(
        loss_function: str = "dice",
        n_classes: Tuple[str] = ("et", "tc", "wt"),
        device='cpu'
# ) -> EDiceLoss:
):
    # TODO: implement more loss functions
    if loss_function == 'dice':
        # loss_function = EDiceLoss(classes=n_classes).to(device)
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, smooth_nr=1, smooth_dr=1, squared_pred=True).to(device)
    else:
        print("Dice loss function is the only accepted")
        sys.exit()

    return loss_function_criterion
