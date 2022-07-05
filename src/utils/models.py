import os
import sys

import torch
import logging
from pathlib import Path
from typing import List, Tuple
from src.models.segmentator.DeepUNet import DeepUNet
from src.models.segmentator.ResidualUNet import resunet_3d
from src.models.segmentator.ShallowUNet import ShallowUNet
from src.models.segmentator.ShallowUNet_v3 import ShallowUNet_v3
from src.models.segmentator.DoubleShallowUNet import DoubleShallowUNet
from src.models.segmentator.ShallowUNet_noPoolings import ShallowUNetNoPoolings
from src.models.segmentator.MultiImageInput_ShallowUNet import MultiImageInput_ShallowUNet
from src.models.segmentator.ShallowUNetSecondStage import ShallowUNetSecondStage
from src.models.segmentator.MultiInputSkippedShallowUNet import MultiInputSkippedShallowUNet
from src.models.segmentator.Unet3D import UNet3D
from src.models.segmentator.VNet import VNet
from src.models.segmentator.UltraDeepUNet import UltraDeepUNet
from src.models.segmentator.ResidualShallowUNet import ResidualShallowUNet
from src.models.segmentator.AttentionShallowUNet import AttentionShallowUNet
from src.models.classifiers.ShallowUNetClassifier import ShallowUNetClassifier
# from src.models.segmentator.MISU import MISU
from ranger import Ranger
from ranger21 import Ranger21
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss
from src.loss.RegionBasedDice import RegionBasedDiceLoss
from src.loss.ExperimentalDICE import ExperimentalDICE
from monai.networks.nets import UNETR
from src.models.segmentator.nnUNet2021 import nnUNet2021


def init_model_segmentation(
        architecture: str,
        sequences: List[str],
        regions: Tuple[str],
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
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
        model = ShallowUNet(sequences=len(sequences), regions=len(regions), width=width,
                            deep_supervision=deep_supervision)
    elif architecture == 'ShallowUNet_v3':
        model = ShallowUNet_v3(sequences=len(sequences), regions=len(regions), width=width,
                               deep_supervision=deep_supervision)
    elif architecture == 'DoubleShallowUNet':
        model = DoubleShallowUNet(sequences=len(sequences), regions=len(regions), width=width,
                                  deep_supervision=deep_supervision)
    elif architecture == 'ShallowUNetNoPoolings':
        model = ShallowUNetNoPoolings(sequences=len(sequences), regions=len(regions), width=width,
                                      deep_supervision=deep_supervision)
    elif architecture == 'ShallowUNetTesting':
        model = MultiImageInput_ShallowUNet(sequences=len(sequences), regions=len(regions), width=width,
                                            deep_supervision=deep_supervision)
    elif architecture == 'ShallowUNetSecondStage':
        model = ShallowUNetSecondStage(sequences=len(sequences), regions=len(regions), width=width,
                                       deep_supervision=deep_supervision)
    elif architecture == 'MultiShallowUNet':
        model = MultiInputSkippedShallowUNet(sequences=len(sequences), regions=len(regions), width=width,
                                             deep_supervision=deep_supervision)
    elif architecture == 'DeepUNet':
        model = DeepUNet(sequences=len(sequences), regions=len(regions), width=width, deep_supervision=deep_supervision)
    elif architecture == 'UltraDeepUNet':
        model = UltraDeepUNet(sequences=len(sequences), regions=len(regions), width=width,
                              deep_supervision=deep_supervision)
    elif architecture == 'AttentionShallowUNet':
        model = AttentionShallowUNet(sequences=len(sequences), regions=len(regions), width=width)
    elif architecture == 'ResidualShallowUNet':
        model = ResidualShallowUNet(sequences=len(sequences), regions=len(regions), width=width)
    elif architecture == 'UNETR':
        model = UNETR(in_channels=len(sequences), out_channels=len(regions), img_size=[160, 224, 160])
    elif architecture == 'nnUNet2021':
        model = nnUNet2021(sequences=len(sequences), regions=len(regions))
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


def init_model_classification(
        architecture: str,
        sequences: List[str],
        classes: int,
        dense_neurons: int = 1280,
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

    if architecture == 'ShallowUNetClassifier':
        model = ShallowUNetClassifier(sequences=len(sequences), classes=classes, width=width,
                                      dense_neurons=dense_neurons)
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
        learning_rate: float,
        num_epochs: int,
        num_batches_per_epoch: int
) -> torch.optim:
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "ranger":
        optimizer = Ranger(model.parameters(), lr=learning_rate)
    elif optimizer == "ranger21":
        optimizer = Ranger21(model.parameters(), lr=learning_rate, num_epochs=num_epochs,
                             num_batches_per_epoch=num_batches_per_epoch, warmdown_min_lr=1e-6)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    return optimizer


def loss_function_loading(loss_function: str = "dice") -> torch.nn.Module:
    if loss_function == 'dice':
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1,
                                           squared_pred=True)
    elif loss_function == "dice_focal":
        loss_function_criterion = DiceFocalLoss(include_background=True, sigmoid=True, squared_pred=True)
    elif loss_function == "generalized_dice":
        loss_function_criterion = GeneralizedDiceLoss(include_background=True, sigmoid=True)
    elif loss_function == "dice_crossentropy":
        loss_function_criterion = DiceCELoss(include_background=True, sigmoid=True, squared_pred=True)
    elif loss_function == "region_based_dice":
        loss_function_criterion = RegionBasedDiceLoss(weight_region=[2, 0.7, 0.3], include_background=True,
                                                      sigmoid=True)
    elif loss_function == "experimental_dice":
        loss_function_criterion = ExperimentalDICE(include_background=True, sigmoid=True, reduction="sum")
    elif loss_function == "jaccard":
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, jaccard=True, reduction="sum")
    else:
        print("Select a loss function allowed: ['dice', 'dice_focal', 'generalized_dice', 'dice_crossentropy', "
              "'region_based_dice']")
        sys.exit()

    return loss_function_criterion
