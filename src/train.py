import argparse
import logging
import os
import pathlib
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ranger import Ranger
from src.dataset.DataAugmenter import DataAugmenter
from src.dataset.brats import get_datasets
from src.loss import EDiceLoss
from src.models.DeepUNet import DeepUNet
from src.models.ResidualUNet import resunet_3d
from src.models.ShallowUNet import ShallowUNet
from src.models.Unet3D import UNet3D
from src.models.VNet import get_Vnet
from src.utils.metrics import save_metrics
from src.utils.miscellany import AverageMeter
from src.utils.miscellany import ProgressMeter
from src.utils.miscellany import generate_segmentations
from src.utils.miscellany import init_log
from src.utils.miscellany import save_args
from src.utils.models import save_checkpoint, load_checkpoint, count_parameters

parser = argparse.ArgumentParser(description='Brats Training')

# Architecture parameters
parser.add_argument('--architecture', default='DeepUNet',
                    choices=['3DUNet', 'VNet', 'ResidualUNet', 'ShallowUNet', 'DeepUNet'],
                    help='Model architecture (default: DeepUNet)')
parser.add_argument('--width', default=48, type=int, choices=[12, 24, 48],
                    help='Width of first convolutional kernel. Doubled at each U-Level (default: 48)')
parser.add_argument('--sequences', nargs="+", default=["_t1", "_t1ce", "_t2", "_flair"],
                    help='Sequences used for feeding the model (default: --sequences _t1 _t1ce _t2 _flair)')
parser.add_argument('--regions', nargs="+", default=["et", "tc", "wt"],
                    help='Subregions to assess (default: --regions et tc wt)')
# Hyperparameters
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Start epoch number (default: 0). Useful on restarts.)')
parser.add_argument('--epochs', default=400, type=int,
                    help='Number of total epochs (default: 400)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, dest='lr',
                    help='Initial learning rate (default: 1e-4)')
parser.add_argument('--optimizer', choices=['adam', 'sgd', 'ranger', 'adamw'], default='ranger',
                    help="Optimizer (default: Ranger)")
parser.add_argument('--workers', default=12, type=int,
                    help='Number of data loading workers (default: 12)')
# Dataset settings
parser.add_argument('--seed', default=9588,
                    help="Seed used to split the dataset in train-val-test sets")
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size (default: 1)')
parser.add_argument('--normalization', action="store_true",
                    help='Boolean to decide weather the images are normalized or not')
parser.add_argument('--low_norm', default=1, type=int,
                    help='Lower percentile to clip the images in the normalization process')
parser.add_argument('--high_norm', default=99, type=int,
                    help='Upper percentile to clip the images in the normalization process')
parser.add_argument('--crop_or_pad', nargs="+", default=[155, 240, 240],
                    help='Resolution of the images after random crops')
parser.add_argument('--fit_boundaries', action="store_true",
                    help='Boolean to decide weather remove as much as possible background or not before cropping')
parser.add_argument('--inverse_seq', action="store_true",
                    help='Boolean to decide weather using sequences and its inverses images')
# Loss and metrics settings
parser.add_argument('--loss', choices=['dice', 'generalized_dice', 'TverskyLoss'], default='dice')
# Others
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--debug_mode', action="store_true")
parser.add_argument('--val', default=3, type=int,
                    help="how often to perform validation step (default: )")
parser.add_argument('--com',
                    help="add a comment to this run!")
parser.add_argument('--postprocessing', default=False, type=bool,
                    help='Postprocessing')


def main(args):

    # This process can not be carry out without a GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeWarning("This process can not be carry out without a GPU")

    # Converting crop_or_pad input to tuple
    args.crop_or_pad = tuple([int(i) for i in args.crop_or_pad]) if len(args.crop_or_pad) == 3 else None

    # Setting up experiment name and folders
    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{args.com.replace(' ', '_') if args.com else ''}" \
                    f"{'_debug_' if args.debug_mode else ''}{current_experiment_time}_" \
                    f"{''.join(args.sequences) if len(args.sequences) < 4 else ''}" \
                    f"_{args.architecture}" \
                    f"_{args.width}" \
                    f"_batch{args.batch_size}" \
                    f"_{args.optimizer}" \
                    f"_lr{args.lr}" \
                    f"_epochs{args.epochs}"
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    os.mkdir(args.save_folder / "Progress/")
    args.save_folder = args.save_folder.resolve()
    save_args(args)  # store config as .yaml file

    # init log
    init_time = time.perf_counter()
    init_log(log_name=f"./runs/{str(args.exp_name)}/execution.log")

    # If we use thw sequences and its inverses sequences the number of sequences is de double
    if args.inverse_seq:
        args.sequences = 2 * args.sequences

    # Implementing the model and turning it from cpu to gpu
    my_model = model_loading(architecture=args.architecture, sequences=args.sequences, regions=args.regions,
                             width=args.width, save_folder=args.save_folder)
    my_model = my_model.cuda() if num_gpus == 1 else torch.nn.DataParallel(my_model).cuda()

    # Implementing loss function and metric
    criterion = loss_function_loading(loss_function=args.loss)
    metric = EDiceLoss(classes=args.regions).cuda().metric

    # optimizer
    rangered, optimizer = optimizer_loading(model=my_model, optimizer=args.optimizer, learning_rate=args.lr)

    # Custom configuration for a debug run
    if args.debug_mode:
        args.epochs = 4
        args.val = 1

    # Loading datasets train-val-test
    train_loader, val_loader, test_loader = dataset_loading(args)
    for n, batch in enumerate(train_loader):
        if n>0:
            break
        print(batch['patient_id'])
        print(batch['sequences'].shape)
        print(np.max(batch['sequences'].cpu().numpy()))
        print(np.mean(batch['sequences'].cpu().numpy()))
        print(batch['ground_truth'].shape)

    # Gradient scaler
    logging.info("Using gradient scaling over losses to prevent underflow in backward pass.")
    scaler = GradScaler()
    logging.info("We will also use automatic mixed precision approach in forward pass.")

    # Initializing scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    # Start training phase
    epochs_not_improve = 0
    best = np.inf
    patients_perf = []
    for epoch in range(args.start_epoch, args.epochs):
        try:
            logging.info(f"\n······························ Epoch {epoch} ····································\n")
            ts = time.perf_counter()

            # Training phase
            my_model.train()
            training_loss = step(train_loader, my_model, criterion, metric, optimizer, epoch,
                                 args.regions, scaler, save_folder=args.save_folder,
                                 patients_perf=patients_perf)
            with open(f"{args.save_folder}/Progress/progressTrain.txt", mode="a") as f:
                print({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch, 'loss_train': training_loss}, file=f)

            te = time.perf_counter()
            logging.info(f"\nTrain Epoch done in {te - ts:.2f} seconds\n")
            logging.info(f"Training loss: {training_loss}")

            # Validation phase
            if (epoch + 1) % args.val == 0:
                my_model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, my_model, criterion, metric, optimizer, epoch,
                                           args.regions, save_folder=args.save_folder,
                                           patients_perf=patients_perf)
                    with open(f"{args.save_folder}/Progress/progressVal.txt", mode="a") as f:
                        print({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch, 'loss_val': validation_loss},
                              file=f)

                if validation_loss < best:
                    epochs_not_improve = 0
                    best = validation_loss

                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            arch=args.architecture,
                            state_dict=my_model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        checkpoint_path=args.save_folder)
                else:
                    epochs_not_improve += 1

                ts = time.perf_counter()
                logging.info(f"\nVal epoch done in {ts - te:.2f} seconds\n")
                logging.info(f"Validation loss: {validation_loss}")

                # Validation
                scheduler.step(validation_loss)

            # Early stopping
            if (epoch / args.epochs > 0.25) and (epochs_not_improve > 30):
                logging.info("\n Early Stopping now! The model hasn't improved in last 15 updates.\n")
                break

        except KeyboardInterrupt:
            logging.info("Stopping training loop, doing benchmark")
            break

    if args.debug_mode:
        save_checkpoint(
            dict(
                epoch=args.epochs,
                arch=args.architecture,
                state_dict=my_model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            checkpoint_path=args.save_folder)

    try:
        df_individual_perf = pd.DataFrame.from_records(patients_perf)
        df_individual_perf.to_csv(path_or_buf=Path(f'{str(args.save_folder)}/patients_indiv_perf.csv'))
        logging.info(df_individual_perf)

        logging.info("\n**********************************************************")
        logging.info("********** START VALIDATION OVER TEST DATASET ************")
        logging.info("**********************************************************\n")

        load_checkpoint(f'{str(args.save_folder)}/model_best.pth.tar', my_model)
        generate_segmentations(test_loader, my_model, args)
    except KeyboardInterrupt:
        logging.info("Stopping right now!")

    # Ending process
    end_time = time.perf_counter()
    logging.info(f"\nTime spend.. {(end_time - init_time) / 60:.2f} minutes\n")


# FUNCTIONS
def model_loading(
        architecture: str,
        sequences: List[str],
        regions: Tuple[str],
        width: int = 48,
        save_folder: Path = ""
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
        model = get_Vnet(sequences=len(sequences), regions=len(regions))
    elif architecture == 'ResidualUNet':
        model = resunet_3d(sequences=len(sequences), regions=len(regions), witdh=width)
    elif architecture == 'ShallowUNet':
        model = ShallowUNet(sequences=len(sequences), regions=len(regions), width=width)
    elif architecture == 'DeepUNet':
        model = DeepUNet(sequences=len(sequences), regions=len(regions), width=width)
    else:
        model = torch.nn.Module()
        assert "The model selected does not exist. " \
               "Please chose between: 3DUNet, VNet, ResidualUNet, ShallowUNet, DeepUNet"

    # Saving the model scheme in a .txt file
    model_file = save_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


def optimizer_loading(
        model: torch.nn.Module,
        optimizer: str,
        learning_rate: float
) -> torch.optim:
    rangered = False  # needed because LR scheduling scheme is different for this optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "ranger":
        optimizer = Ranger(model.parameters(), lr=learning_rate)
        rangered = True
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    return rangered, optimizer


def loss_function_loading(
        loss_function: str = "dice",
        n_classes: Tuple[str] = ("et", "tc", "wt")
) -> EDiceLoss:
    if loss_function == 'dice':
        criterion = EDiceLoss(classes=n_classes).cuda()
    # TODO: implement more loss functions
    # elif loss_function == 'generalized_dice':
    #    criterion = GeneralizedDiceLoss(include_background=False, sigmoid=True, batch=False)
    else:
        criterion = EDiceLoss(classes=n_classes).cuda()

    return criterion


def dataset_loading(args):
    train_dataset, val_dataset, test_dataset = get_datasets(sequences=args.sequences, regions=args.regions,
                                                            seed=args.seed, debug_mode=args.debug_mode,
                                                            has_ground_truth=True, normalization=args.normalization,
                                                            low_norm_percentile=args.low_norm,
                                                            high_norm_percentile=args.high_norm,
                                                            crop_or_pad=args.crop_or_pad,
                                                            fit_boundaries=args.fit_boundaries,
                                                            inverse_seq=args.inverse_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=False,
                                             num_workers=args.workers, )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader, test_loader


def step(
        data_loader: torch.utils.data.Dataset,
        model: torch.nn.Module,
        criterion: EDiceLoss,
        metric,
        optimizer,
        epoch: int,
        regions: Tuple[str],
        scaler=None,
        scheduler=None,
        save_folder=None,
        patients_perf=None
):

    #  <------------ SETUP --------------->
    batch_time, data_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Data', ':6.3f'), AverageMeter('Loss',
                                                                                                               ':.4e')

    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        num_batches=batch_per_epoch,
        meters=[batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")
    end = time.perf_counter()
    #  <------------ SETUP --------------->

    metrics = []
    for i, batch in enumerate(data_loader):
        data_time.update(time.perf_counter() - end)
        patient_id, inputs, ground_truth = batch["patient_id"], batch["sequences"].cuda(), batch["ground_truth"].cuda()

        #  <------------ FORWARD PASS --------------->
        with autocast(enabled=True):

            # If train dada augmentation, else just prediction
            if mode == "train":
                data_aug = DataAugmenter(probability=0.4).cuda()
                inputs = data_aug(inputs)
                segmentation = model(inputs)
                segmentation = data_aug.reverse(segmentation)
            else:
                segmentation = model(inputs)

            # Evaluation
            error_loss = criterion(segmentation, ground_truth)
            patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=error_loss.item()))

            # Checking not nan value
            if not np.isnan(error_loss.item()):
                losses.update(error_loss.item())
            else:
                logging.info("NaN in model loss!!")

            if not model.training:
                metric_ = metric(segmentation, ground_truth)
                metrics.extend(metric_)
        #  <------------ FORWARD PASS --------------->

        #  <------------ BACKWARD PASS --------------->
        if model.training:
            scaler.scale(error_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        #  <------------ BACKWARD PASS --------------->

        #  Displaying execution time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        progress.display(i)

    if not model.training:
        save_metrics(metrics=metrics, current_epoch=epoch, regions=regions, save_folder=save_folder)

    return losses.avg


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
