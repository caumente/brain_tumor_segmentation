import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau  # CosineAnnealingLR
from torch.utils.data import DataLoader
torch.cuda.set_device('cuda:1')

from src.dataset.BraTS_dataset import dataset_loading
from src.loss import EDiceLoss
from src.utils.metrics import save_metrics
from src.utils.miscellany import AverageMeter, ProgressMeter
from src.utils.miscellany import init_log, save_args, seed_everything, generate_segmentations
from src.utils.models import init_model_segmentation
from src.utils.models import save_checkpoint, load_checkpoint, optimizer_loading, loss_function_loading


def load_parameters(filepath=None):
    parser = argparse.ArgumentParser(description='Brats Training')

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            arguments = parser.parse_args()
            arguments.__dict__ = json.load(f)
    else:
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
                            help='Boolean to decide weather removing as much as possible background before cropping')
        parser.add_argument('--inverse_seq', action="store_true",
                            help='Boolean to decide weather using sequences and its inverses images')
        # Loss and metrics settings
        parser.add_argument('--loss', choices=['dice', 'generalized_dice', 'TverskyLoss'], default='dice')
        # Others
        parser.add_argument('--devices', default=1, required=True, type=str,
                            help='Set the CUDA_VISIBLE_DEVICES env var from this string')
        parser.add_argument('--debug_mode', action="store_true")
        parser.add_argument('--val', default=3, type=int,
                            help="how often to perform validation step (default: )")
        parser.add_argument('--com',
                            help="add a comment to this run!")
        parser.add_argument('--postprocessing', default=False, type=bool,
                            help='Postprocessing')

    return arguments


def main(args):

    # This process can not be carry out without a GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        # raise RuntimeWarning("This process can not be carry out without a GPU")
        logging.info("This process may be too slow without a GPU")

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
    args.save_folder = Path(f"./../experiments/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    os.mkdir(args.save_folder / "Progress/")
    args.save_folder = args.save_folder.resolve()
    save_args(args)  # store config as .yaml file

    # init log
    init_time = time.perf_counter()
    init_log(log_name=f"./../experiments/{str(args.exp_name)}/execution.log")
    logging.info(args)

    # If we use the sequences and its inverses sequences the number of sequences is the double
    if args.inverse_seq:
        args.sequences = 2 * args.sequences

    # Implementing the model and turning it from cpu to gpu
    model = init_model_segmentation(architecture=args.architecture, sequences=args.sequences, regions=args.regions,
                                    width=args.width, save_folder=args.save_folder,
                                    deep_supervision=args.deep_supervision)
    model = model.to(device)
    # model = model.to(device) if num_gpus == 1 else torch.nn.DataParallel(model).to(device)

    # Implementing loss function and metric
    criterion = loss_function_loading(loss_function=args.loss).to(device)
    metric = EDiceLoss(classes=args.regions).to(device).metric

    # Loading datasets train-val-test and data augmenter
    train_loader, val_loader, test_loader = dataset_loading(args)

    # optimizer
    optimizer = optimizer_loading(model=model, optimizer=args.optimizer, learning_rate=args.lr, num_epochs=args.epochs,
                                  num_batches_per_epoch=len(train_loader))

    # Custom configuration for a debug run
    if args.debug_mode:
        args.epochs = 1
        args.val = 1

    # Gradient scaler
    logging.info("Using gradient scaling over losses to prevent underflow in backward pass.")
    scaler = GradScaler()
    logging.info("We will also use automatic mixed precision approach in forward pass.")

    # Initializing scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=0)

    # Start training phase
    patience = 0
    best = np.inf
    patients_perf = []
    for epoch in range(args.start_epoch, args.epochs):

        logging.info(f"\n······························ Train Epoch {epoch} ····································\n")
        ts = time.perf_counter()
        mode = "train"
        model.train()
        training_loss = step(data_loader=train_loader,
                             model=model,
                             mode=mode,
                             criterion=criterion,
                             metric=metric,
                             optimizer=optimizer,
                             epoch=epoch,
                             regions=args.regions,
                             scaler=scaler,
                             save_folder=args.save_folder,
                             patients_perf=patients_perf,
                             device=device,
                             auto_cast_bool=args.auto_cast_bool
                             )
        with open(f"{args.save_folder}/Progress/progressTrain.txt", mode="a") as f:
            print({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch, 'loss_train': training_loss}, file=f)

        te = time.perf_counter()
        logging.info(f"\nTrain Epoch done in {te - ts:.2f} seconds")
        logging.info(f"Training loss: {training_loss:.4f}")
        logging.info(f"\n······························ Train Epoch {epoch} ····································\n")

        if (epoch + 1) % args.val == 0:
            logging.info(f"\n······························ Val Epoch {epoch} ····································\n")
            ts = time.perf_counter()
            model.eval()
            mode = "val"
            with torch.no_grad():
                validation_loss = step(val_loader, model, mode, criterion, metric, optimizer, epoch,
                                       args.regions, save_folder=args.save_folder,
                                       patients_perf=patients_perf, device=device, auto_cast_bool=args.auto_cast_bool)
                with open(f"{args.save_folder}/Progress/progressVal.txt", mode="a") as f:
                    print({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch, 'loss_val': validation_loss},
                          file=f)

            if validation_loss < best:
                logging.info("\nBest validation loss improved")
                patience, best = 0, validation_loss

                save_checkpoint(
                    dict(
                        epoch=epoch,
                        arch=args.architecture,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                    ),
                    checkpoint_path=args.save_folder)
            else:
                patience += 1
                logging.info(f"\nBest validation loss did not improve for {patience} epochs")
            scheduler.step(validation_loss)

            te = time.perf_counter()
            logging.info(f"Val epoch done in {te - ts:.2f} seconds")
            logging.info(f"Validation loss: {validation_loss:.4f}")
            logging.info(f"\n······························ Val Epoch {epoch} ····································\n")

        # Early stopping
        if patience >= args.max_patience:
            logging.info(f"\n Early Stopping now! The model hasn't improved in last {args.max_patience} updates.\n")
            break

    if args.debug_mode:
        save_checkpoint(
            dict(
                epoch=args.epochs,
                arch=args.architecture,
                state_dict=model.state_dict(),
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

        load_checkpoint(f'{str(args.save_folder)}/model_best.pth.tar', model)
        generate_segmentations(test_loader, model, args, device=device)
    except KeyboardInterrupt:
        logging.info("Stopping right now!")

    # Ending process
    end_time = time.perf_counter()
    logging.info(f"\nTime spend.. {(end_time - init_time) / 60:.2f} minutes\n")


def step(
        data_loader: torch.utils.data.Dataset,
        model: torch.nn.Module,
        mode: str,
        criterion: torch.nn.Module,
        metric,
        optimizer,
        epoch: int,
        regions: Tuple[str],
        scaler=None,
        scheduler=None,
        save_folder=None,
        patients_perf=None,
        device=torch.device('cpu'),
        auto_cast_bool=False
):

    #  <------------ SETUP --------------->
    batch_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        num_batches=len(data_loader),
        meters=[batch_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")
    end = time.perf_counter()
    #  <------------ SETUP --------------->

    metrics = []
    for i, batch in enumerate(data_loader):
        patient_id = batch["patient_id"]
        inputs = batch["sequences"].to(device)
        ground_truth = batch["ground_truth"].to(device)

        #  <------------ FORWARD PASS --------------->
        with autocast(enabled=auto_cast_bool):

            segmentation = model(inputs)

            # Evaluation depending on whether deep supervision is implemented
            if type(segmentation) == list:
                loss = torch.sum(torch.stack([criterion(s, ground_truth) for s in segmentation]))
            else:
                loss = criterion(segmentation, ground_truth)
            patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss.item()))

            # Checking not nan value
            if not np.isnan(loss.item()):
                losses.update(loss.item())
            else:
                logging.info("NaN in model loss!!")

            if mode == "val":
                if type(segmentation) == list:
                    for s in segmentation:
                        metrics.extend(metric(s, ground_truth))
                else:
                    metrics.extend(metric(segmentation, ground_truth))
        #  <------------ FORWARD PASS --------------->

        #  <------------ BACKWARD PASS --------------->
        if mode == "train":
            scaler.scale(loss).backward()
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

    if mode == "val":
        save_metrics(metrics=metrics, current_epoch=epoch, loss=losses.avg, regions=regions, save_folder=save_folder)

    return losses.avg


if __name__ == '__main__':
    arguments = load_parameters("arguments_experiment.txt")
    seed_everything(seed=arguments.seed)
    main(arguments)
