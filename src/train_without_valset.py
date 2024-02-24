import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.cuda.set_device('cuda:1')

from src.dataset.BraTS_dataset import dataset_loading
from src.utils.miscellany import AverageMeter, ProgressMeter
from src.utils.miscellany import init_log, save_args, seed_everything
from src.utils.models import init_model_segmentation
from src.utils.models import save_checkpoint, optimizer_loading, loss_function_loading


def load_parameters(filepath=None):
    parser = argparse.ArgumentParser(description='Brats Training')
    arguments = parser.parse_args()

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            arguments.__dict__ = json.load(f)

    return arguments


def main(args):

    # This process can not be carry out without a GPU
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

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

    # If we use the sequences and its inverses sequences the number of sequences is the double
    if args.inverse_seq:
        args.sequences = 2 * args.sequences

    # Implementing the model and turning it from cpu to gpu
    model = init_model_segmentation(architecture=args.architecture, sequences=args.sequences, regions=args.regions,
                                    width=args.width, save_folder=args.save_folder,
                                    deep_supervision=args.deep_supervision)
    model = model.to(device)

    # Implementing loss function and metric
    criterion = loss_function_loading(loss_function=args.loss).to(device)

    # Loading datasets train-val-test and data augmenter
    args.production_training = True
    train_loader = dataset_loading(args)
    logging.info(f"Length training set: {len(train_loader)}")

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

    # Start training phase
    patience = 0
    best = np.inf
    patients_perf = []
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"\n······························ Epoch {epoch} ····································\n")
        ts = time.perf_counter()

        # Training phase
        model.train()
        training_loss = step(train_loader, model, criterion, optimizer, epoch, scaler, patients_perf=patients_perf,
                             device=device, auto_cast_bool=args.auto_cast_bool)
        with open(f"{args.save_folder}/Progress/progressTrain.txt", mode="a") as f:
            print({'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch, 'loss_train': training_loss}, file=f)

        te = time.perf_counter()
        logging.info(f"\nTrain Epoch done in {te - ts:.2f} seconds\n")
        logging.info(f"Training loss: {training_loss}")

        # Saving the model based on the train loss
        if training_loss < best:
            logging.info("Best validation loss improved")
            patience = 0
            best = training_loss

            save_checkpoint(
                dict(
                    epoch=args.epochs,
                    arch=args.architecture,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict()
                ),
                checkpoint_path=args.save_folder)

        scheduler.step(training_loss)

        # Early stopping
        if patience > 10:
            logging.info("\n Early Stopping now! The model hasn't improved in last 10 updates.\n")
            break

    # Ending process
    end_time = time.perf_counter()
    logging.info(f"\nTime spend.. {(end_time - init_time) / 60:.2f} minutes\n")


def step(
        data_loader: torch.utils.data.Dataset,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer,
        epoch: int,
        scaler=None,
        scheduler=None,
        patients_perf=None,
        device=torch.device('cpu'),
        auto_cast_bool=False
):

    #  <------------ SETUP --------------->
    batch_time, data_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Data', ':6.3f'), AverageMeter('Loss',
                                                                                                               ':.4e')

    mode = "train"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        num_batches=batch_per_epoch,
        meters=[batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")
    end = time.perf_counter()
    #  <------------ SETUP --------------->

    for i, batch in enumerate(data_loader):
        data_time.update(time.perf_counter() - end)
        patient_id = batch["patient_id"]
        inputs, ground_truth = batch["sequences"].to(device), batch["ground_truth"].to(device)

        #  <------------ FORWARD PASS --------------->
        with autocast(enabled=auto_cast_bool):
            segmentation = model(inputs)
            # Evaluation
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
        #  <------------ FORWARD PASS --------------->

        #  <------------ BACKWARD PASS --------------->
        if model.training:
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

    return losses.avg


if __name__ == '__main__':
    arguments = load_parameters("arguments_experiment.txt")
    seed_everything(seed=arguments.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main(arguments)
