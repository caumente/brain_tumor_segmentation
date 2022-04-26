from pathlib import Path
from typing import List
import logging as log
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from src.dataset.BraTS_Dataloader import Brats


def get_datasets(
        sequences,
        regions,
        seed,
        debug_mode,
        has_ground_truth=True,
        path_images="",
        normalization=True,
        clipping=True,
        low_norm_percentile=1,
        high_norm_percentile=99,
        crop_or_pad=(155, 240, 240),
        fit_boundaries=True,
        histogram_equalization=True,
        inverse_seq=False,
        auto_cast_bool=True,
        oversampling=None,
        production=False
):

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    patients_dir = sorted([x for x in path_images.iterdir() if x.is_dir()])
    log.info(f"Images are contained in the following path: {path_images}")

    # Splitting the path into train_path, val_path and test_path
    if "BRATS2021" in str(path_images):
        mapping = pd.DataFrame(patients_dir, columns=["patients"])
        train_path, val_path, test_path = train_test_val_split_BraTS_2021(mapping=mapping,
                                                                          patients_path=patients_dir,
                                                                          seed=seed,
                                                                          train_size=0.8)
    else:
        mapping = pd.read_csv(f"{path_images}/name_mapping.csv")
        train_path, val_path, test_path = train_test_val_split_BraTS_2020(mapping=mapping,
                                                                          patients_path=patients_dir,
                                                                          seed=seed,
                                                                          train_size=0.8,
                                                                          oversampling=oversampling,
                                                                          production=production
                                                                          )

    # Creating the train-validation-test datasets
    train_dataset = Brats(patients_path=train_path,
                          sequences=sequences,
                          has_ground_truth=has_ground_truth,
                          regions=regions,
                          normalization=normalization,
                          clipping=clipping,
                          low_percentile=low_norm_percentile,
                          high_percentile=high_norm_percentile,
                          crop_or_pad=crop_or_pad,
                          fit_boundaries=fit_boundaries,
                          histogram_equalization=histogram_equalization,
                          inverse_seq=inverse_seq,
                          debug_mode=debug_mode,
                          auto_cast_bool=auto_cast_bool,
                          data_augmentation=True)
    val_dataset = Brats(patients_path=val_path,
                        sequences=sequences,
                        has_ground_truth=has_ground_truth,
                        regions=regions,
                        normalization=normalization,
                        clipping=clipping,
                        low_percentile=low_norm_percentile,
                        high_percentile=high_norm_percentile,
                        crop_or_pad=crop_or_pad,
                        fit_boundaries=fit_boundaries,
                        histogram_equalization=histogram_equalization,
                        inverse_seq=inverse_seq,
                        debug_mode=debug_mode,
                        auto_cast_bool=auto_cast_bool)
    test_dataset = Brats(patients_path=test_path,
                         sequences=sequences,
                         has_ground_truth=has_ground_truth,
                         regions=regions,
                         normalization=normalization,
                         clipping=clipping,
                         low_percentile=low_norm_percentile,
                         high_percentile=high_norm_percentile,
                         crop_or_pad=crop_or_pad,
                         fit_boundaries=fit_boundaries,
                         histogram_equalization=histogram_equalization,
                         inverse_seq=inverse_seq,
                         debug_mode=debug_mode,
                         auto_cast_bool=auto_cast_bool)

    log.info(f"Size of train dataset: {len(train_dataset)}")
    log.info(f"Shape of images used for training: {train_dataset[0]['sequences'].shape}")
    log.info(f"Size of validation dataset: {len(val_dataset)}")
    log.info(f"Shape of images used for validating: {val_dataset[0]['sequences'].shape}")
    log.info(f"Size of test dataset: {len(test_dataset)}")
    log.info(f"Shape of images used for testing: {test_dataset[0]['sequences'].shape}")

    return train_dataset, val_dataset, test_dataset


def train_test_val_split_BraTS_2020(
        mapping: pd.DataFrame,
        patients_path: List[Path],
        seed: int,
        train_size: float = 0.8,
        oversampling: str = None,
        production: bool = False
):

    """ This function splits the dataset into train-val-test based on a train_size."""

    mapping[['-', '--', 'name']] = mapping['BraTS_2020_subject_ID'].str.split("_", expand=True)
    mapping = mapping[['Grade', 'BraTS_2020_subject_ID', 'name']]

    train, val_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True, stratify=mapping['Grade'])
    val, test = train_test_split(val_, test_size=0.5, random_state=int(seed), shuffle=True, stratify=val_['Grade'])

    if not production:
        if oversampling is not None:
            train = oversampled_dataset(train, strategy=oversampling)
            log.info(f"Training set oversampled.")

        log.info(f"{len(train)} patients used for training phase")
        log.info(f"{len(val)} patients used for validation phase")
        log.info(f"{len(test)} patients used for testing phase")

    if production:
        if oversampling is not None:
            train = oversampled_dataset(train, strategy=oversampling)
            val = oversampled_dataset(val, strategy=oversampling)
            test = oversampled_dataset(test, strategy=oversampling)
            log.info(f"Dataset oversampled.")

    train_idx, val_idx, test_idx = train.index, val.index, test.index

    train = [patients_path[i] for i in train_idx]
    val = [patients_path[i] for i in val_idx]
    test = [patients_path[i] for i in test_idx]

    return train, val, test


def train_test_val_split_BraTS_2021(mapping, patients_path, seed, train_size=0.8):
    """ This function splits the dataset into train-val-test based on a train_size."""

    train, val_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True)
    val, test = train_test_split(val_, test_size=0.5, random_state=int(seed), shuffle=True)
    train_idx, val_idx, test_idx = train.index, val.index, test.index
    log.info(f"Patients used for training:\n {train}")
    log.info(f"Patients used for validating:\n {val}")
    log.info(f"Patients used for testing:\n {test}")

    train = [patients_path[i] for i in train_idx]
    val = [patients_path[i] for i in val_idx]
    test = [patients_path[i] for i in test_idx]

    return train, val, test


def dataset_loading(args):
    train_dataset, val_dataset, test_dataset = get_datasets(sequences=args.sequences,
                                                            regions=args.regions,
                                                            seed=args.seed,
                                                            debug_mode=args.debug_mode,
                                                            path_images=args.path_dataset,
                                                            has_ground_truth=True,
                                                            normalization=args.normalization,
                                                            clipping=args.clipping,
                                                            low_norm_percentile=args.low_norm,
                                                            high_norm_percentile=args.high_norm,
                                                            crop_or_pad=args.crop_or_pad,
                                                            fit_boundaries=args.fit_boundaries,
                                                            histogram_equalization=args.histogram_equalization,
                                                            inverse_seq=args.inverse_seq,
                                                            auto_cast_bool=args.auto_cast_bool,
                                                            oversampling=args.oversampling,
                                                            production=args.production_training)
    if args.production_training:
        return DataLoader(ConcatDataset([train_dataset, val_dataset, test_dataset]), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True,
                          persistent_workers=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=False, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True,
                            num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader, test_loader


def oversampled_dataset(df, strategy='LGG'):
    """
    This function oversamples the dataset copying the LGG patients, those that have few pixels in the ET region, or
    taking both.
    """
    if strategy == 'LGG':
        LGGS = df[df.Grade == "LGG"]
        df = pd.concat([df, LGGS, LGGS])
    elif strategy == 'few_ET':
        few_pixels = df[df.name.isin(['086', '087', '141', '262', '263', '264', '265', '266', '268', '269', '271',
                                      '272', '275', '278', '279', '280', '281', '286', '289', '291', '294', '295',
                                      '297', '298', '299', '304', '305', '306', '307', '310', '311', '312', '313',
                                      '315', '319', '321', '322', '324', '325', '329', '330', '335', '361'])]
        df = pd.concat([df, few_pixels, few_pixels])
    elif strategy == "both":
        LGGS = df[df.Grade == "LGG"]
        few_pixels = df[df.name.isin(['086', '087', '141', '262', '263', '264', '265', '266', '268', '269', '271',
                                      '272', '275', '278', '279', '280', '281', '286', '289', '291', '294', '295',
                                      '297', '298', '299', '304', '305', '306', '307', '310', '311', '312', '313',
                                      '315', '319', '321', '322', '324', '325', '329', '330', '335', '361'])]
        df = pd.concat([df, LGGS, LGGS, few_pixels, few_pixels])

    return df


def recover_initial_resolution(image, cropped_indexes, random_indexes):
    """
    This function recover the initial image resolution given the cropped index and the random pads added.

        Params:
        *******
            - image: Image to process
            - cropped_indexes: [Z, Y, X] indexes cropped when fitting brain boundaries
            - random_indexes: [Z, Y, X] indexes added when random padding image

        Return:
        *******
            - image: Image recovered
    """

    def calculate_balance_crop_pad(cropped_index, random_index):

        # Getting Z, Y, X components
        (zmin_pad, zmax_pad), (ymin_pad, ymax_pad), (xmin_pad, xmax_pad) = random_index
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = cropped_index

        # Calculating pads and crops
        zmin_rebuild = zmin - zmin_pad
        zmax_rebuild = 155 - (zmax + zmax_pad)

        ymin_rebuild = ymin - ymin_pad
        ymax_rebuild = 240 - (ymax + ymax_pad)

        xmin_rebuild = xmin - xmin_pad
        xmax_rebuild = 240 - (xmax + xmax_pad)

        dims = [(zmin_rebuild, zmax_rebuild), (ymin_rebuild, ymax_rebuild), (xmin_rebuild, xmax_rebuild)]

        return dims

    def get_crop_idx(image, dims):
        crops_idx = []
        for n, (a, b) in enumerate(dims):
            # Crops
            if a < 0:
                a = -a
            else:
                a = 0

            if b >= 0:
                b = image[0].shape[n]

            crops_idx.append((a, b))

        return crops_idx

    def get_pad_idx(dims):

        pads_idx = [(0, 0)]
        for a, b in dims:
            # Pads
            if a <= 0:
                a = 0
            if b <= 0:
                b = 0

            pads_idx.append((a, b))
        return pads_idx

    # Get balance between cropped indexes and random indexes
    dims = calculate_balance_crop_pad(cropped_indexes, random_indexes)

    # cropped
    zcrop, ycrop, xcrop = [x for x in get_crop_idx(image, dims)]
    image = image[:, zcrop[0]:zcrop[1], ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]

    # padded
    image = np.pad(image, get_pad_idx(dims))

    return image

