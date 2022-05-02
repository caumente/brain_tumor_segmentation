from pathlib import Path
from typing import List
import logging as log
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataset.BraTS_Dataloader import Brats


def get_folded_datasets(
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
        auto_cast_bool=True
):

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    patients_dir = sorted([x for x in path_images.iterdir() if x.is_dir()])
    log.info(f"Images are contained in the following path: {path_images}")

    # Splitting the path into train_path, val_path and test_path
    mapping = pd.read_csv(f"{path_images}/name_mapping.csv")
    folds_path, test_path = train_test_split_BraTS_2020(mapping=mapping,
                                                        patients_path=patients_dir,
                                                        seed=seed,
                                                        train_size=0.9
                                                        )

    fold_dataset = []
    for fold_path in folds_path:
        # Creating the train-validation-test datasets
        fold_dataset.append(Brats(patients_path=fold_path,
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
                            )

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

    log.info(f"Size of train dataset: {len(fold_dataset[0])}")
    log.info(f"Shape of images used for training: {fold_dataset[0][0]['sequences'].shape}")
    log.info(f"Size of test dataset: {len(test_dataset)}")
    log.info(f"Shape of images used for testing: {test_dataset[0]['sequences'].shape}")

    return fold_dataset, test_dataset


def train_test_split_BraTS_2020(
        mapping: pd.DataFrame,
        patients_path: List[Path],
        seed: int,
        train_size: float = 0.9
):

    """ This function splits the dataset into train-val-test based on a train_size."""

    mapping[['-', '--', 'name']] = mapping['BraTS_2020_subject_ID'].str.split("_", expand=True)
    mapping = mapping[['Grade', 'BraTS_2020_subject_ID', 'name']]

    train, test = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True,
                                   stratify=mapping['Grade'])

    # creating folds
    lggs = train[train.Grade == "LGG"]
    hggs = train[train.Grade == "HGG"]

    fold1, fold2, fold3, fold4 = np.array_split(hggs, 4)
    folds = [pd.concat([fold, lggs]) for fold in [fold1, fold2, fold3, fold4]]
    folds_idx = [fold.index for fold in folds]

    # fold1_idx, fold2_idx, fold3_idx, fold4_idx = fold1.index, fold2.index, fold3.index, fold4.index
    test_idx = test.index

    fold1 = [patients_path[i] for i in folds_idx[0]]
    fold2 = [patients_path[i] for i in folds_idx[1]]
    fold3 = [patients_path[i] for i in folds_idx[2]]
    fold4 = [patients_path[i] for i in folds_idx[3]]
    test = [patients_path[i] for i in test_idx]

    return [fold1, fold2, fold3, fold4], test


def folded_dataset_loading(args):
    folds_dataset, test_dataset = get_folded_datasets(sequences=args.sequences,
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
                                                      auto_cast_bool=args.auto_cast_bool)

    fold_train_loader = [DataLoader(fold, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                    pin_memory=False, drop_last=True, persistent_workers=True)
                         for fold in folds_dataset]
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    return fold_train_loader, test_loader


# if __name__ == '__main__':
#     folds, test = get_folded_datasets(sequences=["_t1", "_t1ce", "_t2", "_flair"], regions=["et", "tc", "wt"],
#                                       seed=1, debug_mode=False, crop_or_pad=(20, 20, 20),
#                                       path_images="./datasets/BRATS2020/TrainingData/")
#
#     print(len(test))
#
#     for f in folds:
#         print(len(f))

