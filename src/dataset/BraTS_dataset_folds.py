import logging as log
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    # Mapping
    mapping = pd.read_csv(f"{path_images}/name_mapping.csv")
    mapping = mapping[['BraTS_2020_subject_ID', 'Grade']]
    mapping.replace({"Grade": {"HGG": 1, "LGG": 0}}, inplace=True)
    name_grade = dict(zip(mapping.BraTS_2020_subject_ID, mapping.Grade))

    # Splitting the path into train_path, val_path and test_path
    mapping = pd.read_csv(f"{path_images}/name_mapping.csv")
    mapping[['-', '--', 'name']] = mapping['BraTS_2020_subject_ID'].str.split("_", expand=True)
    mapping = mapping[['Grade', 'BraTS_2020_subject_ID', 'name']]
    # mapping = mapping[mapping.Grade == 'HGG']

    patients_path_train, patients_path_val, patients_path_test = [], [], []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for n, (train_ix, test_ix) in enumerate(kfold.split(mapping, mapping['Grade'])):

        train_val_mapping, test_mapping = mapping.iloc[train_ix], mapping.iloc[test_ix]
        test_mapping['fold'] = [n] * len(test_mapping)

        train, val = train_test_split(train_val_mapping, train_size=0.9, random_state=int(seed), shuffle=True,
                                      stratify=train_val_mapping['Grade'])

        patients_path_train.append([patients_dir[i] for i in train.index])
        patients_path_val.append([patients_dir[i] for i in val.index])
        patients_path_test.append([patients_dir[i] for i in test_mapping.index])

    fold_trainset, fold_valset, fold_testset = [], [], []
    for fold_train_path, fold_val_path, fold_test_path in zip(patients_path_train, patients_path_val,
                                                              patients_path_test):
        # Creating the train-validation-test datasets
        fold_trainset.append(Brats(patients_path=fold_train_path,
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
                                   data_augmentation=True,
                                   name_grade=name_grade)
                             )

        fold_valset.append(Brats(patients_path=fold_val_path,
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
                                 data_augmentation=True,
                                 name_grade=name_grade)
                           )
        fold_testset.append(Brats(patients_path=fold_test_path,
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
                                  name_grade=name_grade)
                            )

    return fold_trainset, fold_valset, fold_testset


def folded_dataset_loading(args):
    fold_trainset, fold_valset, fold_testset = get_folded_datasets(sequences=args.sequences,
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

    train_loader = [DataLoader(fold, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               pin_memory=False, drop_last=True, persistent_workers=True)
                    for fold in fold_trainset]
    val_loader = [DataLoader(fold, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             pin_memory=False, drop_last=True, persistent_workers=True)
                  for fold in fold_valset]
    test_loader = [DataLoader(fold, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=False, drop_last=True, persistent_workers=True)
                   for fold in fold_testset]

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train, val, test = get_folded_datasets(sequences=["_t1", "_t1ce", "_t2", "_flair"], regions=["et", "tc", "wt"],
                                           seed=1, debug_mode=False, crop_or_pad=(20, 20, 20),
                                           path_images="./datasets/BRATS2020/TrainingData/")

    for f in test:
        print(f)
