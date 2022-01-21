import logging as log
import os
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from skimage import util
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.utils.dataset import fit_brain_boundaries
from src.utils.dataset import cleaning_outliers_and_scaler
from src.utils.dataset import load_nii
from src.utils.dataset import random_pad_or_crop


class Brats(Dataset):
    """
    Brats class reads NIfTI files from BraTS 2020 challenge.

    Params:
    *******
        - patients_dir (List[Path]): list of paths where the images are
        - sequences (List[str], optional): list of MRI's sequences (["T1", "T2", "FLAIR", "T1ce"])
        - not_ground_truth (bool, optional): weather there is segmentation or not (inference phase)
        - regions (List[str], optional): list of regions of interest ["et", "tc", "wt"]
        - normalization (str, optional): normalization technique
        - low_percentile (int, optional): lower percentile to remove
        - high_percentile (str, optional): upper percentile to remove
        - crop_or_pad (Tuple, optional): indicates the dimensions of the images cropped (160, 224, 160)
        - fit_boundaries (Bool, optional): if true the images are preprocess to adjust them to the brain boundaries
        - inverse_seq (bool, optional): if true the sequences are inverted
                                      (https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.invert)
        - debug_mode (bool, optional): if debug_mode just three images are loaded
    """

    def __init__(
            self,
            patients_path: List[Path],
            sequences: List[str] = None,
            has_ground_truth: bool = True,
            regions: Tuple[str] = ("et", "tc", "wt"),
            normalization: bool = True,
            low_percentile: int = 1,
            high_percentile: int = 99,
            crop_or_pad: tuple = (155, 240, 240),
            fit_boundaries: bool = True,
            inverse_seq: bool = False,
            debug_mode: bool = False,
            auto_cast_bool: bool = False
    ):
        super(Brats, self).__init__()

        if sequences is None:
            sequences = ["_t1", "_t2", "_t1ce", "_flair"]
        self.sequences = sequences
        self.has_ground_truth = has_ground_truth
        if self.has_ground_truth:
            self.sequences += ["_seg"]
        self.regions = regions
        self.normalization = normalization
        self.low_norm_percentile = low_percentile
        self.high_norm_percentile = high_percentile
        self.crop_or_pad = crop_or_pad
        self.fit_boundaries = fit_boundaries
        self.inverse_sec = inverse_seq
        self.debug_mode = debug_mode
        self.auto_cast_bool = auto_cast_bool
        self.data = []

        for patient_path in patients_path:
            patient_id = patient_path.name
            paths = [patient_path / f"{patient_id}{sequence}.nii.gz" for sequence in self.sequences]

            patient_info = dict(id=patient_id)
            for n, sequence in enumerate(self.sequences):
                patient_info[sequence.replace("_", "")] = paths[n]
            if not self.has_ground_truth:
                patient_info['seg'] = None

            self.data.append(patient_info)

    def __len__(self):
        return len(self.data) if not self.debug_mode else 3

    def __getitem__(self, idx):

        # Variables initialization
        # zmin, ymin, xmin, zmax, ymax, xmax = 0, 0, 0, 155, 240, 240
        cropped_indexes = [(0, 155), (0, 240), (0, 240)]
        et_present = False
        ground_truth = None

        # Load sequences and ground truth if it exists
        patient_info = self.data[idx]
        sequences = {key: load_nii(patient_info[key]) for key in patient_info if key not in ["id", "seg"]}
        if self.has_ground_truth:
            ground_truth = load_nii(patient_info["seg"])

        # Sequences normalization and stacking
        if self.normalization:
            sequences = {key: cleaning_outliers_and_scaler(
                image=sequences[key],
                low_norm_percentile=self.low_norm_percentile,
                high_norm_percentile=self.high_norm_percentile
            ) for key in sequences}
        sequences = np.stack([sequences[sequence] for sequence in sequences])

        # Sequences inversion
        if self.inverse_sec:
            inv_sequences = np.stack([util.invert(sequence) for sequence in sequences])
            sequences = np.concatenate((sequences, inv_sequences), 0)

        # From medical segmentation using labels (1, 2, 4) to regions of interest (et, wt, tc)
        if self.has_ground_truth:
            et_present, ground_truth = self.labels_to_regions(segmentation=ground_truth)
        else:
            ground_truth = np.zeros((len(self.regions), sequences.shape[1], sequences.shape[2], sequences.shape[3]))

        # Fit the sequences to the brain boundaries by cropping and the cropping/padding to the resolution defined
        if self.fit_boundaries:
            cropped_indexes, (sequences, ground_truth) = fit_brain_boundaries(sequences=sequences,
                                                                              segmentation=ground_truth,
                                                                              max_dims=self.crop_or_pad)

        # Cropping/padding to the resolution defined
        sequences, ground_truth, random_indexes = random_pad_or_crop(sequences=sequences, segmentation=ground_truth, target_size=self.crop_or_pad)

        # Type casting for sequences and ground truth
        if self.auto_cast_bool:
            sequences, ground_truth = [from_numpy(x) for x in [sequences.astype("float16"), ground_truth.astype("bool")]]
        else:
            sequences, ground_truth = [from_numpy(x) for x in [sequences.astype("float32"), ground_truth.astype("bool")]]

        return dict(
            patient_id=patient_info["id"],
            sequences=sequences,
            ground_truth=ground_truth,
            seg_path=str(patient_info["seg"]) if self.has_ground_truth else str(""),
            cropped_indexes=cropped_indexes,
            random_indexes=random_indexes,
            et_present=et_present
        )

    def labels_to_regions(self, segmentation: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        This function takes as input the image of a medical segmentation and transform it into 3 images stacked. Each
        of the dimensions correspond to the regions of interest.

        - The region ET (enhancing tumor) is composed by the label 4
        - The region TC (necrotic & not enhancing tumor core) is composed by the labels 2 and 4
        - The WT (whole tumor) is composed by all the labels, 1, 2 and 4

        Params:
        *******
            - img_segmentation: image segmented to transform into regions

        Return:
        *******
            - et_present: it is true if the segmentation possess the ET region
            - img_segmentation: stack of images of the regions
        """
        et = segmentation == 4
        tc = np.logical_or(segmentation == 4, segmentation == 1)
        wt = np.logical_or(tc, segmentation == 2)
        regions_dict = {"et": et, "tc": tc, "wt": wt}

        et_present = True if np.sum(et) >= 1 else False
        segmentation = np.stack([regions_dict[region] for region in self.regions])

        return et_present, segmentation

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

    def calculate_balance_crop_pad(cropped_indexes, random_indexes):

        # Getting Z, Y, X components
        (zmin_pad, zmax_pad), (ymin_pad, ymax_pad), (xmin_pad, xmax_pad) = random_indexes
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = cropped_indexes

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


def get_datasets(
        sequences,
        regions,
        seed,
        debug_mode,
        has_ground_truth=True,
        path_images="",
        normalization=True,
        low_norm_percentile=1,
        high_norm_percentile=99,
        crop_or_pad=(155, 240, 240),
        fit_boundaries=True,
        inverse_seq=False,
        auto_cast_bool=True
):

    # Checking if the path where the images are exist
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
                                                                          train_size=0.8)

    # Creating the train-validation-test datasets
    train_dataset = Brats(patients_path=train_path,
                          sequences=sequences,
                          has_ground_truth=has_ground_truth,
                          regions=regions,
                          normalization=normalization,
                          low_percentile=low_norm_percentile,
                          high_percentile=high_norm_percentile,
                          crop_or_pad=crop_or_pad,
                          fit_boundaries=fit_boundaries,
                          inverse_seq=inverse_seq,
                          debug_mode=debug_mode,
                          auto_cast_bool=auto_cast_bool)
    val_dataset = Brats(patients_path=val_path,
                        sequences=sequences,
                        has_ground_truth=has_ground_truth,
                        regions=regions,
                        normalization=normalization,
                        low_percentile=low_norm_percentile,
                        high_percentile=high_norm_percentile,
                        crop_or_pad=crop_or_pad,
                        fit_boundaries=fit_boundaries,
                        inverse_seq=inverse_seq,
                        debug_mode=debug_mode,
                        auto_cast_bool=auto_cast_bool)
    test_dataset = Brats(patients_path=test_path,
                         sequences=sequences,
                         has_ground_truth=has_ground_truth,
                         regions=regions,
                         normalization=normalization,
                         low_percentile=low_norm_percentile,
                         high_percentile=high_norm_percentile,
                         crop_or_pad=crop_or_pad,
                         fit_boundaries=fit_boundaries,
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


def train_test_val_split_BraTS_2020(mapping, patients_path, seed, train_size=0.8):
    """ This function splits the dataset into train-val-test based on a train_size."""

    mapping[['-', '--', 'name']] = mapping['BraTS_2020_subject_ID'].str.split("_", expand=True)
    mapping = mapping[['Grade', 'BraTS_2020_subject_ID', 'name']]

    train, val_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True,
                                   stratify=mapping['Grade'])
    val, test = train_test_split(val_, test_size=0.5, random_state=int(seed), shuffle=True, stratify=val_['Grade'])
    train_idx, val_idx, test_idx = train.index, val.index, test.index
    log.info(f"Patients used for training:\n {train}")
    log.info(f"Patients used for validating:\n {val}")
    log.info(f"Patients used for testing:\n {test}")

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
                                                            low_norm_percentile=args.low_norm,
                                                            high_norm_percentile=args.high_norm,
                                                            crop_or_pad=args.crop_or_pad,
                                                            fit_boundaries=args.fit_boundaries,
                                                            inverse_seq=args.inverse_seq,
                                                            auto_cast_bool=args.auto_cast_bool)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=False,
                                             num_workers=args.workers, )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader, test_loader
