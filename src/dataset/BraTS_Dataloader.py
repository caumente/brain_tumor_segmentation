from pathlib import Path
from random import random
from typing import List
from typing import Tuple

import numpy as np
import torch
from skimage import util
from torch import from_numpy
from torch.utils.data.dataset import Dataset

from src.utils.dataset import cleaning_outlier_voxels
from src.utils.dataset import fit_brain_boundaries
from src.utils.dataset import image_histogram_equalization
from src.utils.dataset import load_nii
from src.utils.dataset import random_pad_or_crop
from src.utils.dataset import scaler


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
            clipping: bool = True,
            histogram_equalization: bool = True,
            crop_or_pad: tuple = (155, 240, 240),
            fit_boundaries: bool = True,
            inverse_seq: bool = False,
            debug_mode: bool = False,
            auto_cast_bool: bool = False,
            data_augmentation: bool = False,
            name_grade: dict = None
    ):
        super(Brats, self).__init__()

        if sequences is None:
            sequences = ["_t1", "_t1ce", "_t2", "_flair"]
        self.sequences = sequences
        self.has_ground_truth = has_ground_truth
        if self.has_ground_truth and "_seg" not in self.sequences:
            self.sequences += ["_seg"]
        self.regions = regions
        self.normalization = normalization
        self.clipping = clipping
        self.low_norm_percentile = low_percentile
        self.high_norm_percentile = high_percentile
        self.histogram_equalization = histogram_equalization
        self.crop_or_pad = crop_or_pad
        self.fit_boundaries = fit_boundaries
        self.inverse_sec = inverse_seq
        self.debug_mode = debug_mode
        self.auto_cast_bool = auto_cast_bool
        self.data_augmentation = data_augmentation
        self.name_grade = name_grade
        self.data = []

        for patient_path in patients_path:
            patient_id = patient_path.name
            paths = [patient_path / f"{patient_id}{sequence}.nii.gz" for sequence in self.sequences]
            patient_info = dict(id=patient_id)

            for n, sequence in enumerate(self.sequences):
                patient_info[sequence.replace("_", "")] = paths[n]
            if not self.has_ground_truth:
                patient_info['seg'] = None

            if self.has_ground_truth and '2021' not in str(patients_path):
                if self.auto_cast_bool:
                    patient_info["grade"] = torch.tensor(self.name_grade.get(patient_id), dtype=torch.half)
                else:
                    patient_info["grade"] = torch.tensor(self.name_grade.get(patient_id), dtype=torch.float32)
            else:
                patient_info["grade"] = ""

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
        sequences = {key: load_nii(patient_info[key]) for key in patient_info if key not in ["id", "seg", "grade"]}
        if self.has_ground_truth:
            ground_truth = load_nii(patient_info["seg"])

        # Sequences clipping, equalization, normalization and stacking
        if self.clipping:
            # log.info("Clipping images")
            sequences = {key: cleaning_outlier_voxels(
                image=sequences[key],
                low_norm_percentile=self.low_norm_percentile,
                high_norm_percentile=self.high_norm_percentile
            ) for key in sequences}

        if self.histogram_equalization:
            # log.info("Equalizing histograms")
            sequences = {key: image_histogram_equalization(image=sequences[key]) for key in sequences}

        if self.normalization:
            # log.info("Images normalization")
            sequences = {key: scaler(
                image=sequences[key]
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

        if self.data_augmentation:
            # Fit the sequences to the brain boundaries by cropping and the cropping/padding to the resolution defined
            if self.fit_boundaries:
                cropped_indexes, (sequences, ground_truth) = fit_brain_boundaries(sequences=sequences,
                                                                                  segmentation=ground_truth,
                                                                                  max_dims=self.crop_or_pad)

            # Cropping/padding to the resolution defined
            sequences, ground_truth, random_indexes = random_pad_or_crop(sequences=sequences,
                                                                         segmentation=ground_truth,
                                                                         target_size=self.crop_or_pad)
        else:
            # Fit the sequences to the brain boundaries by cropping and the cropping/padding to the resolution defined
            if self.fit_boundaries:
                cropped_indexes, (sequences, ground_truth) = fit_brain_boundaries(sequences=sequences,
                                                                                  segmentation=ground_truth,
                                                                                  max_dims=(160, 224, 160)
                                                                                  )

            # Cropping/padding to the resolution defined
            sequences, ground_truth, random_indexes = random_pad_or_crop(sequences=sequences,
                                                                         segmentation=ground_truth,
                                                                         target_size=(160, 224, 160))

        if self.data_augmentation:
            # compose = transforms.Compose([
            #     transforms.RandGaussianNoise(prob=0.3, mean=0, std=0.1),
            #     transforms.RandStdShiftIntensity(factors=(1, 2), prob=0.1, nonzero=False, channel_wise=False),
            #     transforms.RandAdjustContrast(prob=0.2, gamma=(1, 1.5)),
            #     transforms.RandGaussianSmooth(prob=0.1),
            #     transforms.RandGibbsNoise(prob=0.1, alpha=(0.1, 0.2)),
            #     transforms.RandKSpaceSpikeNoise(prob=0.1, intensity_range=(2, 4)),
            #     transforms.RandCoarseDropout(prob=0.3, holes=10, spatial_size=20),
            #     ])
            # sequences = compose(sequences)

            if random() < 0.5:
                sequences, ground_truth = np.flip(sequences, 3), np.flip(ground_truth, 3)
            if random() < 0.5:
                sequences, ground_truth = np.flip(sequences, 2), np.flip(ground_truth, 2)
            if random() < 0.5:
                sequences, ground_truth = np.flip(sequences, 1), np.flip(ground_truth, 1)

        if self.auto_cast_bool:
            sequences, ground_truth = [from_numpy(x) for x in
                                       [sequences.astype("float16"), ground_truth.astype("bool")]]
        else:
            sequences, ground_truth = [from_numpy(x) for x in
                                       [sequences.astype("float32"), ground_truth.astype("bool")]]

        return dict(
            patient_id=patient_info["id"],
            sequences=sequences,
            ground_truth=ground_truth,
            grade=patient_info["grade"],
            seg_path=str(patient_info["seg"]) if self.has_ground_truth else str(""),
            cropped_indexes=cropped_indexes,
            random_indexes=random_indexes,
            et_present=et_present
        )

    def labels_to_regions(self, segmentation: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        This function takes as input the image of a medical segmentation and transform it into 3 images stacked. Each
        of the dimensions corresponds to the regions of interest.

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
