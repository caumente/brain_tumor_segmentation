import torch
from typing import Union, Tuple
import random
import numpy as np
from torchio import CropOrPad
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from SimpleITK import GetArrayFromImage, ReadImage


def load_nii(path_folder):
    """  This function loads a NIfTI image as array."""

    return GetArrayFromImage(ReadImage(str(path_folder)))


def irm_min_max_preprocess(image: np.array, low_norm_percentile: int = 1, high_norm_percentile: int = 99) -> np.array:
    """ Main pre-processing function used for the challenge (seems to work the best).
        Clean outliers voxels by means of removing percentiles 1-99, and then min-max scale.
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros],
                              [low_norm_percentile, high_norm_percentile])  # calculate percentiles
    image = np.clip(image, low, high)  # limit the values to low and high
    image = min_max_scaler(image)
    return image


def min_max_scaler(image: np.array) -> np.array:
    """ Min max scaler function."""

    min_, max_ = np.min(image), np.max(image)
    image = (image - min_) / (max_ - min_)
    return image


def zscore_scaler(image: np.ndarray) -> np.ndarray:
    """  Z-score scaler function."""

    slices = (image != 0)
    image[slices] = (image[slices] - np.mean(image[slices])) / np.std(image[slices])
    return image


def crop_useful_image(sequences: np.ndarray, segmentation: np.ndarray
                      ) -> Union[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    This function crop the sequences and their segmentation removing as much background as possible. As result, the
    sequences and segmentation are fit to the brain boundaries.

    Params:
    *******
        sequences (np.array): MRI sequences
        segmentation (np.array): MRI segmentation

    Return:
    *******
        (zmin, zmax), (ymin, ymax), (xmin, xmax):

    """
    # Remove maximum extent of the zero-background to make future crop more useful
    z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(sequences, axis=0) != 0)
    # Add 1 pixel in each side
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    sequences = sequences[:, zmin:zmax, ymin:ymax, xmin:xmax]
    segmentation = segmentation[:, zmin:zmax, ymin:ymax, xmin:xmax]

    return (zmin, zmax), (ymin, ymax), (xmin, xmax), (sequences, segmentation)


def pad_power_two(image: np.ndarray) -> np.ndarray:
    """
    This function adds padding until the image has shape divisible by two
    """
    new_dims = np.empty(3, dtype=int)

    for n, dim in enumerate(list(image.shape)[1:]):
        dim = dim - 1
        while dim & dim - 1:
            dim = dim & dim - 1

        new_dims[n] = dim << 1
    new_dims[0] = 128

    transform = CropOrPad(tuple(new_dims))
    image_padded = transform(torch.Tensor(image)).numpy()

    return image_padded


def random_pad_or_crop(image: np.ndarray, seg: np.ndarray = None, target_size=(128, 144, 144)):
    """
    This function takes as input an  4-D array which represents an image with shape (sequences, channels, heigth, width)
     and transform it applying a crop or pad to the target size.
    """

    def get_left_right_idx_should_pad(target_size, dim):
        if dim >= target_size:
            return [False]
        elif dim < target_size:
            pad_extent = target_size - dim
            left = random.randint(0, pad_extent)
            right = pad_extent - left
            return True, left, right

    def get_crop_slice(target_size, dim):
        if dim > target_size:
            crop_extent = dim - target_size
            left = random.randint(0, crop_extent)
            right = crop_extent - left
            return slice(left, dim - right)
        elif dim <= target_size:
            return slice(0, dim)

    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)

    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg

    return image


def custom_collate(batch):
    batch = pad_batch_to_max_shape(batch)
    return default_collate(batch)


def determinist_collate(batch):
    batch = pad_batch_to_max_shape(batch)
    return default_collate(batch)


def pad_batch_to_max_shape(batch):
    shapes = (sample['label'].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem['label']
        zpad, ypad, xpad = zmax - exple.shape[1], ymax - exple.shape[2], xmax - exple.shape[3]
        assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [random.randint(0, pad) for pad in (zpad, ypad, xpad)]
        right_zpad, right_ypad, right_xpad = [pad - left_pad for pad, left_pad in
                                              zip((zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad))]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem['image'], elem['label'] = F.pad(elem['image'], pads), F.pad(elem['label'], pads)
    return batch


def pad_batch1_to_compatible_size(image):
    """ Function to reshape image to be able to divide it by 16"""

    zyx = list(image.shape[-3:])

    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            zyx[i] = ((dim // max_stride) + 1) * max_stride

    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - image.size(2), ymax - image.size(3), xmax - image.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"

    pads = (0, xpad, 0, ypad, 0, zpad)
    image = F.pad(image, pads)

    return image, (zpad, ypad, xpad)
