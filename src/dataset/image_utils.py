import torch
from typing import Union, Tuple
import random
import numpy as np
from torchio import CropOrPad
from SimpleITK import GetArrayFromImage, ReadImage


def load_nii(path_folder):
    """  This function loads a NIfTI image as array."""

    return GetArrayFromImage(ReadImage(str(path_folder)))


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


def cleaning_outliers_and_scaler(
        image: np.array,
        low_norm_percentile: int = 1,
        high_norm_percentile: int = 99,
        scaler: str = "min_max"
) -> np.array:
    """
        This function cleans outliers voxels by means of clipping values to the percentiles interval 1-99.
        After that the images are scaled using Min-Max or Z-score technique.

        Params:
        *******
            - image (np.array): image to process
            - low_norm_percentile (int, Optional): it defines the lower limit
            - high_norm_percentile (int, Optional): it defines the upper limit
            - scaler (str, Optional): strategy to normalize the image.
                - Min-Max: x-min(x)/(max(x)-min(x))
                - Z-score: x-mean(x)/std(x)

        Return:
        *******
            - image: image post-process
    """

    # Calculate percentiles using non-zero values and limit the values to the low and high
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_norm_percentile, high_norm_percentile])
    image = np.clip(image, low, high)

    # Image normalization
    if scaler == "min_max":
        image = min_max_scaler(image)
    elif scaler == "z_score":
        image = zscore_scaler(image)

    return image


def crop_useful_image(
        sequences: np.ndarray,
        segmentation: np.ndarray
) -> Union[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    This function crop the sequences and its segmentation removing as much background as possible. As result, the
    sequences and segmentation are fit to the brain boundaries.

    Params:
    *******
        - sequences (np.array): MRI sequences.
        - segmentation (np.array): MRI segmentation

    Return:
    *******
        - (zmin, zmax): limits on Z dimension
        - (ymin, ymax): limits on Y dimension
        - (xmin, xmax): limits on X dimension
        - (sequences, segmentation): sequences and segmentation fitted to brain boundaries

    """

    # Getting all non-xero indexes
    z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(sequences, axis=0) != 0)

    # Calculating lower and upper boundaries by each dimension. Add a extra pixel in each dimension
    zmin, ymin, xmin = [max(0, int(np.min(idx) - 1)) for idx in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

    # Fitting sequences and segmentation to brain boundaries
    sequences = sequences[:, zmin:zmax, ymin:ymax, xmin:xmax]
    segmentation = segmentation[:, zmin:zmax, ymin:ymax, xmin:xmax]

    return (zmin, zmax), (ymin, ymax), (xmin, xmax), (sequences, segmentation)


def random_pad_or_crop(
        sequences: np.ndarray,
        segmentation: np.ndarray = None,
        target_size: tuple = (155, 240, 240)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes as input an  4D array which represents an image with shape (sequences, depth, height, width)
     and transform it applying a crop or pad to the target size.
    """

    def get_left_right_idx_should_pad(
            out_resolution: int,
            in_resolution: int
    ) -> Union[list[bool], Tuple[bool, int, int]]:
        """
        This function gets the indexes to pad the images based on the current and target resolution. If the
        in size is greater or equal than the out size a False is returned. Else, the left and right pads are returned.

        Params:
        *******
            - out_resolution: it defines the target resolution
            - in_resolution: it defines the current resolution

        Return:
        *******
            - slice: it returns the slice to crop
        """

        if in_resolution >= out_resolution:
            return [False]
        elif in_resolution < out_resolution:
            pad_extent = out_resolution - in_resolution
            left = random.randint(0, pad_extent)
            right = pad_extent - left
            return True, left, right

    def get_crop_slice(out_resolution: int, in_resolution: int) -> slice:
        """
        This function gets the slices to crop the images based on the current and target resolution. If the
        current size is greater than the target size, it is returned the slices to crop the image randomly to
        adjust it to the size. Else it is returned a slice of the current dimensions.

        Params:
        *******
            - out_resolution: it defines the target resolution
            - in_resolution: it defines the current resolution

        Return:
        *******
            - slice: it returns the slice to crop
        """

        if in_resolution > out_resolution:
            extra_size = in_resolution - out_resolution
            left = random.randint(0, extra_size)
            right = extra_size - left
            return slice(left, in_resolution - right)
        else:
            return slice(0, in_resolution)

    # Cropping the sequences and segmentation if it is necessary
    c, z, y, x = sequences.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, current) for target, current in zip(target_size, (z, y, x))]
    sequences, segmentation = sequences[:, z_slice, y_slice, x_slice], segmentation[:, z_slice, y_slice, x_slice]

    # Padding the sequences and segmentation if it is necessary
    pads = [get_left_right_idx_should_pad(target, current) for target, current in zip(target_size, [z, y, x])]
    pad_list = [(0, 0)]
    for pad in pads:
        if pad[0]:
            pad_list.append((pad[1], pad[2]))
        else:
            pad_list.append((0, 0))
    sequences, segmentation = np.pad(sequences, pad_list), np.pad(segmentation, pad_list)

    return sequences, segmentation


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
