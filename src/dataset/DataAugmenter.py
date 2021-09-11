import torch
import logging
from torch import nn
from random import random, sample, uniform


class DataAugmenter(nn.Module):
    """
    This class augments the dataset by means of:
        - Add white noise
        - Flip images over axis 3 (Batch, Channel, Deep, Height, Width)
        - Shuffling channels (T1, T2, T1ce, FLAIR)
        - Transposing axis [2, 4]
        - Dropping channel
    """

    def __init__(
            self,
            probability: float = 0.5,
            flip: bool = False,
            white_noise: bool = False,
            seq_shuffling: bool = False,
            drop_seq:bool = False
    ) -> torch.Tensor:
        super(DataAugmenter, self).__init__()
        self.p = probability
        self.flip = flip
        self.white_noise = white_noise
        self.seq_shuffling = seq_shuffling
        self.drop_seq = drop_seq
        self.toggle = False

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:

                if self.white_noise:
                    x = self.add_white_noise(x)

                if random() < 0.2 and self.seq_shuffling:
                    x = channel_shuffling(x)

                if random() < 0.2 and self.drop_seq:
                    x = drop_channel(x)

                if self.flip:
                    x = x.flip(3)

                self.toggle = not self.toggle

        return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle

            return x.flip(3)
        else:
            return x


def add_white_noise(image: torch.Tensor) -> torch.Tensor:
    """
    This function takes a tensor as input an generate a white noise. The way to calculate the white noise is as follow:
        First multiply the image by a value got from a uniform distribution between 0.9 and 1.1.
        The standard deviation is calculated individually by channel.
        Generate noise as a normal distribution using mean 0 and std calculated before

    Params:
    *******
        - image (torch.Tensor): Image to transform

    Return:
    *******
        . image (torch.Tensor): Image transformed
    """

    image = image * uniform(0.9, 1.1)
    std_per_channel = torch.stack(list(torch.std(image[:, i][image[:, i] > 0]) for i in range(image.size(1))))
    noise = torch.stack([torch.normal(0, std*0.1, size=image[0, 0].shape) for std in std_per_channel]).to(image.device)
    image = image + noise

    return image


def channel_shuffling(image: torch.Tensor) -> torch.Tensor:
    """
    This function takes a tensor as input an generate a channel shuffling. In our case, since we do not have
    channels this function exchange the order of the sequences.

    Params:
    *******
        - image (torch.Tensor): Image to transform

    Return:
    *******
        . image (torch.Tensor): Image transformed
    """

    new_channel_order = sample(range(image.size(1)), image.size(1))  # calculate new ordination for channels
    image = image[:, new_channel_order]

    logging.info(f"Channel shuffling: new channel order {new_channel_order}")

    return image


def drop_channel(image: torch.Tensor) -> torch.Tensor:
    """
    This function takes a tensor as input an generate a drop channel. This means replace some of the sequences
    by a Tensor composed by zero values.

    Params:
    *******
        - image (torch.Tensor): Image to transform

    Return:
    *******
        . image (torch.Tensor): Image transformed
    """
    new_channel_drop = sample(range(image.size(1)), 1)
    image[:, new_channel_drop] = 0  # All values from a channel to 0
    logging.info(f"Channel dropping: {new_channel_drop}")

    return image
