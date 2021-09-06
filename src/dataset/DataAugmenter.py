import torch
import logging
from torch import nn
from random import random, sample, uniform


class DataAugmenter(nn.Module):
    """
    This class augment the dataset by means of:
        - Add white noise
        - Flip images over axis 3 (Batch, Channel, Deep, Height, Width)
        - Shuffling channels (T1, T2, T1ce, FLAIR)
        - Transposing axis [2, 4]
        - Dropping channel
    """

    def __init__(self, probability=0.5, noise_only=False, channel_shuffling=False, drop_channnel=False):
        super(DataAugmenter, self).__init__()
        self.p = probability
        #self.transpose = [2, 4]
        self.flip = 3
        self.noise_only = noise_only
        self.channel_shuffling = channel_shuffling
        self.drop_channel = drop_channnel
        self.toggle = False

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:
                x = self.add_white_noise(x)
                if self.noise_only:
                    return x

                if random() < 0.2 and self.channel_shuffling:
                    new_channel_order = sample(range(x.size(1)), x.size(1))  # calculate new ordenation for channels
                    x = x[:, new_channel_order]
                    logging(f"Channel shuffling: new channel order {new_channel_order}")

                if random() < 0.2 and self.drop_channel:
                    new_channel_drop = sample(range(x.size(1)), 1)
                    x[:, new_channel_drop] = 0  # All values from a channel to 0
                    logging(f"Channel dropping: {new_channel_drop}")


                self.toggle = not self.toggle
                #new_x = x.transpose(*self.transpose).flip(self.flip)
                new_x = x.flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            if isinstance(x, list):  # case of deep supervision
                seg, deeps = x
                #reversed_seg = seg.flip(self.flip).transpose(*self.transpose)
                #reversed_deep = [deep.flip(self.flip).transpose(*self.transpose) for deep in deeps]
                reversed_seg = seg.flip(self.flip)
                reversed_deep = [deep.flip(self.flip) for deep in deeps]
                return reversed_seg, reversed_deep
            else:
                #return x.flip(self.flip).transpose(*self.transpose)
                return x.flip(self.flip)
        else:
            return x


    def add_white_noise(self, x):
        x = x * uniform(0.9, 1.1)
        std_per_channel = torch.stack(list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1))))
        noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel]).to(x.device)
        x = x + noise

        return x