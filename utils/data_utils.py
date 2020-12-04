import torch
from torch.autograd import Variable


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def unnormalize_batch(batch, mean_, std_, div_factor=1.0):
    """
    Unnormalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: unnormalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using dataset mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = mean_[0]
    mean[:, 1, :, :] = mean_[1]
    mean[:, 2, :, :] = mean_[2]
    std[:, 0, :, :] = std_[0]
    std[:, 1, :, :] = std_[1]
    std[:, 2, :, :] = std_[2]
    batch = torch.div(batch, div_factor)

    batch *= Variable(std)
    batch = torch.add(batch, Variable(mean))
    return batch


def linear_scaling(x):
    return (x * 255.) / 127.5 - 1.


def linear_unscaling(x):
    return (x + 1.) * 127.5 / 255.
