import torch
import numpy as np
import kornia
import copy

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from datasets.graffiti_dataset.dataset import DatasetSample as sample_graffiti


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


def put_graffiti(args, mask_smoother):
    tensorize = transforms.ToTensor()
    resizer = transforms.Resize((args.DATASET.SIZE, args.DATASET.SIZE))
    sample = sample_graffiti(args.TEST.GRAFFITI_PATH)
    masks = tensorize(resizer(Image.fromarray(sample.graffiti_mask))).cuda()
    graffiti_img = tensorize(resizer(Image.fromarray(sample.image))).cuda()
    c_x = graffiti_img * masks
    smooth_masks = mask_smoother(1 - masks) + masks
    smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
    return c_x, smooth_masks


def paste_facade(args, x, c_x, mask_smoother):
    facades = kornia.geometry.scale(c_x, torch.tensor([1 / 8]))
    facades = linear_scaling(facades)
    coord_x, coord_y = np.random.randint(args.DATASET.SIZE - args.DATASET.SIZE // 8, size=(2,))
    x_scaled = copy.deepcopy(x)
    x_scaled[:, :, coord_x:coord_x + facades.size(2), coord_y:coord_y + facades.size(3)] = facades
    masks = torch.zeros((facades.size(0), 1, args.DATASET.SIZE, args.DATASET.SIZE)).cuda()
    masks[:, :, coord_x:coord_x + facades.size(2), coord_y:coord_y + facades.size(3)] = torch.ones_like(facades)
    smooth_masks = mask_smoother(1 - masks) + masks
    smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
    return x_scaled, smooth_masks


def put_text(args, x, COLORS, mask_smoother):
    to_pil = transforms.ToPILImage()
    tensorize = transforms.ToTensor()

    text = args.TEST.TEXT
    mask = to_pil(torch.zeros_like(x[0]).squeeze(0).cpu())
    d_m = ImageDraw.Draw(mask)
    font = ImageFont.truetype(args.TEST.FONT, args.TEST.FONT_SIZE)
    font_w, font_h = d_m.textsize(text, font=font)
    c_w = (args.DATASET.SIZE - font_w) // 2
    c_h = (args.DATASET.SIZE - font_h) // 2
    d_m.text((c_w, c_h), text, font=font, fill=(255, 255, 255))

    masks = tensorize(mask)[0].unsqueeze(0).cuda()
    smooth_masks = mask_smoother(1 - masks) + masks
    smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

    c_x = torch.ones_like(x) * torch.tensor(np.random.choice(list(COLORS.values()))).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

    return c_x, smooth_masks


def swap_faces(args, x, c_x, mask_smoother):
    c_x = kornia.geometry.center_crop(c_x, (args.DATASET.SIZE // 2, args.DATASET.SIZE // 2))

    coord_x = coord_y = (args.DATASET.SIZE - args.DATASET.SIZE // 2) // 2
    x_scaled = copy.deepcopy(x)
    x_scaled[:, :, coord_x:coord_x + args.DATASET.SIZE // 2, coord_y:coord_y + args.DATASET.SIZE // 2] = c_x

    masks = torch.zeros((x.size(0), 1, args.DATASET.SIZE, args.DATASET.SIZE)).cuda()
    masks[:, :, coord_x:coord_x + args.DATASET.SIZE // 2, coord_y:coord_y + args.DATASET.SIZE // 2] = torch.ones_like(c_x)

    smooth_masks = mask_smoother(1 - masks) + masks
    smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

    return x_scaled, smooth_masks
