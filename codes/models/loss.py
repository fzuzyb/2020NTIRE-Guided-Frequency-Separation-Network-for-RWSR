import torch
import torch.nn as nn
import sys
# sys.path.insert(0, './PerceptualSimilarity')
# import PerceptualSimilarity as ps
import random
import models.arch.arch_util as arch_util
from torchvision.models.vgg import vgg16, vgg19

import torch
import torch.nn.functional as F
class CharbonnierLoss(nn.Module):


    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class SRLoss(nn.Module):
    """Define different SRLoss objectives.
    The SRLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, loss_type):
        """ Initialize the SRLoss class.
        Parameters:
            gan_mode (str) - - the type of SRLoss objective. It currently supports vanilla, lsgan, and wgangp.

        """
        super(SRLoss, self).__init__()

        self.loss_type = loss_type
        if loss_type == 'L2':
            self.loss = nn.MSELoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'CR':
            self.loss = CharbonnierLoss()
        elif loss_type == 'L1_SSIM':
            self.loss1=nn.L1Loss()
            self.loss2=SSIM()
        elif loss_type == 'L1_MS_SSIM':
            self.loss1=nn.L1Loss()
            self.loss2=MS_SSIM()
        elif loss_type == 'CR_MS_SSIM':
            self.loss1=CharbonnierLoss()
            self.loss2=MS_SSIM()



        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_combine_loss(self,SR,HR):
        return self.loss1(SR,HR),1/self.loss2(SR,HR)

    def __call__(self, SR,HR):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            SR (tensor) - -
            HR (tensor) - -
        Returns:
            the calculated loss.
        """
        if self.loss_type in ["L1_SSIM",'L1_MS_SSIM','CR_MS_SSIM']:
            loss=self.get_combine_loss(SR,HR)
        else:
            loss=[self.loss(SR,HR)]
        return loss
# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_type, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_type
        if gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients

    else:
        return 0.0, None

# class PerceptualLossLPIPS(nn.Module):
#     def __init__(self):
#         super(PerceptualLossLPIPS, self).__init__()
#         self.loss_network = ps.PerceptualLoss(use_gpu=torch.cuda.is_available())
#
#     def forward(self, x, y):
#         return self.loss_network.forward(x, y, normalize=True).mean()
class PerceptualLossVGG16(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG16, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))


class PerceptualLossVGG19(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))

class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_type='VGG16',rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()


        if perceptual_type =='VGG16':
            self.loss = PerceptualLossVGG16()
        elif perceptual_type =='VGG19':
            self.loss = PerceptualLossVGG19()

        else:
            raise NotImplementedError("{} has not been implemented".format(perceptual_type))
        self.rotations = rotations
        self.flips = flips

    def forward(self, x, y):
        if self.rotations:
            k_rot = random.choice([-1, 0, 1])
            x = torch.rot90(x, k_rot, [2, 3])
            y = torch.rot90(y, k_rot, [2, 3])
        if self.flips:
            if random.choice([True, False]):
                x = torch.flip(x, (2,))
                y = torch.flip(y, (2,))
            if random.choice([True, False]):
                x = torch.flip(x, (3,))
                y = torch.flip(y, (3,))
        return self.loss(x, y)


class ColorLoss(nn.Module):
    def __init__(self,recursions=1,use_gaussian=False):
        super(ColorLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.color_filter = arch_util.FilterLow(recursions=recursions,gaussian=use_gaussian)


    def forward(self, x, y):
        # Color Loss
        loss = self.pixel_loss(self.color_filter(x), self.color_filter(y))
        return loss



class GradientPenaltyLoss(nn.Module):
    def __init__(self,device):
        super(GradientPenaltyLoss, self).__init__()
        self.device=device
    def forward(self, netD,real_data, fake_data):



        alpha = torch.rand(real_data.shape[0], 1, device=self.device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)


        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1.0) ** 2).mean()        # added eps
        return gradient_penalty




def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
    Returns:
        torch.Tensor: ssim results
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    if nonnegative_ssim:
        cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, K=(0.01, 0.03),
         nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True, K=K, nonnegative_ssim=nonnegative_ssim)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None,
            K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True, K=K, nonnegative_ssim=nonnegative_ssim)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3, K=(0.01, 0.4),
                 nonnegative_ssim=True):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average, K=self.K,
                    nonnegative_ssim=self.nonnegative_ssim)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3, weights=None,
                 K=(0.01, 0.4), nonnegative_ssim=True):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights, K=self.K, nonnegative_ssim=self.nonnegative_ssim)