import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class BasicConvModule(nn.Module):
    '''BasicConvModule block
        Conv-normalze-activate

    '''
    def __init__(self, in_channels, out_channels, activate=None,normalized=None, deconv=False, is_3d=False, **kwargs):
        super(BasicConvModule, self).__init__()


        self.type_normal=normalized
        if activate=="relu":
            self.activate = nn.ReLU(inplace=True)
        elif activate=="lrelu": #"leaky relu"
            self.activate = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        elif activate is None:
            self.activate = None
        else:
            raise NotImplemented("Activate funtion can not be recognized")

        self.bias=False
        if is_3d:
            if normalized == "BN":
                self.normalized = nn.BatchNorm3d(out_channels)
            elif normalized == "IN":
                self.normalized = nn.InstanceNorm3d(out_channels)
                self.bias =True

            elif normalized is None:
                self.normalized = None
            else:
                raise NotImplemented("Normalized can not be recognized")


            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=self.bias, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=self.bias, **kwargs)


        else:
            if normalized == "BN":
                self.normalized = nn.BatchNorm2d(out_channels)
            elif normalized == "IN":
                self.normalized = nn.InstanceNorm2d(out_channels)
                self.bias = True
            elif normalized == "AdaIN":
                self.normalized = AdaptiveInstanceNorm2d(out_channels)
                self.bias = True
            elif normalized is None:
                self.normalized = None
            else:
                raise NotImplemented("Normalized can not be recognized")

            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=self.bias, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=self.bias, **kwargs)

    def forward(self, x,param=None):
        x = self.conv(x)

        if self.normalized:
            if self.type_normal=="AdaIN":

                x = self.normalized(x,param)
            else:
                x = self.normalized(x)
        if self.activate:
            x = self.activate(x)
        return x


class Conv2xModule(nn.Module):

    '''
    Conv2x Module

    '''

    def __init__(self, in_channels, out_channels,activate="relu",normalized="BN",deconv=False, is_3d=False):
        super(Conv2xModule, self).__init__()

        self.type_normal=normalized
        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3


        self.conv = BasicConvModule(in_channels, out_channels,activate=activate,normalized=normalized, deconv=deconv, is_3d=is_3d, kernel_size=kernel,
                               stride=2, padding=1)

    def forward(self, x,param=None):

        if self.type_normal=="AdaIN":
            return self.conv(x,param)
        else:
            return self.conv(x)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output



class ResidualBlock_IN(nn.Module):
    '''
     Resduial bolck with Instance normalized
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_IN, self).__init__()
        self.block = nn.Sequential(

            BasicConvModule(in_channels=nf,out_channels=nf,activate="relu",normalized="IN",is_3d=False,deconv=False,kernel_size=3, stride=1, padding=1),
            BasicConvModule(in_channels=nf, out_channels=nf, activate="relu", normalized="IN", is_3d=False,deconv=False, kernel_size=3, stride=1, padding=1)

        )


    def forward(self, x):

        return x + self.block(x)


class ResidualBlock(nn.Module):
    '''
     Resduial bolck with Instance normalized
    '''

    def __init__(self, nf=64,norm='BN',activate='relu'):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(

            BasicConvModule(in_channels=nf,out_channels=nf,activate=activate,normalized=norm,is_3d=False,deconv=False,kernel_size=3, stride=1, padding=1),
            BasicConvModule(in_channels=nf, out_channels=nf, activate=None, normalized=norm, is_3d=False,deconv=False, kernel_size=3, stride=1, padding=1)

        )


    def forward(self, x):

        return x + self.block(x)
class ResidualCBAMBlock(nn.Module):
    '''
     Resduial bolck with Instance normalized
    '''

    def __init__(self, nf=64,norm='BN',activate='relu'):
        super(ResidualCBAMBlock, self).__init__()
        self.input = BasicConvModule(in_channels=nf,out_channels=nf,activate=activate,normalized=norm,is_3d=False,deconv=False,kernel_size=3, stride=1, padding=1)


        self.sa=SpatialAttention()
        self.output=BasicConvModule(in_channels=nf, out_channels=nf, activate=None, normalized=norm, is_3d=False, deconv=False,kernel_size=3, stride=1, padding=1)


    def forward(self, x,):
        res = x
        x = self.input(x)
        x = self.sa(x)*x
        x = self.output(x)
        return x + res
class ResidualCBAMBlockAdaIN(nn.Module):
    '''
     Resduial bolck with Instance normalized
    '''

    def __init__(self, nf=64,norm='BN',activate='relu'):
        super(ResidualCBAMBlockAdaIN, self).__init__()
        self.type_norm=norm
        self.input = BasicConvModule(in_channels=nf,out_channels=nf,activate=activate,normalized=norm,is_3d=False,deconv=False,kernel_size=3, stride=1, padding=1)


        self.sa=SpatialAttention()
        self.output=BasicConvModule(in_channels=nf, out_channels=nf, activate=None, normalized=norm, is_3d=False, deconv=False,kernel_size=3, stride=1, padding=1)


    def forward(self, x,param=None):
        res = x
        if self.type_norm=="AdaIN":
            x = self.input(x,param)
        else:
            x = self.input(x)
        x = self.sa(x)*x
        if self.type_norm == "AdaIN":
            x = self.output(x,param)
        else:
            x = self.output(x)

        return x + res



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()




        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def init_weights(net, init_type='normal', activate='relu',init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',nonlinearity=activate)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, out_channels):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.innorm = nn.InstanceNorm2d(out_channels)
        # weight_spatial

    def forward(self, x, params):
        # assert weight_spatial is not None and bias_spatial is not None, \
        #     'Please assign spatial weight and bias before calling AdaIN!'
        [weight_spatial ,bias_spatial] = params
        x = self.innorm(x)
        x = weight_spatial * x + bias_spatial

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.out_channels) + ')'
