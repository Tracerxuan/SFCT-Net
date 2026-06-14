import torch.nn.functional as F
import pywt
import pywt.data
import torch
import torch.nn as nn
from functools import partial

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class Wavelet(nn.Module):
    def __init__(self, in_channels, out_channels, wt_ll=False, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels
        self.wt_ll = wt_ll
        self.wt_filter, _ = create_wavelet_filter(wt_type, in_channels, out_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)

    def forward(self, x):
        if self.wt_ll:
            x = x[:, :, 0, :, :]

        curr_shape = x.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            x = F.pad(x, curr_pads)

        x = self.wt_function(x)
        return x

class WavePad(nn.Module):
    def __init__(self, out_channels, i):
        super().__init__()
        self.num = i
        self.out_channels = out_channels
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = x[::-1]
        for i in range(len(x)):
            curr_shape = x[i].shape
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                for j in range(i, len(x)):
                    curr_pads = (0, (curr_shape[3] % 2) * (j - i + 1), 0, (curr_shape[2] % 2) * (j - i + 1))
                    x[j] = F.pad(x[j], curr_pads)
        assert x[self.num].shape[1] == self.out_channels
        out = self.conv(x[self.num])
        return out

class IWavelet(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels
        _, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, out_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

    def forward(self, x):
        shape_x = x.shape
        next_x = self.iwt_function(x)
        next_x = next_x[:, :, :shape_x[3] * 2, :shape_x[4] * 2]
        return next_x

class WaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        assert in_channels == out_channels
        self.wavelet_convs = nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=kernel_size, padding='same',
                                       stride=1, dilation=1, groups=out_channels * 4, bias=False)

    def forward(self, x):
        shape_x = x.shape
        x_map = x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        x_map = self.wavelet_convs(x_map)
        x = x_map.reshape(shape_x)
        return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.wavelet_convs = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1,
                                       dilation=1, groups=in_channels * 4, bias=False)
        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        curr_x_ll = x
        curr_shape = curr_x_ll.shape

        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            curr_x_ll = F.pad(curr_x_ll, curr_pads)

        curr_x = self.wt_function(curr_x_ll)
        curr_x_ll = curr_x[:, :, 0, :, :]

        shape_x = curr_x.shape
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        curr_x_tag = self.wavelet_scale(self.wavelet_convs(curr_x_tag))
        curr_x_tag = curr_x_tag.reshape(shape_x)

        curr_x_ll = curr_x_tag[:, :, 0, :, :]
        curr_x_h = curr_x_tag[:, :, 1:4, :, :]

        curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
        next_x_ll = self.iwt_function(curr_x)

        next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_iwt = next_x_ll

        if self.do_stride is not None:
            x_iwt = self.do_stride(x_iwt)

        return x_iwt
