
import torch
import torch.nn.functional as F

import numpy as np

from .utils import default_image_normalizer, t2a


def compute_embed_shape(net, shape):
    """ cmpute shape in the resulting feature embedding
    given an input of size shape [C,H,W] """

    tensor_example = torch.zeros((1,) + tuple(shape))

    forwarded = net.forward(tensor_example)

    unbatched_size = tuple(forwarded.shape)[1:]

    return unbatched_size


def flatten_batch(batch):

    flattened_shape = tuple(batch.shape[0:1]) + \
                            (int(np.prod(batch.shape[1:])),)
    flattened = batch.reshape(flattened_shape)
    return flattened


class BatchFlattener(torch.nn.Module):

    def forward(self, batch):
        return flatten_batch(batch)


class RunningMean(torch.nn.Module):
    def __init__(self, window_size, stride=1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        self._c_window = None
        self._c_stride = None

    def forward(self, batch):

        return [self.forward_single(b) for b in batch]

    def forward_single(self, data):
        if self.window_size == 1 and self.stride == 1:
            return data

        elif self.window_size == -1:
            if data.shape[0] == 1:
                return data
            else:
                return data.mean(dim=0)[None, ...]

        else:
            if self._c_window is None or self._c_stride is None:
                self._c_window = torch.ones(1, 1, self.window_size, 1) / self.window_size
                self._c_stride = (self.stride, 1)

            ravg = torch.nn.functional.conv2d(data[None, None, ...],
                                              self._c_window,
                                              stride=self._c_stride)

            return ravg[0, 0, ...]


def GlobalMean():
    return RunningMean(-1)


class MultiRunningMean(torch.nn.Module):
    def __init__(self, window_sizes, strides):
        assert len(window_sizes) == len(strides)

        super().__init__()
        self.window_sizes = window_sizes
        self.strides = strides

        self.running_means = [RunningMean(w, s) for w, s in zip(self.window_sizes, self.strides)]

    def forward(self, data):
        ret_l = np.array([r_m.forward(data) for r_m in self.running_means], dtype=object).T.tolist()
        return [torch.cat(r) for r in ret_l]


class RecursiveReduction(torch.nn.Module):
    def __init__(self, elements=None, window_size=2, stride=2,
                 activation=torch.nn.functional.leaky_relu):
        super().__init__()

        self.window_size = window_size
        self.stride = stride

        self.activation = activation

        self.c = None
        if elements is not None:
            self.init_conv(elements)

    def forward(self, batch):
        self.init_conv(batch[0].shape[-1])
        return [self.forward_single(b) for b in batch]

    def forward_single(self, data):
        data  = data[None, None, ...].permute(0, 3, 1, 2)
        # i need to finish this
        while data.shape[-1] > 1:
            to_pad = self.stride - (data.shape[-1] - self.window_size) % self.stride

            if to_pad < self.stride:
                l_pad = np.floor(to_pad / 2).astype(int)
                u_pad = np.ceil(to_pad / 2).astype(int)

                data = F.pad(data, (l_pad, u_pad, 0, 0), mode="replicate")
            data = self.activation(self.c(data))
        
        return data.squeeze()[None, ...]

    def init_conv(self, elements):
        if self.c is None:
            w = torch.eye(elements)[None, ...].repeat((self.window_size,1,1)) 
            w = (w / self.window_size)[None, ...].permute(2,3,0,1)
            self.c = torch.nn.Conv2d(elements, 
                                     elements,
                                     (1, self.window_size),
                                     stride=self.stride, bias=False)
            c_w = self.c._parameters["weight"]

            self.c._parameters["weight"] = torch.nn.Parameter( c_w *1e-1 +  w)


class ReductionByDistance(torch.nn.Module):

    def __init__(self, cutoff):
        super().__init__()
        self._cutoff = cutoff

    def forward(self, batch):
        return [self.forward_single(b) for b in batch]

    def forward_single(self, elem):
        elem_res = []

        representative, rep_id = next(itr)

        cutoff = self.get_cutoff(elem)

        for frame, f_ind in itr:
            d = torch.norm(representative - frame)
            if d > cutoff:
                elem_res.append(torch.mean(elem[rep_id:f_ind], dim=0))
                representative = frame
                rep_id = f_ind

        elem_res.append(torch.mean(elem[rep_id:], dim=0))

        return torch.stack(elem_res)


    def get_cutoff(self, data):
        if callable(self._cutoff):
            return self._cutoff(data)
        else:
            return self._cutoff



class InternalCutoff(object):


    def __init__(self, rel_cutoff, stride=1, percentile=None):
        self.rel_cutoff = rel_cutoff
        self.stride = stride
        self.percentile = percentile
        if self.percentile is not None:
            assert self.percentile >= 0 and self.percentile <= 100

    def __call__(self, data):

        d1 = data[:-self.stride, ...]
        d2 = data[self.stride:, ...]

        euclideans = F.pairwise_distance(d1, d2)

        if self.percentile is not None:
            value = np.percentile(t2a(euclideans), self.percentile)
        else:
            value = t2a(euclideans.mean())

        return float(value) * self.rel_cutoff


class SequenceSequential(torch.nn.Sequential):

    def forward(self, data):

        i_len = [len(item) for item in data]
        data = torch.cat(data)
        data = super().forward(data)

        data = torch.split(data, i_len)

        return data


_AGGREGATORS = {"mean": GlobalMean,
                "running_mean": RunningMean,
                "multi_running_mean": MultiRunningMean,
                "recursive": RecursiveReduction}


def get_aggregator(key):
    return _AGGREGATORS[key]
