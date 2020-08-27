
import random


import torch
import numpy as np
import sklearn.utils


class ImageNormalizer(torch.nn.Module):

    def __init__(self, mean, std, channel_first=True):
        super(ImageNormalizer, self).__init__()
        self.channel_first = channel_first
        if self.channel_first:
            mean = torch.from_numpy(np.asarray(mean)[:, None, None])
            std = torch.from_numpy(np.asarray(std)[:, None, None])
        else:
            mean = torch.from_numpy(np.asarray(mean)[:, None, None])
            std = torch.from_numpy(np.asarray(std)[:, None, None])

        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, batch):
        zero_to_one = batch.float() / 255.
        return (zero_to_one - self.mean) / self.std


def default_image_normalizer():
    return ImageNormalizer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def t2a(tensor):
    return tensor.cpu().detach().numpy()


def a2t(array_like):
    return astensor(array_like)


def assqarray(data):
    if isinstance(data, (np.ndarray, list, tuple)):
        if isinstance(data, np.ndarray):
            sq = data.squeeze()
            if sq.shape == ():
                sq = np.array([sq])
        else:
            raise ValueError("unsupported type {} ".format(type(data)))

    else:
        sq = np.array([data])

    return sq


def astensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)


def a_app(arr, elem, ndim=1):

    elem_ndim = len(elem.shape)
    if ndim == elem_ndim:
        elem_exp = elem
    else:
        final_dim = (None,) * (ndim - elem_ndim) + (...,)
        elem_exp = elem[final_dim]

    if len(arr) == 0:
        return elem_exp.copy()
    else:
        return np.concatenate([arr, elem_exp])


def t_app(tensor, elem, ndim=1):
    elem_ndim = len(elem.shape)
    if ndim == elem_ndim:
        elem_exp = elem
    else:
        final_dim = (None,) * (ndim - elem_ndim) + (...,)
        elem_exp = elem[final_dim]

    if len(tensor) == 0:
        return elem_exp.clone()
    else:
        return torch.cat([tensor, elem_exp])


def inverse_argsort(array_like):
    idx = np.arange(len(array_like))
    inverse = np.arange(len(array_like))

    seq_argsorted = np.argsort(array_like)

    inverse[seq_argsorted[idx]] = idx

    return inverse


def shuffle_with_probablity(labels, prob_new, seed):
    rnd = sklearn.utils.check_random_state(seed)


    orig_state = random.getstate()

    random.seed(rnd.randint(2**32 -1))

    uniq_lab = np.unique(labels)

    l_s = rnd.permutation(np.array([set(np.where(labels == l)[0]) for l in uniq_lab]))

    s_old = set()

    final_order = np.tile(-1, len(labels))
    get_new = rnd.uniform(size=len(labels)) < prob_new

    l_s_ind = 0

    for itx in range(len(labels)):

        if (get_new[itx] and l_s_ind < l_s.shape[0]) or len(s_old) == 0:
                new_set = l_s[l_s_ind]

                new_val = random.sample(new_set, 1)[0]
                new_set.remove(new_val)
                s_old.update(new_set)

                l_s_ind += 1

        else:
                new_val = random.sample(s_old, 1)[0]
                s_old.remove(new_val)

        final_order[itx] = new_val


    assert (final_order >= 0).all()

    random.setstate(orig_state)

    return final_order


def epoch_seed(seed, epoch):
    if seed is None:
        return seed
    rnd = sklearn.utils.check_random_state(seed)
    return rnd.randint(2**32, size=(epoch,))[-1]


def default_notimplemented(*args):
    raise NotImplementedError()


def default_ignore(*args):
    pass


def reduce_packed_array(target, indices):
    res = np.zeros(len(indices), dtype=indices.dtype)
    res[0] = target[:indices[0]].argmin()
    for i in range(1, len(indices)):
        res[i] = target[indices[i - 1]:indices[i]].argmin()
    return res
