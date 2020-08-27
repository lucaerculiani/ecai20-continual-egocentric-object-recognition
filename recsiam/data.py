"""
module containing utilities to load
the dataset for the training
of the siamese recurrent network.
"""
import json
import copy
import functools
import lz4.frame

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Subset
from skimage import io
import torch

import recsiam.utils as utils

# entry
# { "id" :  int,
#   "paths" : str,
#   "metadata" : str}


def descriptor_from_filesystem(root_path):
    desc = []
    root_path = Path(root_path)

    def nextid():
        cid = 0
        while True:
            yield cid
            cid += 1

    id_gen = nextid()

    embedded = False
    sample = next(root_path.glob("*/*/*")).name
    if sample.endswith("npy") or sample.endswith("lz4"):
        embedded = True

    for subd in sorted(root_path.iterdir()):

        obj_desc = {"id":  next(id_gen), "name": str(subd)}

        if not embedded:
            seqs = sorted(str(subsub) for subsub in subd.iterdir()if subsub.is_dir())
        else:
            seqs = sorted(str(subsub / sample) for subsub in subd.iterdir()if subsub.is_dir())

        obj_desc["paths"] = seqs

        desc.append(obj_desc)


    return desc


class VideoDataSet(Dataset):
    """
    Class that implements the pythorch Dataset
    of sequences of frames
    """
    def __init__(self, descriptor):

        self.descriptor = descriptor
        if isinstance(self.descriptor, (list, tuple, np.ndarray)):
            self.data = np.asarray(self.descriptor)
        else:
            with Path(self.descriptor).open("r") as ifile:
                self.data = np.array(json.load(ifile))

        self.paths = np.array([d["paths"] for d in self.data])
        self.seq_number = np.array([len(path) for path in self.paths])

        def get_id_entry(elem_id):
            return self.data[elem_id]["id"]

        self.id_table = np.vectorize(get_id_entry)

        self.embedded = False
        self.compressed = False
        try:
            np.load(self.paths[0][0])
            self.embedded = True
        except Exception:
            pass

        try:
            with lz4.frame.open(self.paths[0][0], mode="rb") as f:
                np.load(f)
                self.embedded = True
                self.compressed = True
        except Exception:
            pass

        self.n_elems = len(self.paths)

    @property
    def is_embed(self):
        return self.embedded

    def load_array(self, path):
        if not self.compressed:
            loaded = np.load(str(path))

        else:
            with lz4.frame.open(str(path), mode="rb") as f:
                loaded = np.load(f)

        return loaded

    def __len__(self):
        return self.n_elems

    def __getitem__(self, value):
        return self._getitem(value)

    def _getitem(self, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            if self._valid_t(value):
                return self._get_single_item(*value)
            elif np.all([self._valid_t(val) for val in value]):
                return np.array([self._get_single_item(*val) for val in value])
            else:
                raise TypeError("Invalid argument type: {}.".format(value))
        else:
            raise TypeError("Invalid argument type: {}.".format(value))

    @staticmethod
    def _valid_t(value):
        return isinstance(value, (tuple, list, np.ndarray)) and \
                len(value) == 3 and \
                isinstance(value[0], (int, np.integer)) and \
                isinstance(value[1], (int, np.integer)) and \
                isinstance(value[2], (int, np.integer,
                                      slice, list, np.ndarray))

    def sample_size(self):
        return self._get_single_item(0, 0, slice(0, 1)).shape[1:]

    def _get_single_item(self, idx1, idx2, idx3):

        path = self.paths[idx1]
        seq_path = path[idx2]
        if not self.is_embed:
            p_list = np.array(sorted(Path(seq_path).iterdir()))[idx3]
        else:
            p_list = seq_path, idx3

        sequences = self._load_sequence(p_list)

        return sequences

    def _load_sequence(self, paths_list):

        if not self.is_embed:
            sequence = np.array(io.imread_collection(paths_list,
                                                     conserve_memory=False))
            sequence = np.transpose(sequence, (0, 3, 1, 2))

        else:
            sequence = self.load_array(paths_list[0])[paths_list[1]]

        return sequence

    def gen_embed_dataset(self):
        for obj in range(self.n_elems):
            for seq in range(len(self.paths[obj])):
                yield self[obj, seq, :], self.paths[obj][seq]


def dataset_from_filesystem(root_path):
    descriptor = descriptor_from_filesystem(root_path)
    return VideoDataSet(descriptor)


class TrainSeqDataSet(VideoDataSet):

    def __getitem__(self, value):
        if isinstance(value, (list, tuple, np.ndarray)) and \
           len(value) == 2 and \
           np.all([self._valid_t(val) for val in value]):

            items = self._getitem(value)
            seq_len = np.array([len(val)
                                for val in items])

            return items, seq_len, (value[0][0], value[1][0])
        else:
            error_str = "The input must be in the form " +\
                        "((int, int, slice), (int, int,  slice)). " +\
                        "Found {}"

            raise ValueError(error_str.format(value))


class FlattenedDataSet(VideoDataSet):

    def __init__(self, *args, preload=True, pre_embed=None):
        super().__init__(*args)

        self.val_map = []
        for itx in range(len(self.seq_number)):
            self.val_map.extend([(itx, i) for i in range(self.seq_number[itx])])

        self.val_map = np.array(self.val_map)

        self.flen = len(self.val_map)
        self.pre_embed = pre_embed

        self.preloaded = None
        if preload:
            self.preloaded = []
            for i in range(len(self)):
                self.preloaded.append(self.real_getitem(i))

    def map_value(self, value):
        return self.val_map[value]

    def __len__(self):
        return self.flen

    def get_label(self, value):
        ndim = np.ndim(value)
        if ndim == 0:
            return self.map_value(value)[0]
        elif ndim == 1:
            return self.map_value(value)[:, 0]
        else:
            raise ValueError("np.ndim(value) > 1")

    def __getitem__(self, i):
        if self.preloaded is not None:
            return self.preloaded[i]
        else:
            return self.real_getitem(i)

    def real_getitem(self, value):
        t = tuple(self.map_value(value)) + (slice(None),)
        items = super().__getitem__(t)
        if self.pre_embed is not None:
            items = self.pre_embed([utils.a2t(items)])[0]
        return items, t[0]

    def balanced_sample(self, elem_per_class, rnd):
        p_ind = rnd.permutation(len(self.val_map))
        perm = self.val_map[p_ind]
        cls = perm[:, 0]

        _, indices = np.unique(cls, return_index=True)

        remaining_ind = np.delete(np.arange(len(cls)), indices)

        for i in range(elem_per_class - 1):
            p = cls[remaining_ind]
            _, ind = np.unique(p, return_index=True)

            indices = np.concatenate([indices, remaining_ind[ind]])
            remaining_ind = np.delete(remaining_ind, ind)

        return p_ind[indices]

    def get_n_objects(self, number, rnd):
        obj_ind = rnd.choice(len(self.seq_number), size=number, replace=False)
        return np.where(np.isin(self.val_map[:, 0], obj_ind))[0]


def list_collate(data):
    emb = [utils.astensor(d[0]) for d in data]
    lab = np.array([d[1] for d in data])

    return emb, lab


class ExtendedSubset(Subset):

    def get_label(self, value):
        return self.dataset.get_label(self.indices[value])


def train_val_split(dataset, seed, dl_arg={},
                    incremental_evaluation=None, prob_new=None):
    rs = np.random.RandomState
    rnd_s, rnd_e, rnd_i = [rs(s) for s in rs(seed).randint(2**32 - 1, size=3)]
    val_ind = dataset.balanced_sample(1, rnd_e)

    if incremental_evaluation is not None:
        ie_ind = dataset.get_n_objects(incremental_evaluation["number"], rnd_i)
        val_ind = np.setdiff1d(val_ind, ie_ind)
        del_ind = np.concatenate((val_ind, ie_ind))

        if incremental_evaluation["setting"] is None:
            ie_ind = rnd_i.permutation(ie_ind)
        else:
            ie_ind = utils.shuffle_with_probablity(
                                        dataset.get_label(ie_ind),
                                        incremental_evaluation["setting"],
                                        rnd_i)

        ie_ds = ExtendedSubset(dataset, ie_ind)
        ie_dl = torch.utils.data.DataLoader(ie_ds, shuffle=False, collate_fn=list_collate, **dl_arg)

    else:
        del_ind = val_ind
        ie_dl = None

    val_ds = ExtendedSubset(dataset, val_ind)

    ordered_train_ind = np.delete(np.arange(len(dataset)), del_ind)

    if prob_new is None:
        train_ind = rnd_s.permutation(ordered_train_ind)
    else:
        new_order = utils.shuffle_with_probablity(
                                        dataset.get_label(ordered_train_ind),
                                        prob_new,
                                        rnd_s)
        train_ind = ordered_train_ind[new_order]

    train_ds = ExtendedSubset(dataset, train_ind)

    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, collate_fn=list_collate, **dl_arg)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, collate_fn=list_collate, **dl_arg)

    return train_dl, val_dl, ie_dl


def train_test_shuf(train_ds, test_ds, seed, dl_arg={}, prob_new=None):
    rnd = np.random.RandomState(seed)
    ordered_train_ind = np.arange(len(train_ds))

    if prob_new is None:
        train_ind = rnd.permutation(ordered_train_ind)
    else:
        new_order = utils.shuffle_with_probablity(
                                        dataset.get_label(ordered_train_ind),
                                        prob_new,
                                        rnd)
        train_ind = ordered_train_ind[new_order]

    shuf_train_ds = ExtendedSubset(train_ds, train_ind)

    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False,collate_fn=list_collate, **dl_arg)
    train_dl = torch.utils.data.DataLoader(shuf_train_ds, shuffle=False,collate_fn=list_collate, **dl_arg)

    return train_dl, test_dl


def train_test_desc_split(descriptor, test_seed):
    rnd = np.random.RandomState(test_seed)

    test_desc = copy.deepcopy(descriptor)
    train_desc = copy.deepcopy(descriptor)

    for test, train in zip(test_desc, train_desc):
        ind = rnd.randint(len(train["paths"]))
        test["paths"] = [train["paths"][ind]]
        del train["paths"][ind]

    return train_desc, test_desc


def train_test_factory(descriptor, test_seed, dl_arg, prob_new=None, pre_embed=None):

    train_desc, test_desc = train_test_desc_split(descriptor, test_seed)

    test_ds = FlattenedDataSet(test_desc, pre_embed=pre_embed)
    train_ds = FlattenedDataSet(train_desc, pre_embed=pre_embed)

    return functools.partial(train_test_shuf, train_ds, test_ds, dl_arg=dl_arg, prob_new=prob_new)


def train_val_factory(descriptor, test_seed, dl_arg,
                      remove_test=True, incremental_evaluation=None,
                      prob_new=None, pre_embed=None):
    if remove_test:
        train_desc, test_desc = train_test_desc_split(descriptor, test_seed)
    else:
        train_desc = descriptor

    train_ds = FlattenedDataSet(train_desc, pre_embed=pre_embed)

    return functools.partial(train_val_split, train_ds,
                             dl_arg=dl_arg, incremental_evaluation=incremental_evaluation,
                             prob_new=prob_new)
