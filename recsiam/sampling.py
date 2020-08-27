"""
module containing utilities to
sample the dataset for the training
of the siamese recurrent network.
"""
import torch
import numpy as np
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler


class SeqSampler(Sampler):

    def __init__(self,
                 dataset,
                 true_frac=0.5,
                 base_seed=None,
                 start_epoch=1):

        self.dataset = dataset
        self.true_frac = true_frac

        self.start_epoch = start_epoch
        self.current_epoch = self.start_epoch

        self.base_seed = base_seed
        if self.base_seed is None:
            # explicitly draw a random seed
            self.base_seed = np.random.randint(0, 2**32 - 1)

        self.true_num, self.false_num = self.get_pos_neg()

        self.size = self.true_num + self.false_num

    def __len__(self):
        return self.size

    def __iter__(self):
        epoch = self.current_epoch
        self.current_epoch += 1
        return (self.format_sample(spl) for spl in self.generate_samples(epoch))

    def get_pos_neg(self):
        return self.compute_pos_neg_number(len(self.dataset), self.true_frac)

    def format_sample(self, sample):
        return sample[0:2]

    def generate_samples(self, epoch):

        rnd = self.get_rnd_for_epoch(epoch)

        d_idx = np.arange(len(self.dataset))
        rnd.shuffle(d_idx)
        index_iter = iter(d_idx)

        labels = np.concatenate((np.ones(self.true_num),
                                 np.zeros(self.false_num))).astype(np.short)
        rnd.shuffle(labels)

        for lab in labels:
            yield self.make_sample(lab, index_iter, rnd)

    def make_sample(self, label, d_iter, rnd):

        elem_one = next(d_iter)
        if label == 0:
            elem_two = next(d_iter)

            seqs = np.array((rnd.choice(self.dataset.seq_number[elem_one]),
                             rnd.choice(self.dataset.seq_number[elem_two])))
        else:
            elem_two = elem_one
            seqs = rnd.choice(self.dataset.seq_number[elem_one], size=2, replace=False)

        return ((elem_one, seqs[0], slice(None)),
                (elem_two, seqs[1], slice(None)),
                label)

    def get_rnd_for_epoch(self, epoch):
        base_rnd = np.random.RandomState(self.base_seed)

        seeds = base_rnd.randint(0, 2**32 - 1, size=epoch)

        return np.random.RandomState(seeds[-1])

    def get_next_rnd(self):

        rnd = self.get_rnd_for_epoch(self.current_epoch)

        self.current_epoch += 1

        return rnd

    @staticmethod
    def compute_pos_neg_number(d_size, true_frac):
        true_num = int(np.floor((d_size * true_frac)))
        false_num = int(np.floor((d_size - true_num) / 2.))
        return true_num, false_num

    @staticmethod
    def get_max_size(d_size, true_frac):
        return sum(SeqSampler.compute_pos_neg_number(d_size, true_frac))


class RepeatingSeqSampler(SeqSampler):
    """
    Behaves like SeqSampler, but returns alaways
    an iterators that returns the same values.
    To be used for evaluation/validation
    """

    def __init__(self,
                 dataset,
                 true_frac=0.5,
                 base_seed=None):

        super(RepeatingSeqSampler, self).__init__(dataset,
                                                  true_frac=true_frac,
                                                  base_seed=base_seed)

    def __iter__(self):
        epoch = self.current_epoch
        # skip current_epoch increment
        return (self.format_sample(spl) for spl in self.generate_samples(epoch))




class UniversalBatchSampler(BatchSampler):
    r"""Subclass of torchutils.data.sampler.Batchsampler
    Wraps another sampler to yield a mini-batch of indices.

    This is almost the same of the original, except that allows for
    indices with a type different from int

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)   # the original was casting idx to int
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def collate_train_dataset(raw_batch):
    r_it = range(len(raw_batch))
    batch_1 = [raw_batch[i][0][0] for i in r_it]
    batch_2 = [raw_batch[i][0][1] for i in r_it]

    lengths_1 = np.array([raw_batch[i][1][0] for i in r_it])
    lengths_2 = np.array([raw_batch[i][1][1] for i in r_it])

    labels = np.array([raw_batch[i][2] for i in r_it])

    return (batch_1, lengths_1), (batch_2, lengths_2), labels


def sort_batch_sequences(batch):
    seq_argsorted = np.argsort(- batch[1])
    old_order = np.arange(len(seq_argsorted))

    idx = np.arange(len(seq_argsorted))
    old_order[seq_argsorted[idx]] = idx

    if type(batch[0]) == type([]):
        return ([torch.from_numpy(batch[0][seq]) for seq in seq_argsorted],
                torch.from_numpy(batch[1][seq_argsorted, ...]),
                torch.from_numpy(old_order))


    return (torch.from_numpy(batch[0][seq_argsorted, ...]),
            torch.from_numpy(batch[1][seq_argsorted, ...]),
            torch.from_numpy(old_order))


def collate_and_sort(raw_batch):
    collated = collate_train_dataset(raw_batch)
    sorted_0 = sort_batch_sequences(collated[0])
    sorted_1 = sort_batch_sequences(collated[1])

    return sorted_0, sorted_1, torch.from_numpy(collated[2].astype(int))



class SeadableRandomSampler(RandomSampler):

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rnd = np.random.RandomState(seed)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(self.rnd.randint(high=n, size=(self.num_samples,), dtype=int).tolist())
        return iter(rnd.permutation(n).tolist())

    def __len__(self):
        return len(self.data_source)

