"""
HH this is 4 U
Unit test module for function and classes defined in  recsiam.data
"""

import unittest
import pathlib

import numpy as np

import recsiam.data as data
import recsiam.sampling as samp


TEST_DATASET_PATH = pathlib.Path(__file__).parent.parent / "testdata" / "dataset"

TEST_DATASET = data.descriptor_from_filesystem(TEST_DATASET_PATH)

TEST_DATASET_LEN = len(TEST_DATASET)


class TestSeqSampler(unittest.TestCase):
    """Unit tests for class samp.SeqSampler"""

    def test_size(self):

        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15

        sampler = samp.SeqSampler(dataset, true_frac=0.0)
        self.assertTrue(len(sampler), int(np.floor(len(dataset) / 2)))

        sampler = samp.SeqSampler(dataset, true_frac=1.0)
        self.assertTrue(len(sampler), len(dataset))
        epoch = list(sample for sample in iter(sampler))
        for itx in range(len(epoch)):
            with self.subTest(i=itx):
                self.assertEqual(epoch[itx][0][0], epoch[itx][1][0])

        sampler = samp.SeqSampler(dataset, true_frac=0.0)
        self.assertTrue(len(sampler), int(np.floor(len(dataset))))

        epoch = list(sample for sample in iter(sampler))
        for itx in range(len(epoch)):
            with self.subTest(i=itx):
                self.assertNotEqual(epoch[itx][0][0], epoch[itx][1][0])

        self.assertEqual(samp.SeqSampler.get_max_size(10, 1.0), 10)
        self.assertEqual(samp.SeqSampler.get_max_size(10, 0.0), 5)

    def test_make_sample(self):
        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15

        sampler = samp.SeqSampler(dataset, true_frac=0.5)

        d_iter = (i for i in range(len(dataset)))

        rnd = np.random.RandomState(0)
#       checking correct (two different seq from the same obj) sample
        sample = sampler.make_sample(1, d_iter, rnd)

        self.assertTrue(dataset._valid_t(sample[0]))
        self.assertTrue(dataset._valid_t(sample[1]))
        self.assertEqual(sample[0][0], sample[1][0])
        self.assertNotEqual(sample[0][1], sample[1][1])

#       checking wrong sample
        sample = sampler.make_sample(0, d_iter, rnd)

        self.assertTrue(dataset._valid_t(sample[0]))
        self.assertTrue(dataset._valid_t(sample[1]))
        self.assertNotEqual(sample[0][0], sample[1][0])

    def test_iter(self):
        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15

        sampler = samp.SeqSampler(dataset, true_frac=0.5)

        sample_it = iter(sampler)

        for sample in sample_it:
            self.assertTrue(dataset._valid_t(sample[0]))
            self.assertTrue(dataset._valid_t(sample[1]))

    def test_seed(self):

        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15

        def get_sampler(seed):
            return samp.SeqSampler(dataset, true_frac=0.5, base_seed=seed)

        base = 0
        sampler = get_sampler(base)

        epoch_one = list(sample for sample in iter(sampler))

        sampler = get_sampler(base)

        epoch_two = list(sample for sample in iter(sampler))

        self.assertEqual(epoch_one, epoch_two)

        new_sampler = get_sampler(None)
        self.assertNotEqual(new_sampler.base_seed, None)

        # covering random edge case with probability 2**-32. lel
        while new_sampler.base_seed == 0:
            new_sampler = get_sampler(None)

        new_epoch_one = list(sample for sample in iter(new_sampler))

        self.assertNotEqual(epoch_one, new_epoch_one)

    def test_restore(self):
        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15
        base = 0

        from_one = samp.SeqSampler(dataset, start_epoch=1, base_seed=base)

        from_one_epochs = [list(sample for sample in iter(from_one))
                           for i in range(1, 11)]

        from_five = samp.SeqSampler(dataset, start_epoch=6, base_seed=base)
        from_five_epochs = [list(sample for sample in iter(from_five))
                            for i in range(6, 11)]

        self.assertEqual(from_one_epochs[5:], from_five_epochs)

        new_base = base + 1

        new_from_five = samp.SeqSampler(dataset,
                                        start_epoch=6,
                                        base_seed=new_base)
        new_from_five_epochs = [list(sample for sample in iter(new_from_five))
                                for i in range(6, 11)]

        self.assertNotEqual(new_from_five_epochs, from_five_epochs)




class TestRepeatingSeqSampler(unittest.TestCase):
    """Unit tests for class samp.TestRepeatingSeqSampler"""

    def test_repetition(self):
        dataset = data.TrainSeqDataSet(TEST_DATASET)
        dataset.pad = 15
        base = 0

        sampler = samp.RepeatingSeqSampler(dataset, base_seed=base)

        sampled_list = [list(sample for sample in iter(sampler))
                        for i in range(1, 11)]

        for itx in range(len(sampled_list) - 1):
            with self.subTest(i=itx):
                self.assertEqual(*sampled_list[itx:itx+2])

        new_sampler = samp.RepeatingSeqSampler(dataset, base_seed=base)
        new_sampled = list(sample for sample in iter(new_sampler))

        self.assertEqual(new_sampled, sampled_list[0])

        new_base = base + 1
        diff_sampler = samp.RepeatingSeqSampler(dataset, base_seed=new_base)
        diff_sampled = list(sample for sample in iter(diff_sampler))

        self.assertNotEqual(new_sampled, diff_sampled)




class TestsDataFunctions(unittest.TestCase):
    """ Testcases for functions in data module"""
    def test_collate_train_dataset(self):

        inp = [(np.array((np.tile(i, (5, 5, 5)), np.tile(j, (5, 5, 5)))),
                np.array([i, j]),
                1)
               for i, j in zip(range(3), range(3, 6))]

        b1 = np.array([np.tile(i, (5, 5, 5)) for i in range(3)])
        b2 = np.array([np.tile(i, (5, 5, 5)) for i in range(3, 6)])

        l1 = np.array(list(range(3)))
        l2 = np.array(list(range(3, 6)))

        lab = np.tile(1, 3)

        res = samp.collate_train_dataset(inp)

        (batch_1, lengths_1), (batch_2, lengths_2), labels = res

        self.assertTrue(np.array_equal(b1, batch_1))
        self.assertTrue(np.array_equal(b2, batch_2))
        self.assertTrue(np.array_equal(l1, lengths_1))
        self.assertTrue(np.array_equal(l2, lengths_2))
        self.assertTrue(np.array_equal(lab, labels))

    def test_sort_batch_sequences(self):
        b1 = np.array([np.tile(i, (5, 5, 5)) for i in range(1, 6)])
        l1 = np.array(list(range(1, 6)))

        rnd = np.random.RandomState(0)
        rnd.shuffle(l1)

        res = samp.sort_batch_sequences((b1, l1))

        self.assertTrue(np.array_equal(b1, res[0][res[2], ...]))
        self.assertTrue(np.array_equal(l1, res[1][res[2], ...]))
        self.assertFalse(np.array_equal(np.arange(len(l1)), res[2]))
