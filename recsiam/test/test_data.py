"""
HH this is 4 U
Unit test module for function and classes defined in  recsiam.data
"""

import unittest
import pathlib

import numpy as np
import numpy.testing as npt

import recsiam.data as data

TEST_DATASET_PATH = pathlib.Path(__file__).parent.parent / "testdata" / "dataset"

TEST_DATASET = data.descriptor_from_filesystem(TEST_DATASET_PATH)

TEST_DATASET_LEN = len(TEST_DATASET)


class TestVideoDataSet(unittest.TestCase):
    """
    Test class for data.VideoDataSet
    """

    def test_sample_size(self):
        new_dataset = data.VideoDataSet(TEST_DATASET)
        s_size = new_dataset.sample_size()

        self.assertEqual((3, 120, 120), s_size)

    def test_len(self):
        dataset = data.VideoDataSet(TEST_DATASET)
        self.assertEqual(len(dataset), TEST_DATASET_LEN)

    def test_getitem(self):
        dataset = data.VideoDataSet(TEST_DATASET)
        seq_shape = (5, 3, 120, 120)

        seq = dataset[(0, 0, slice(0, 5))]
        self.assertEqual(seq.shape, seq_shape)

        seq = dataset[(0, 0, slice(0, 10, 2))]
        self.assertEqual(seq.shape, (5,) + seq_shape[1:])

        seq = dataset[(0, 0, np.arange(5).tolist())]
        self.assertEqual(seq.shape, seq_shape)

        seq = dataset[((0, 0, np.arange(5)), (0, 0, np.arange(5)))]
        self.assertEqual(seq.shape, (2,) + seq_shape)

        seq = dataset[((0, 0, slice(0, 5)), (0, 0, slice(0, 5)))]
        self.assertEqual(seq.shape, (2,) + seq_shape)

        with self.assertRaises(TypeError):
            dataset[(5, 2)]

        with self.assertRaises(TypeError):
            dataset[(1, slice(None), 2)]

        with self.assertRaises(TypeError):
            dataset[1]

        with self.assertRaises(TypeError):
            dataset[slice(None)]

        with self.assertRaises(TypeError):
            dataset[(1, 1)]


class TrainSeqDataSet(unittest.TestCase):
    """
    Test class for data.TrainSeqDataSet
    """

    def test_getitem(self):

        dataset = data.TrainSeqDataSet(TEST_DATASET)
        value = ((0, 0, slice(0, 20)), (0, 2, slice(10, 30)))

        res = dataset[value]

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 20, 3, 120, 120))
        self.assertTrue(np.array_equal(res[1], np.array((20, 20))))
        self.assertEqual(res[2][0], res[2][1])

        value = ((0, 0, slice(0, 20)), (1, 0, slice(10, 30)))
        res = dataset[value]

        self.assertNotEqual(res[2][0], res[2][1])

        with self.assertRaises(ValueError):
            dataset[(0, np.arange(90))]

        with self.assertRaises(ValueError):
            dataset[0]

        with self.assertRaises(ValueError):
            wrong_value = ((1, slice(0, 20)), (0, slice(20, 40)))
            dataset[wrong_value + wrong_value]


class TrainFlattenedDataSet(unittest.TestCase):
    """
    Test class for data.FlattenedDataSet
    """

    def test_getitem(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        seq_shape = (30, 3, 120, 120)

        seq = dataset[0]
        self.assertEqual(seq[0].shape, seq_shape)

        seq2 = dataset[15]
        self.assertNotEqual(seq[1],  seq2[1])

    def test_get_label(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)

        seq = dataset[0]

        self.assertEqual(seq[1], dataset.get_label(0))

    def test_len(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        self.assertEqual(len(dataset), TEST_DATASET_LEN * 3)

    def test_balanced_sample(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)

        rnd = np.random.RandomState(0)
        ind = dataset.balanced_sample(1, rnd)

        lab = np.array([dataset.get_label(i) for i in ind])

        self.assertEqual(lab.shape[0], TEST_DATASET_LEN)
        self.assertEqual(np.unique(lab).shape[0], TEST_DATASET_LEN)

        rnd = np.random.RandomState(0)
        ind2 = dataset.balanced_sample(1, rnd)

        npt.assert_equal(ind, ind2)

        def check_sampling(num):
            rnd = np.random.RandomState(num)
            ind = dataset.balanced_sample(num, rnd)
            lab = np.array([dataset.get_label(i) for i in ind])

            self.assertEqual(lab.shape[0], TEST_DATASET_LEN * num)
            self.assertEqual(np.unique(ind).shape[0], TEST_DATASET_LEN * num)

            uniq, cnt = np.unique(lab, return_counts=True)

            self.assertEqual(uniq.shape[0], TEST_DATASET_LEN)
            npt.assert_equal(cnt, np.tile(num, len(cnt)))

        check_sampling(2)
        check_sampling(3)


class TestSplitFunctions(unittest.TestCase):

    def test_train_val(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        train_dl, val_dl = data.train_val_split(dataset, 0)

        indices = np.concatenate((train_dl.dataset.indices, val_dl.dataset.indices))

        self.assertEqual(val_dl.dataset.indices.shape[0], TEST_DATASET_LEN)
        self.assertEqual(indices.shape[0], TEST_DATASET_LEN * 3)
        self.assertEqual(indices.shape[0], np.unique(indices).shape[0])

        lab = np.array([val_dl.dataset.dataset.get_label(i) for i in indices])
        self.assertEqual(lab.shape[0], TEST_DATASET_LEN * 3)
        self.assertEqual(np.unique(lab).shape[0], TEST_DATASET_LEN)

    def test_train_test_desc_split(self):
        tr, te = data.train_test_desc_split(TEST_DATASET, 0)

        tr_paths = set(np.concatenate([p["paths"] for p in tr]))
        te_paths = set(np.concatenate([p["paths"] for p in te]))

        self.assertEqual(len(tr_paths | te_paths), TEST_DATASET_LEN * 3)
        self.assertEqual(len(tr_paths & te_paths), 0)

        train_d = data.FlattenedDataSet(tr)
        test_d = data.FlattenedDataSet(te)

        self.assertEqual(len(train_d), TEST_DATASET_LEN * 2)
        self.assertEqual(len(test_d), TEST_DATASET_LEN)
