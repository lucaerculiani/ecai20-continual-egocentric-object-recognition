"""
HH this is 4 U
Unit test module for function and classes defined in  recsiam.utils
"""

import unittest

import numpy as np
import numpy.testing as npt

import torch

import recsiam.utils as utils



def t_assert_equal(a, b):
    npt.assert_equal(utils.t2a(a), utils.t2a(b))


class TestUtils(unittest.TestCase):
    """Unit tests for class samp.SeqSampler"""

    def test_shuffle_with_probablity(self):

        lab = np.tile(np.arange(3),3)

        order = utils.shuffle_with_probablity(lab, 1.0, 1)

        ordered = lab[order]
    
        self.assertEqual(np.unique(ordered[:3]).shape[0], 3)
        self.assertEqual(np.unique(order).shape[0], lab.shape[0])
        self.assertTrue((order == utils.shuffle_with_probablity(lab, 1.0, 1)).all())
        
        

        order = utils.shuffle_with_probablity(lab, 0.0, 1)

        ordered = lab[order]
    
        for i in range(3):
            with self.subTest(i=i):
                self.assertEqual(np.unique(ordered[i*3:(i+1)*3]).shape[0], 1)

        self.assertEqual(np.unique(order).shape[0], lab.shape[0])
        self.assertTrue((order == utils.shuffle_with_probablity(lab, 0.0, 1)).all())



        order = utils.shuffle_with_probablity(lab, 0.5, 1)
        self.assertTrue((order == utils.shuffle_with_probablity(lab, 0.5, 1)).all())

    def test_a_app(self):

        a = np.array(1.0)
        app = np.empty(0)
        app = utils.a_app(app, a)
        npt.assert_equal(app, a[None, ...])

        a = np.ones(10)
        app = np.empty(0)
        app = utils.a_app(app, a)
        npt.assert_equal(app, a)

        app = utils.a_app(app, a)
        npt.assert_equal(app, np.ones(20))

        a = np.ones((1,10))
        app = np.empty(0)
        app = utils.a_app(app, a, ndim=2)
        npt.assert_equal(app, a)

        app = utils.a_app(app, a, ndim=2)
        npt.assert_equal(app, np.ones((2, 10)))


    def test_t_app(self):

        a = torch.tensor(1.0)
        app = torch.empty(0)
        app = utils.t_app(app, a)
        t_assert_equal(app, a[None, ...])

        a = torch.ones(10)
        app = torch.empty(0)
        app = utils.t_app(app, a)
        t_assert_equal(app, a)

        app = utils.t_app(app, a)
        t_assert_equal(app, torch.ones(20))

        a = torch.ones((1, 10))
        app = torch.empty(0)
        app = utils.t_app(app, a, ndim=2)
        t_assert_equal(app, a)

        app = utils.t_app(app, a, ndim=2)
        t_assert_equal(app, torch.ones((2, 10)))

    def test_reduce_packed_array(self):
        target = np.arange(20)
        indices = np.array([4, 6, 10]).cumsum()

        expected = np.array([0, 0, 0])

        npt.assert_equal(utils.reduce_packed_array(target, indices), expected)

        target = target[::-1]
        expected = np.array([3, 5, 9])

        npt.assert_equal(utils.reduce_packed_array(target, indices), expected)
