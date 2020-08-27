"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""

import unittest
import numpy as np
import numpy.testing as npt

import recsiam.models as models
import recsiam.utils as utils
import torch


class TestREdux(unittest.TestCase):
    """Unit tests for class evaluation.PaiwiseEvaluator"""
    
    def test_globalmean(self):
        bogusdata = [torch.ones(5, 3), torch.zeros(6, 3)]

        gm = models.GlobalMean()

        fded = gm.forward(bogusdata)

        fded = np.array([utils.t2a(f.squeeze(dim=0)) for f in fded])

        npt.assert_equal(fded, np.array([np.ones(3), np.zeros(3)]))

        fded = gm.forward(bogusdata[:1])

        fded = utils.t2a(fded[0].squeeze(dim=0))

        npt.assert_equal(fded, np.ones(3))

    def test_recursivereduction(self):
        bogusdata = [torch.ones(50, 3), torch.zeros(60, 3)]

        rr = models.RecursiveReduction(3)

        fded = rr.forward(bogusdata)

        fded_a = np.concatenate([utils.t2a(f) for f in fded])

        self.assertEqual(fded_a.shape, (2, 3))

