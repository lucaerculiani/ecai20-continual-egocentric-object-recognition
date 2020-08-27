"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""

import unittest
import numpy as np

import recsiam.closedworld as ev
import recsiam.utils as utils
import torch




class PairwiseEvaluator(unittest.TestCase):
    """Unit tests for class evaluation.PaiwiseEvaluator"""

    def test_all(self):


        p0 = np.array([2, 1,4,0,3 ])
        emb0 = [ utils.a2t(np.tile(np.arange(100, step=10), (3, 1)) + 5*i).float() for i in range(5)]
        emb0 = np.array(emb0, dtype="object")[p0]
        cl0 = (np.arange(5)*2 +15) [p0]


        p1 = np.array([1,4,2,3,0])
        emb1 = [ utils.a2t(np.tile(np.arange(100, step=10), (3, 1)) + 5*i).float() for i in range(5)]
        emb1 = np.array(emb1, dtype="object")[p1]
        cl1 = (np.arange(5)*2 + 15)[p1]

        pairev = ev.PairwiseEvaluator(emb0, cl0, emb1, cl1)

        min_seq_cg = pairev.compute_min_seq()
        
        self.assertTrue(np.all(min_seq_cg == np.tile(5,5)))
        

        min_frames_cg = pairev.compute_min_frames()
        self.assertTrue(np.all(min_frames_cg == np.tile(15,15)))


class TestCumulativeGain(unittest.TestCase):
    """Unit tests for cumulative gain functions in data module"""

    def test_cumulative_gain(self):
        examples = 5
        relevance = np.identity(examples)

        cgain = ev.cumulative_gain(relevance)

        self.assertEqual(cgain.shape, (examples,))
        self.assertTrue((cgain == np.arange(1, examples + 1)).all())

        cgain_at_1 = ev.cumulative_gain(relevance, 1)
        cgain_at_max = ev.cumulative_gain(relevance, examples)

        self.assertEqual(cgain_at_1, 1)
        self.assertEqual(cgain_at_max, examples)

        cg_uni = (ev.cumulative_gain(relevance[0]) == np.ones(examples)).all()
        self.assertTrue(cg_uni)
        self.assertEqual(ev.cumulative_gain(relevance[0], upto=1), 1)
        self.assertEqual(ev.cumulative_gain(relevance[0], upto=examples), 1)

    def test_dcg(self):

        relevance = np.array([1.0, 0., 0.])

        dcg = ev.discontinued_cumulative_gain(relevance)
        self.assertTrue((dcg == np.ones(3)).all())

        examples = 5
        relevances = np.identity(examples)
        dcgs = np.array([ev.discontinued_cumulative_gain(relevances[itx],
                                                         upto=examples)
                         for itx in range(examples)])

        descending = - np.sort(- dcgs, axis=-1)
        self.assertTrue((dcgs == descending).all())

    def test_ndcg(self):
        examples = 5
        relevances = - np.sort(- np.random.uniform(size=examples))

        ndcg = ev.normalized_discontinued_cumulative_gain(relevances)

        self.assertTrue((ndcg == np.ones(examples)).all())





