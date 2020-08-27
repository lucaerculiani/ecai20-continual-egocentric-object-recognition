"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""

import unittest
import numpy as np
import numpy.testing as npt

import recsiam.openworld as ow
import recsiam.utils as utils
import recsiam.agent as ag
import torch

from .test_agent import simplemodel, simplesupervisor


class DL():

    def __init__(self, data, lab):
        self.d = data
        self.l = lab

    def __iter__(self):
        return (([d],l) for d, l in zip(self.d, self.l))


def df(seed):
    s_data = np.array([
        [[1., 0, 0], [1., 0, 0]],
        [[2., 0, 0], [2., 0, 0]],
        [[5., 0, 0], [5., 0, 0]],
        [[6., 0, 0], [6., 0, 0]],
        [[17., 0, 0], [17., 0, 0]],
        [[18., 0, 0], [18., 0, 0]],
        ])

    s_lab = np.array([0, 0, 1, 1, 2, 2])

    e_data = np.array([
        [[1.5, 0, 0], [1., 0, 0]],
        [[5.5, 0, 0], [5., 0, 0]],
        [[17.5, 0, 0], [17., 0, 0]],
        ])

    e_lab = np.array([0, 1, 2])
    return DL(utils.a2t(s_data).float(), s_lab), DL(utils.a2t(e_data).float(), e_lab)


def sf(dl):
    return simplesupervisor(dl.l)


def af(seed, sup):
    torch.manual_seed(seed)
    agent = ag.Agent(seed, ag.ObjectsMemory(), ag.SupervisionMemory(),
                     simplemodel(), sup, bootstrap=2,
                     max_neigh_check=1,
                     add_seen_element=ag.add_seen_separate)

    return agent


def get_ow(seed=0):
    return ow.OpenWorld(af, df, sf, seed)


class TestOpenWorld(unittest.TestCase):
    """Unit tests for class evaluation.PaiwiseEvaluator"""

    def test_de_experiments(self):
        env = get_ow(0)
        e_n = 3

        generator = env.gen_experiments(e_n)

        s_data_len = 6
        e_data_len = 3

        res_l = []
        for g, itx in zip(generator, range(100)):
            with self.subTest(i=itx):
                res = ow.do_experiment(*g, do_eval=False)
                res_l += [res]

                self.assertEqual(len(res), 2)
                s_d, e_d = res

                for key in s_d:
                    if key is not "cc":
                        self.assertEqual(s_d[key].size, s_data_len)
                #for key in ("pred", "neigh"):
                #    self.assertEqual(e_d[key].shape, (s_data_len, e_data_len))
                #self.assertEqual(e_d["class"].size, e_data_len)

        self.assertEqual(len(res_l), e_n)

    def test_stack_results(self):
        env = get_ow(0)
        e_n = 3

        res_l = [ow.do_experiment(*g, do_eval=False) for g in env.gen_experiments(e_n)]

        stacked = ow.stack_results([r[0] for r in res_l])

        self.assertEqual(set(res_l[0][0].keys()), set(stacked.keys()))
        for key in res_l[0][0]:
            npt.assert_equal(stacked[key][0], res_l[0][0][key])


class TestMetrics(unittest.TestCase):

    def test_new_obj_in_seq(self):
        seq = np.arange(10)

        npt.assert_equal(ow.new_obj_in_seq(seq)[0], np.ones(10, dtype=np.bool))

        seq = np.concatenate([seq, seq])

        expected = np.concatenate([np.ones(10, dtype=np.bool), np.zeros(10, dtype=np.bool)])
        npt.assert_equal(ow.new_obj_in_seq(seq)[0], expected)

        seq = np.array([np.arange(10), np.zeros(10)])
        expected = np.array([np.ones(10, dtype=np.bool), np.zeros(10, dtype=np.bool)])
        expected[1, 0] = True
        npt.assert_equal(ow.new_obj_in_seq(seq), expected)

    def test_known_class_mat(self):
        seq = np.arange(10)
        new_obj = ow.new_obj_in_seq(seq)
        expected = np.identity(10).astype(np.bool)

        npt.assert_equal(ow.known_class_mat(seq, new_obj)[0], expected)

        seq = np.concatenate([seq, seq])
        new_obj = ow.new_obj_in_seq(seq)
        expected = np.zeros((10, 20), dtype=np.bool)
        expected[:, :10] = np.identity(10)

        npt.assert_equal(ow.known_class_mat(seq, new_obj)[0], expected)

    def test_prec_rec_ko(self):
        real_classes = np.array([[0, 1, 2, 0, 1, 2, 2]])
        pred_known = np.array([[0, 0, 1, 1, 1, 0, 0]]).astype(np.bool)
        pred_neigh = np.array([[0, 0, 1, 0, 1, 2, 5]])

        real_prec_ko = np.array([2. / 3])
        real_recall_ko = np.array([2. / 4])

        real_known_obj = ~ ow.new_obj_in_seq(real_classes)
        same_class = ow.is_same_class(pred_neigh, real_classes)

        p, r = ow.prec_rec_ko(real_known_obj, pred_known, same_class)

        npt.assert_allclose(p, real_prec_ko)
        npt.assert_allclose(r, real_recall_ko)

    def test_eval_one_seen_unseen_acc(self):
        sess_classes = np.array([[0, 1, 2, 0, 1,]])
        eval_classes = np.arange(3)[None, ...]

        eval_known = np.concatenate([np.zeros((3,3)), np.ones((2,3))]).astype(np.bool)[None, ...]
        eval_neigh = np.concatenate([np.zeros((3,3)), np.tile(np.arange(3), (2, 1))])[None, ...]
        eval_neigh = eval_neigh.astype(np.int)

        e_d = {"class": eval_classes, "neigh": eval_neigh, "pred": eval_known}
        s_d = {"class": sess_classes}

        real_prec_ko = np.array([0., 0., 0., 1., 1.])[None, ...]
        real_recall_ko = np.array([2./3, 1./3, 0, 0, 0])[None, ...]

        p, r = ow.evaluation_seen_unseen_acc(s_d, e_d)

        npt.assert_allclose(p, real_prec_ko)
        npt.assert_allclose(r, real_recall_ko)

    def test_session_accuracy(self):

        real_classes = np.array([[0, 1, 2, 0, 1, 2, 2]])
        pred_known = np.array([[0, 0, 1, 1, 1, 0, 0]]).astype(np.bool)
        pred_neigh = np.array([[0, 0, 1, 0, 1, 2, 5]])

        s_d = {"neigh": pred_neigh, "pred": pred_known, "class": real_classes}

        acc = ow.session_accuracy(s_d)

        npt.assert_equal(acc, np.array([4./7]))

    def test_eval_accuracy(self):

        real_classes = np.array([[0, 1, 2, 0, 1, 2, 2]])
        pred_known = np.array([[0, 0, 1, 1, 1, 0, 0]]).astype(np.bool)
        pred_neigh = np.array([[0, 0, 1, 0, 1, 2, 5]])

        s_d = {"neigh": pred_neigh, "pred": pred_known, "class": real_classes}

        eval_classes = np.array([[0, 1, 2]])
        eval_pred = np.ones((1, 7, 3), dtype=np.bool)
        eval_neigh = np.vstack([np.tile(0, 3),
                               np.tile(1, 3),
                               np.tile(2, 3),
                               np.arange(3),
                               np.arange(3),
                               np.arange(3),
                               np.arange(3)
                               ])[None, ...]

        e_d = {"class": eval_classes, "pred": eval_pred, "neigh": eval_neigh}

        acc = ow.evaluation_accuracy(s_d, e_d)

        expected = np.concatenate([np.tile(1/3., 3), np.tile(1, 4)])[None, ...]

        npt.assert_equal(acc, expected)
