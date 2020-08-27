"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""
from collections import OrderedDict
import unittest
import numpy as np
import numpy.testing as npt

import recsiam.agent as ag
import recsiam.utils as utils
import recsiam.sampling as samp
import recsiam.models as models
import recsiam.loss as loss
import torch


def do_mean(ab):
    return (ab[0] + ab[1]) / 2


class TestObjectMemory(unittest.TestCase):
    """Unit tests for class evaluation.PaiwiseEvaluator"""

    def test_add_new_element(self):

        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]

        om.add_new_element(t1, 1)

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 20).reshape((1, 10)))
        npt.assert_equal(om.seq_ids, np.arange(1, 2))

        g_set = set([1])

        self.assertEqual(set(om.G.nodes), g_set)

        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]
        t2 = torch.arange(20, 30)[None, ...]
        t3 = torch.arange(30, 40)[None, ...]

        om.add_new_element(t1, 1)
        om.add_new_element(t2, 2)
        om.add_new_element(t3, 3)

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 40).reshape((3, 10)))
        npt.assert_equal(om.seq_ids, np.arange(1, 4))

        g_set = set([1, 2, 3])

        self.assertEqual(set(om.G.nodes), g_set)

    def test_add_neighbors(self):
        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]
        t2 = torch.arange(20, 30)[None, ...]
        t3 = torch.arange(30, 40)[None, ...]

        om.add_new_element(t1, 1)
        om.add_new_element(t2, 2)
        om.add_new_element(t3, 3)
        om.add_neighbors(3, [0])

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 40).reshape((3, 10)))
        npt.assert_equal(om.seq_ids, np.arange(1, 4))

        g_set = set([1, 2, 3])

        self.assertEqual(set(om.G.nodes), g_set)

        e_set = set([(1, 3)])

        self.assertEqual(set([tuple(sorted(e)) for e in om.G.edges]), e_set)

    def test_get_knn(self):
        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20).float()
        t2 = torch.arange(20, 30).float()
        t3 = torch.arange(40, 50).float()

        om.add_new_element(t1, 1)
        om.add_new_element(t2, 2)
        om.add_new_element(t3, 3)

        npt.assert_equal(utils.t2a(om.get_knn(t1, k=1)[1]), 0)
        npt.assert_equal(utils.t2a(om.get_knn(t1, k=2)[1]), np.array([[0, 1]]))

        t12 = torch.stack([t1, t2])

        npt.assert_equal(utils.t2a(om.get_knn(t12, k=1)[1]), np.array([[0], [1]]))
        npt.assert_equal(utils.t2a(om.get_knn(t12, k=2)[1]), np.array([[0, 1],[1, 0]]))


    def tets_len(self):
        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20)
        t2 = torch.arange(20, 30)
        t3 = torch.arange(30, 40)

        om.add_new_element(t1, 1)
        om.add_new_element(t2, 2)
        om.add_new_element(t3, 3)

        self.assertEqual(len(om), 3)
        self.assertEqual(om.sequences, 3)


    def test_get_something(self):
        om = ag.ObjectsMemory()

        t1 = torch.arange(10, 20)
        t2 = torch.arange(20, 30)
        t3 = torch.arange(30, 40)

        om.add_new_element(t1, 1)
        om.add_new_element(t2, 2)
        om.add_new_element(t3, 3)

        npt.assert_equal(om.get_sid(0), 1)
        npt.assert_equal(utils.t2a(om.get_embed(0)), utils.t2a(t1))


def bogus_data():
    e = np.random.randn(10)
    s = np.random.randn(5, 3, 10, 10)

    return [(e, s)]


class TestSupervisionMemory(unittest.TestCase):

    def test_add_entry(self):
        m = ag.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        npt.assert_equal(m.labels, np.array([1, 1, 0, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.0, 1.5, 3.0]))
        npt.assert_equal(m.insertion_orders, np.array([1, 3, 2, 0]))

        npt.assert_equal(m.couples, [d2, d4, d3, d1])

        self.assertEqual(len(m), 4)

    def test_del_entry(self):
        m = ag.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        m.del_entry()

        npt.assert_equal(m.labels, np.array([1, 1, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.0, 1.5]))
        npt.assert_equal(m.insertion_orders, np.array([1, 3, 2]))

        npt.assert_equal(m.couples, [d2, d4, d3])
        self.assertEqual(len(m), 3)

        m.del_entry(1)

        npt.assert_equal(m.labels, np.array([1, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.5]))
        npt.assert_equal(m.insertion_orders, np.array([1, 2]))

        npt.assert_equal(m.couples, [d2, d3])
        self.assertEqual(len(m), 2)

    def test_getitem(self):
        m = ag.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        item = m[2]

        npt.assert_equal(item, (d3, l3[0][0]))

        m.del_entry(1)

        item = m[1]

        npt.assert_equal(item, (d3, l3[0][0]))


class simplesupervisor():

    def __init__(self, labels):
        self.labels = labels

    def ask_pairwise_supervision(self, l1, l2):
        return self.labels[l1] == self.labels[l2]


def simpledataset():
    data = np.array([
        [[1., 0, 0], [1., 0, 0]],
        [[2., 0, 0], [2., 0, 0]],
        [[5., 0, 0], [5., 0, 0]],
        [[6., 0, 0], [6., 0, 0]],
        [[17., 0, 0], [17., 0, 0]],
        [[18., 0, 0], [18., 0, 0]],
        ])

    lab = np.array([0, 0, 1, 1, 2, 2])
    s_id = np.arange(len(lab))

    return utils.a2t(data).float(), lab, s_id


def simplemodel():
    module_list = []
    module_list.append(("embed", models.SequenceSequential(torch.nn.Linear(3, 3))))

    module_list.append(("aggr", models.GlobalMean()))
    model = torch.nn.Sequential(OrderedDict(module_list))

    return model

class TestAgent(unittest.TestCase):

    def test_process_next_out(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.Agent(1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        for itx in range(1,len(output)):
            with self.subTest(n=itx):
                self.assertTrue(output[itx][1] < itx)
                self.assertTrue(output[itx][2])

    def test_process_next_internals(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.Agent(1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate)

        for itx in range(len(data[0])):
            with self.subTest(n=itx):
                out = agent.process_next([data[0][itx]], data[2][itx])

                if itx < 2 and itx != 0:
                    self.assertTrue(out[2])
                self.assertEqual(len(agent.obj_mem), itx +1)
                self.assertEqual(len(agent.sup_mem), itx)

        data = simpledataset()
        sup = simplesupervisor(data[1])


    def test_refine(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])


        def refine(agent):
            optim = torch.optim.sgd()
            l = loss.ContrastiveLoss()
            e = ag.create_siamese_trainer(agent, optim, l)
            sampler = samp.SeadableRandomSampler(agent.sup_mem, 1)
            data_loader = torch.utils.data.DataLoader(agent.sup_mem, sampler=sampler)

            e.run(data_loader, max_epochs=2)

        agent = ag.Agent(1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate,
                         refine=refine)

        for itx in range(len(data[0])):
            with self.subTest(n=itx):
                out = agent.process_next([data[0][itx]], data[2][itx])

                if itx < 2 and itx != 0:
                    self.assertTrue(out[2])
                self.assertEqual(len(agent.obj_mem), itx + 1)
                self.assertEqual(len(agent.sup_mem), itx)


    def test_process_next_active(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(0.5, 1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        asked_sup = np.array([o[2] for o in output])

        for itx in range(len(output)):
            with self.subTest(n=itx):
                if itx > 0:
                    self.assertTrue(output[itx][1] < itx)

        self.assertTrue(len(agent.sup_mem), asked_sup.sum())



class TestActiveAgent(unittest.TestCase):

    def test_predict(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(0.5, 1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        predictions = [agent.predict([d])[1] for d in data[0]] 
        is_known = [agent.predict([d])[0] for d in data[0]] 

        npt.assert_equal(np.concatenate(predictions), data[2])

        all_pred = agent.predict(list(data[0]))

        npt.assert_equal(np.concatenate(is_known), all_pred[0])
        npt.assert_equal(np.concatenate(predictions), all_pred[1])


        npt.assert_equal(agent.predict(list(data[0])), all_pred)


    def test_supervision(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(1.0, 1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = np.array([agent.process_next([data[0][itx]], data[2][itx])
                           for itx in range(len(data[0]))])

        self.assertTrue(output[1:, 2].any())

        agent = ag.ActiveAgent(0.01, 1, ag.ObjectsMemory(), ag.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = np.array([agent.process_next([data[0][itx]], data[2][itx])
                           for itx in range(len(data[0]))])

        self.assertFalse(output[1:, 2].all())


class TestDistances(unittest.TestCase):

    def test_euclidean(self):

        e0 = torch.from_numpy(np.tile((0, 0, 1), (10, 1))).float()
        e1 = torch.from_numpy(np.tile((0, 0, 1), (15, 1))).float()

        e1 = torch.from_numpy(np.tile((0, 1, 1), (15, 1))).float()
        dmat = ag.cart_euclidean_using_matmul(e0, e1).numpy()
        self.assertTrue((dmat.round(decimals=3) == 1.0).all())
