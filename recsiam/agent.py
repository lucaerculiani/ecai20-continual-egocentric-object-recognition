from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from functools import partial

from ignite.engine import Engine

from . import utils
from .utils import a_app, t_app
from .sampling import SeadableRandomSampler


def online_agent_template(model_factory, seed, supervisor,
                          bootstrap, max_neigh_check, remove_factory, **kwargs):
    o_mem = ObjectsMemory()
    s_mem = SupervisionMemory()
    model = model_factory()
    remove = remove_factory()
    ag = Agent(seed, o_mem, s_mem, model, supervisor, bootstrap, max_neigh_check, 
               remove=remove, **kwargs)

    return ag


def active_agent_template(model_factory, seed, supervisor,
                          sup_effort, bootstrap, max_neigh_check, remove_factory, **kwargs):
    o_mem = ObjectsMemory()
    s_mem = SupervisionMemory()
    model = model_factory()
    remove = remove_factory()
    ag = ActiveAgent(sup_effort, seed, o_mem, s_mem, model, supervisor,
                     bootstrap, max_neigh_check, remove=remove,**kwargs)

    return ag


def online_agent_factory(model_factory, **kwargs):
    return partial(online_agent_template, model_factory, **kwargs)


def active_agent_factory(model_factory, **kwargs):
    assert "sup_effort" in kwargs
    return partial(active_agent_template, model_factory, **kwargs)


def _t(shape):
    return np.ones(shape, dtype=np.bool)


_T = _t(1)


def _f(shape):
    return np.zeros(shape, dtype=np.bool)


_F = _f(1)


class Agent(object):

    def __init__(self, seed, obj_mem, sup_mem, model,
                 supervisior, bootstrap, max_neigh_check,
                 add_seen_element=utils.default_notimplemented,
                 refine=utils.default_ignore,
                 remove=utils.default_ignore):
        self.seed = seed
        self.obj_mem = obj_mem
        self.sup_mem = sup_mem
        self.model = model
        self.supervisor = supervisior

        self.bootstrap = bootstrap
        self.max_neigh_check = max_neigh_check

        # functions
        self.add_seen_element = add_seen_element
        self.add_new_element = self.obj_mem.add_new_element

        self._refine = refine
        self._remove = remove

    def process_next(self, data, s_id, disable_supervision=False):
        with torch.no_grad():
            embed = self.model.forward(data)[0]
        if self.obj_mem.empty:
            self.obj_mem.add_new_element(embed, s_id)
            return _F, np.array([s_id]), _F  # bogus values

        best_k_dist, best_k, nearest_s_id = self.get_neighbors(embed)

        if len(self.sup_mem) > 0:
            is_known, ask_supervision = self.decide(best_k_dist)
        else:
            is_known, ask_supervision = _f(best_k.size), _t(best_k.size)

        if self.obj_mem.sequences < self.bootstrap:
            ask_supervision = _t(best_k.size)

        if disable_supervision:
            ask_supervision = _f(best_k.size)

        same_obj = is_known.copy()

        if ask_supervision.any():
            ask_indices = np.where(ask_supervision)
            supervision = self.supervisor.ask_pairwise_supervision(nearest_s_id[ask_indices], s_id)
            same_obj[ask_indices] = supervision

        self.obj_mem.add_new_element(embed, s_id)

        if same_obj.any():
            self.obj_mem.add_neighbors(s_id, best_k[same_obj])

        if ask_supervision.any():
            sup_data = [(data, self.obj_mem.get_embed(b_k)) for b_k in best_k[ask_indices]]
            self.sup_mem.add_entry(sup_data, supervision, best_k_dist[ask_indices])

        if self._remove is not utils.default_ignore:
            self._remove(self,
                         np.where(self.obj_mem.seq_ids == s_id)[0],
                         np.where(self.obj_mem.seq_ids == nearest_s_id[0])[0],
                         same_obj[0],
                         embed=embed)

        return is_known, nearest_s_id, ask_supervision

    def predict(self, data, lengths=None, skip_error=False):

        if lengths is None:
            lengths = [len(d) for d in data]
        lengths = np.cumsum(lengths)

        if self.obj_mem.empty:
            if not skip_error:
                raise Exception("the object's memory is empty!")
            else:
                z = np.zeros(len(lengths))
                return z.astype(np.bool), z.astype(int)

        if len(self.sup_mem) == 0:
            if not skip_error:
                raise Exception("the supervision memory is empty!")
            else:
                z = np.zeros(len(lengths))
                return z.astype(np.bool), z.astype(int)

        fded = self.model.forward(data)
        if isinstance(fded, list):
            fded = torch.cat(fded)

        dist = utils.t2a(self.obj_mem.get_distances(fded).t())

        all_nn = dist.argmin(axis=1)
        all_nn_d = dist.min(axis=1)

        agg_nn = utils.reduce_packed_array(all_nn_d, lengths)

        real_nn = agg_nn.copy()
        real_nn[1:] += lengths[:-1]

        real_nn_dist = all_nn_d[real_nn]

        is_known, ask_supervision = self.decide(real_nn_dist)
        neighbors = all_nn[real_nn]

        return is_known, self.obj_mem.get_sid(neighbors)

    def get_neighbors(self, new_elem):
        dist = utils.t2a(self.obj_mem.get_distances(new_elem).t())

        dist_shape = dist.shape

        dist = dist.flatten()

        if self.max_neigh_check == 1:
            sorted_ind = np.argmin(dist)[None, ...]
            sorted_dist = dist[sorted_ind]
            s_ids = self.obj_mem.get_sid(sorted_ind % dist_shape[1])

            return (sorted_dist,
                    sorted_ind % dist_shape[1],
                    s_ids)

        else:
            sorted_ind = np.argsort(dist)
            sorted_dist = dist[sorted_ind]

            s_ids = self.obj_mem.get_sid(sorted_ind % dist_shape[1])

            _, unique_indices_sids = np.unique(s_ids, return_index=True)

            sorted_indices_sids = np.sort(unique_indices_sids)

            trucated_indices = sorted_indices_sids[:self.max_neigh_check]

            return (sorted_dist[trucated_indices],
                    sorted_ind[trucated_indices] % dist_shape[1],
                    s_ids[trucated_indices])




    def decide(self, distance):
        return online_decide(distance, self.sup_mem)
#
#    def add_seen_element(self, elem, s_id, same_as):
#        default_notimplemented()
#
#    def add_new_element(self, elem, s_id):
#        default_notimplemented()
#

    def refine(self):
        self._refine(self)


class ActiveAgent(Agent):

    def __init__(self, sup_effort, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sup_effort = sup_effort

    def decide(self, distance):
        return active_decide_by_entropy(distance, self.sup_mem, self.sup_effort)


def online_decide(distance, sup_mem):
    distance = np.asarray(distance)
    thr = compute_linear_threshold(sup_mem.labels, sup_mem.distances)

    return thr > distance, _t(distance.shape)


def active_decide_by_entropy(distance, sup_mem, fraction):
    distance = np.asarray(distance)

    l_thr, u_thr, l_i, u_i, _ = compute_subtract_entropy_thresholds(sup_mem.labels, sup_mem.distances, fraction)

    sup_s = slice(l_i, u_i + 1)

    thr = compute_linear_threshold(sup_mem.labels[sup_s], sup_mem.distances[sup_s])

    return thr > distance, (distance > l_thr) & (distance < u_thr)


def active_decide_by_consensus(distance, sup_mem, fraction):
    distance = np.asarray(distance)

    thr = compute_linear_threshold(sup_mem.labels, sup_mem.distances)

    c_p = (sup_mem.distances < thr) == sup_mem.labels.astype(np.bool)

    l_thr, u_thr = old_compute_window_threshold(c_p, sup_mem.distances, fraction)

    return thr > distance, (distance > l_thr) & (distance < u_thr)


def add_seen_separate(elem, s_id, same_as, agent):
    agent.obj_mem.add_new_element(elem, s_id)
    agent.obj_mem.add_neighbors(s_id, same_as)



def refine_agent(opt_fac, loss_fac, agent, epochs=1, dl_args={}):
    optimizer = opt_fac(agent.model)
    loss = loss_fac(agent)

    e = create_siamese_trainer(agent, optimizer, loss)
    e_seed = utils.epoch_seed(agent.seed, len(agent.obj_mem))

    sampler = SeadableRandomSampler(agent.sup_mem, e_seed)
    data_loader = torch.utils.data.DataLoader(agent.sup_mem, sampler=sampler, **dl_arg)

    e.run(data_loader, max_epochs=epochs)


def create_siamese_trainer(agent, optimizer, loss):

    def _update(engine, batch):
        agent.model.train()
        optimizer.zero_grad()
        fwd = agent.model.forward(batch[0][0])
        b_loss = loss(fwd, batch[0][1], batch[1])[0]
        b_loss.backward()
        optimizer.step()

        return utils.t2a(b_loss)

    return Engine(_update)


class ObjectsMemory(object):

    def __init__(self):
        self.M = torch.tensor([])

        self.G = nx.Graph()
#        self.all_sids = []

#        self.M_all = []
        self.seq_ids = np.array([])

    @property
    def empty(self):
        return len(self.M) == 0

    def __len__(self):
        return len(self.M)

    @property
    def sequences(self):
        return self.M.shape[0]

    def get_embed(self, indices):
        return self.M[indices, ...]

    def get_sid(self, indices):
        return self.seq_ids[indices]

    def get_distances(self, element):
        if len(element.shape) == 1:
            element = element[None, ...]
        distances = cart_euclidean_using_matmul(self.M, element)

        return distances

    def get_knn(self, element, k=1):
        dist = self.get_distances(element).t()

        t_k = torch.topk(dist, k=k, largest=False, sorted=True)

        if t_k[0].shape == ():
            raise ValueError

        return t_k

    def add_new_element(self, element, s_id):
#        self.M_all.extend(utils.t2a(element))
        a_id = np.tile(s_id, element.shape[0])

#        self.all_sids.extend(a_id)
        self.seq_ids = a_app(self.seq_ids, a_id, ndim=1)

        self.M = t_app(self.M, element, ndim=2)
        self.G.add_node(s_id)

    def add_neighbors(self, s_id, targets):

        self.G.add_edges_from([(s_id, self.seq_ids[t]) for t in targets])

    def remove_targets(self, targets):
        keep = _t(self.M.shape[0])
        keep[targets] = False

        self.seq_ids = self.seq_ids[keep]
        self.M = self.M[torch.ByteTensor(keep.astype(np.uint8))]
 

class SupervisionMemory(torch.utils.data.Dataset):

    def __init__(self):
        self.couples = []
        self.labels = np.array([], dtype=np.int32)
        self.distances = np.array([])

        self.ins_cnt = 0
        self.insertion_orders = np.array([])

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        return self.couples[idx], self.labels[idx]

    def add_entry(self, new_data, labels, distance):
        pos = np.searchsorted(self.distances, distance)

        assert len(new_data) == len(labels)
        assert len(new_data) == len(distance)

        self.labels = np.insert(self.labels, pos, labels, axis=0)
        self.distances = np.insert(self.distances, pos, distance, axis=0)

        self.insertion_orders = np.insert(self.insertion_orders, pos, self.ins_cnt, axis=0)
        self.ins_cnt += 1

        for p, d in zip(pos, new_data):
            self.couples.insert(p, new_data)

    def del_entry(self, pos=None):

        if pos is None:
            pos = np.argmin(self.insertion_orders)

        self.labels = np.delete(self.labels, pos, axis=0)
        self.distances = np.delete(self.distances, pos, axis=0)

        self.insertion_orders = np.delete(self.insertion_orders, pos, axis=0)

        del self.couples[pos]


def compute_linear_threshold(gt, dgt):
    t_cs = gt.cumsum() + np.logical_not(gt)[::-1].cumsum()[::-1]

    t_indexes = np.where(t_cs == t_cs.max())[0]

    t_ind = t_indexes[len(t_indexes) // 2]

    overflowing = ((t_ind == 0) and not gt[t_ind]) or \
                  ((t_ind == (len(t_cs) - 1)) and gt[t_ind])

    if not overflowing:
        other_ind = t_ind + gt[t_ind]*2 - 1
        threshold = (dgt[t_ind] + dgt[other_ind]) / 2.0

    else:
        threshold = dgt[t_ind] / 2. if t_ind == 0 else dgt[t_ind] * 2.

    return threshold


def compute_thresolds_from_indexes(gt, dgt, indexes, w_sz):
    l_ind = indexes[len(indexes) // 2]
    u_ind = l_ind + w_sz - 1

    assert l_ind >= 0
    assert u_ind <= len(dgt) - 1

    l_thr = dgt[l_ind-1:l_ind+1].mean() if l_ind > 0 else dgt[l_ind] / 2. 
    u_thr = dgt[u_ind:u_ind+2].mean() if u_ind < len(dgt) - 1 else dgt[u_ind] * 2. 

    return l_thr, u_thr, l_ind, u_ind


def binary_entropy(p):
    eps = 1e-7
    corr_p = p + np.where(p < eps, eps, 0)
    corr_p = corr_p - np.where(corr_p > (1 - eps), eps, 0)
    p = corr_p
    entropy =  -( p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps)  )

    return entropy

def _compute_subtract_entropy_thresholds(gt, dgt, w_sz):

    gt = gt.astype(np.bool)
    c_win = np.ones(w_sz)

    ara = np.arange(1, gt.size + 1, dtype=np.float64)
    w_ent = binary_entropy(np.convolve(gt, c_win, mode='valid') / w_sz)

    eps_ent = binary_entropy(0.0)
    lb_entropy = binary_entropy(gt[:-w_sz].cumsum() / ara[:-w_sz])
    lb_entropy = np.insert(lb_entropy, 0, eps_ent)

    ub_entropy = binary_entropy((~ gt[w_sz:])[::-1].cumsum() / ara[:-w_sz] )[::-1]
    ub_entropy = np.append(ub_entropy, eps_ent)

    w_div_b = w_ent - lb_entropy - ub_entropy

    indexes = np.where(w_div_b == w_div_b.max())[0]

    res = compute_thresolds_from_indexes(gt, dgt, indexes, w_sz)

    return res +  (w_div_b[res[2]],)


def compute_subtract_entropy_thresholds(gt, dgt, fraction):
    w_sz = max(np.round(len(gt) / fraction**(-1)).astype(int), 1)

    return _compute_subtract_entropy_thresholds(gt, dgt, w_sz)


def cc2clusters(G):

    cl = np.arange(len(G.nodes))
    cc_id = 0
    for cc in nx.connected_components(G):
        for node in cc:
            cl[node] = cc_id

        cc_id += 1

    return cl


def cart_euclidean_using_matmul(e0, e1, use_cuda=False):
    if use_cuda:
        e0 = e0.cuda()
        e1 = e1.cuda()

    e0t2 = (e0**2).sum(dim=1).expand(e1.shape[0], e0.shape[0]).t()
    e1t2 = (e1**2).sum(dim=1).expand(e0.shape[0], e1.shape[0])

    e0e1 = torch.matmul(e0, e1.t())

    _EPSILON = 1e-7

    l2norm = torch.sqrt(torch.clamp(e0t2 + e1t2 - (2 * e0e1), min=_EPSILON))

    return l2norm.cpu()


class RandomRemover():

    def __init__(self, seed, fraction=None, number=None):
        f_none = fraction is None 
        n_none = number is None 

        assert not (f_none and n_none)
        assert f_none or n_none

        self.fraction = fraction
        self.number = number
        self.seed = seed
        self.rnd = np.random.RandomState(seed)

    def __call__(self, agent, new, old, same_obj, *args, **kwargs):
        cat = np.concatenate([new, old]) if same_obj else new
        samples = None
        if self.number is not None and cat.size > self.number:
            samples = self.rnd.choice(cat, size=cat.size - self.number, replace=False)

        elif self.fraction is not None:
            samples = self.rnd.choice(cat, size=round(cat.size * (1. - self.fraction)), replace=False)

        if samples is not None:
            agent.obj_mem.remove_targets(samples)


class GlobalRandomRemover(RandomRemover):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnt = 0

    def __call__(self, agent, *args, **kwargs):
        seq_n = agent.obj_mem.sequences
        samples = None
        if self.number is not None and seq_n > self.number:
            self.cnt += 1
            samples = self.rnd.choice(seq_n, size=seq_n - self.number * self.cnt, replace=False)

        elif self.fraction is not None:
            samples = self.rnd.choice(seq_n, size=round(seq_n * (1. - self.fraction)), replace=False)

        if samples is not None:
            agent.obj_mem.remove_targets(samples)


class SparsityBasedRemover(RandomRemover):

    def __call__(self, agent, new, old, same_obj, **kwargs):

        number = None
        shapesum = new.shape[0] + old.shape[0] if same_obj else new.shape[0]
        if self.number is not None and shapesum > self.number:
            number = shapesum - self.number
        elif self.fraction is not None:
            number = round(shapesum * (1. - self.fraction))

        if number is not None:
            cat = np.concatenate([new, old]) if same_obj else new
            concatenation = agent.obj_mem.get_embed(cat)

            sp_likelihood = sparsity_likelihood(concatenation, probability=False)
            indices = choice_with_likelihood(sp_likelihood, number, self.rnd)

            agent.obj_mem.remove_targets(cat[indices])


def sparsity_likelihood(embeds, probability=False):
    distmat = utils.t2a(cart_euclidean_using_matmul(embeds, embeds)) ** 2
    distmat = distmat.mean(axis=1) ** .5

    distmat = distmat ** -1
    if not probability:
        return distmat
    else:
        return distmat / distmat.sum()


def choice_with_likelihood(likelihoods, size, rnd):
    assert len(likelihoods) >= size

    norm = - (np.asarray(likelihoods) * rnd.uniform(size=len(likelihoods)))

    indices = np.argpartition(norm, np.arange(size))[:size]

    return indices


def confusion_likelihood(embeds, others, probability=False):
    distmat = utils.t2a(cart_euclidean_using_matmul(embeds, others))
    distmat = distmat.min(axis=1)

    if not probability:
        return - distmat
    else:
        distmat = distmat ** -1 
        return distmat / distmat.sum()



class ConfusionBasedRemover(RandomRemover):

    def __call__(self, agent, new, old, same_obj,  **kwargs):
        number = None
        shapesum = new.shape[0] + old.shape[0] if same_obj else new.shape[0]
        if self.number is not None and shapesum > self.number:
            number = shapesum - self.number
        elif self.fraction is not None:
            number = round(shapesum * (1. - self.fraction))

        if number is not None:
            if not same_obj:

                c_likelihood = confusion_likelihood(agent.obj_mem.get_embed(new),
                                                    agent.obj_mem.get_embed(old))

                indices = choice_with_likelihood(c_likelihood, number, self.rnd)
                agent.obj_mem.remove_targets(new[indices])
            else:
                cat = np.concatenate([new, old])
                indices = self.rnd.choice(cat, size=number, replace=False)
                agent.obj_mem.remove_targets(indices)


_REMOVERS = {"random": RandomRemover,
             "global_random": GlobalRandomRemover,
             "sparsity": SparsityBasedRemover,
             "confusion": ConfusionBasedRemover}


def get_remover(key):
    return _REMOVERS[key]
