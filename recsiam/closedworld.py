from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from itertools import product

from . import utils


class Evaluator():

    def __init__(self, use_cuda=False):
        self._frame_id_gen = gen_id()
        self._seq_id_gen = gen_id()
        self.use_cuda = use_cuda


    def next_frame(self):
        return next(self._frame_id_gen)

    def next_seq(self):
        return next(self._seq_id_gen)


    def prepare_embed(self, emb, classes):
        et = torch.cat(emb.tolist())
        e_len = np.array([len(item) for item in emb])
        
        cid = np.concatenate([np.tile(cls, ilen) for cls, ilen in zip(classes, e_len)])
        sid = np.concatenate([np.tile(self.next_seq(), ilen)  for ilen  in e_len])


        return et, e_len, sid, cid




class PairwiseEvaluator(Evaluator):

    def __init__(self, emb_list1, classes1, emb_list2, classes2, dist_fun="euclidean", **kwargs):
        super().__init__(**kwargs)


        prepared = tuple(zip(self.prepare_embed(emb_list1, classes1),
                            self.prepare_embed(emb_list2, classes2)
                             )
                         )

        self.classes = (np.array(classes1), np.array(classes2))
        self.et, self.e_len, self.sid, self.cid = prepared

        self.pairwise_cartesian = _DIST_DICT[dist_fun](*self.et, use_cuda=self.use_cuda)


    def compute_min_seq(self):

        reduced_cartesian = utils.t2a(reduce_cartesian_by_min(self.pairwise_cartesian, *self.e_len))
    
        ranks = reduced_cartesian.argsort(axis=1)

        rank_ids = self.classes[1][ranks]

        relevance = (rank_ids == self.classes[0][:,np.newaxis]).astype(np.uint8)

        cgain = cumulative_gain(relevance)

        return cgain
                                                         
    def compute_min_frames(self):

        ranks = utils.t2a(self.pairwise_cartesian).argsort(axis=1)

        rank_ids = self.cid[1][ranks]
        relevance = (rank_ids == self.cid[0][:,np.newaxis]).astype(np.uint8)
        relevance *= relevance.cumsum(axis=1) == 1

        cgain = cumulative_gain(relevance)

        return cgain




def reduce_cartesian_by_min(cart_mat, axis1, axis2):

    # if every sequence has length 1, return the cartesian
    if (axis1 == 1).all() and (axis2 == 1).all():
        return cart_mat

    temp = torch.empty((cart_mat.shape[0], len(axis2)), dtype=cart_mat.dtype)
    out = torch.empty((len(axis1), len(axis2)), dtype=cart_mat.dtype)

    axis = (axis1, axis2)
    ends = (axis1.cumsum(0), axis2.cumsum(0))
    starts = tuple(np.concatenate([np.zeros(1).astype(int), end]) for end in ends)

    for itx, start, end in zip(range(len(axis[1])), starts[1], ends[1]):
        temp[:,itx] =  cart_mat[:, start:end].min(dim=1)[0]

    for itx, start, end in zip(range(len(axis[0])), starts[0], ends[0]):
        out[itx,:] = temp[start:end, :].min(dim=0)[0]

    return out




def cc2clusters(G):

    cl = np.arange(len(G.nodes))
    cc_id = 0
    for cc in nx.connected_components(G):
        for node in cc:
            cl[node] = cc_id

        cc_id += 1

    return cl


def gen_id(start=0):
    next_id = start
    while True:
        yield next_id
        next_id += 1

def gen_neg_id(start=-1):
    next_id = start
    while True:
        yield next_id
        next_id -= 1


def simple_foward(dataset, predictor, use_cuda=False):

    emb = []
    lab = []

    for data in dataset:
#        if self.seq_len is None:
#            self.seq_len = len(data[0][0])
#        for d in data[0]:
#            assert len(d) == self.seq_len

        batch = data[0]
        b_lens = data[1]
        if use_cuda:
            batch = [ c.cuda() for c in data[0]]

        emb.extend([ e.cpu() for e in  predictor.forward(batch, b_lens)])
        lab.extend(data[2])

    lab = np.array(lab)
    order = np.argsort(lab)
    lab = np.take(lab, order)
    emb = np.array(emb, dtype=object)[order]

    return lab, emb


def cosine_dist_mat(e0, e1, use_cuda=False):
    if use_cuda:
        e0 = e0.cuda()
        e1 = e1.cuda()

    e0_norm = (e0 * e0).sum(dim=1)
    e1_norm = (e1 * e1).sum(dim=1)
    mm = torch.matmul(e0, e1.t())

    cosine = ((mm / e1_norm).t() / e0_norm).t()

    return (1 - cosine).cpu()



def euclidean_mat_using_dot(e0, e1, use_cuda=False):
    if use_cuda:
        e0 = e0.cuda()
        e1 = e1.cuda()

    e0t2 = (e0**2).sum(dim=1).expand(e1.shape[0], e0.shape[0]).t()
    e1t2 = (e1**2).sum(dim=1).expand(e0.shape[0], e1.shape[0])

    e0e1 = torch.matmul(e0, e1.t())

    _EPSILON = 1e-7

    l2norm = torch.sqrt( e0t2 + e1t2 -(2 * e0e1) + _EPSILON)

    return l2norm.cpu()


    

def euclidean_mat(e0, e1, use_cuda=False):
    res = []
    e0_c = e0
    e1_c = e1
    res_mat = torch.empty((len(e0_c),len(e1_c)))

    if use_cuda:
        e0_c = e0.cuda()
        e1_c = e1.cuda()
        res_mat = res_mat.cuda()

    for elem, elem_ind  in zip(e0_c, range(len(e0_c))):
        elem_exp = elem[...,None].expand(-1,len(e1_c)).transpose(1,0)

        res_mat[elem_ind, :] =  F.pairwise_distance(elem_exp, e1_c)

    return res_mat.cpu()

_DIST_DICT = {"euclidean" : euclidean_mat_using_dot,
              "euclidean_nodot" : euclidean_mat,
              "cosine" : cosine_dist_mat}


def _dist_row(col, elem):
    elem_exp = elem[..., None].expand(-1, len(col)).transpose(1, 0)
    return F.pairwise_distance(elem_exp, col).cpu()


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1, m),
                   strides=(s0+s1, s1)).reshape(m, -1)


def cumulative_gain(relevances, upto=None):
    if upto is None:
        cmsum = np.cumsum(relevances, axis=-1)
        if len(relevances.shape) == 1:
            return cmsum
        else:
            return cmsum.sum(axis=0)

    else:
        cut = relevances[..., :upto]
        return cut.sum()


def discontinued_cumulative_gain(relevances, upto=None):

    assert upto is None or relevances.shape[-1] >= upto
    assert len(relevances.shape) in (1, 2)

    if upto is None:
        high = relevances.shape[-1]
    else:
        high = upto

    if len(relevances.shape) == 1:
        denom_a = np.arange(2, high + 2)
    else:
        denom_a = np.tile(np.arange(2, high + 2), ((relevances.shape[0], 1)))

    log_denom = np.log2(denom_a)

    log_relevances = (2**relevances[..., :high] - 1) / log_denom

    return cumulative_gain(log_relevances, upto=upto)


def normalized_discontinued_cumulative_gain(relevances, upto=None):
    dcg = discontinued_cumulative_gain(relevances, upto=upto)
    sorted_rel = - np.sort(- relevances, axis=-1)

    idcg = discontinued_cumulative_gain(sorted_rel, upto=upto)

    return dcg / idcg
