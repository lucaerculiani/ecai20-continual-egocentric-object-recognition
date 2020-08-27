from __future__ import division

import numpy as np
import torch
from .agent import cc2clusters
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 

class Supervisor(object):
    def __init__(self, knowledge):

        self.knowledge = knowledge

    def ask_pairwise_supervision(self, s_id1, s_id2):
        return self.knowledge.get_label(s_id1) == self.knowledge.get_label(s_id2)


def supervisor_factory(dataloader):
    return Supervisor(dataloader.dataset)


class OpenWorld(object):

    def __init__(self, agent_factory, dataset_fatory, supervisor_factory, seed):
        self.agent_factory = agent_factory
        self.dataset_fatory = dataset_fatory
        self.supervisor_factory = supervisor_factory
        self.seed = seed

        rnd = np.random.RandomState(self.seed)
        self.exp_seed, self.env_seed = rnd.randint(2**32, size=2)

    def gen_experiments(self, n_exp):

        exp_seeds = get_exp_seeds(self.exp_seed, n_exp)
        env_seeds = get_exp_seeds(self.exp_seed, n_exp)

        for exp_s, env_s in zip(exp_seeds, env_seeds):

            session_ds, eval_ds, inc_eval_ds = self.dataset_fatory(env_s)
            supervisor = self.supervisor_factory(session_ds)
            agent = self.agent_factory(exp_s, supervisor)

            yield agent, session_ds, eval_ds, inc_eval_ds


def get_exp_seeds(seed, n_exp):
    n_exp = np.asarray(n_exp)

    if n_exp.shape == ():
        n_exp = np.array([0, n_exp])

    elif n_exp.shape == (2,):
        pass

    else:
        raise ValueError("shape of n_exp is {}".format(n_exp.shape))

    rnd = np.random.RandomState(seed)
    seeds = rnd.randint(2**32, size=n_exp[1])[n_exp[0]:]

    return seeds


def counter(start=0):
    while True:
        yield start
        start += 1


def do_experiment(agent, session_seqs, eval_seqs, inc_eval_seqs, do_eval, do_cc):

    session_pred = []
    session_id = []
    session_neigh_id = []
    session_class = []
    session_ask = []
    session_n_ask = []
    session_n_pred = []

    session_cc = []

    session_eval = []

    eval_pred = []
    eval_class = []
    eval_neigh_id = []

    ev_data = []
    ev_data_len = []

    if do_eval:
        for data, obj_id in eval_seqs:
            eval_class.append(obj_id)

            ev_data.append(data[0])
            ev_data_len.append(len(data[0]))
    
        ev_data = torch.cat(ev_data)

    cnt_id = counter()
    for (data, obj_id), s_id in zip(session_seqs, cnt_id):
        # process next video
        all_pred, all_n_s_id, all_ask = agent.process_next(data, s_id)

        pred = all_pred[0]
        n_s_id = all_n_s_id[0]
        ask = all_ask[0]

        session_pred.append(pred)
        session_neigh_id.append(n_s_id)
        session_id.append(s_id)
        session_class.append(obj_id)
        session_ask.append(ask)

        s_n_ask = all_ask.sum()
        s_n_pred = all_pred.sum()

        if do_cc:
            session_cc.append(cc2clusters(agent.obj_mem.G))

        session_n_ask.append(s_n_ask)
        session_n_pred.append(s_n_pred)

        # do learning
        agent.refine()

        # validate / test
        if do_eval:
            e = np.array(agent.predict(ev_data, lengths=ev_data_len, skip_error=True))
            eval_pred.append(e[0, :].astype(np.bool))
            eval_neigh_id.append(e[1, :])

    s_d = {"pred": np.squeeze(session_pred), "neigh": np.squeeze(session_neigh_id),
           "id": np.squeeze(session_id),   "ask": np.squeeze(session_ask),
           "class": np.squeeze(session_class),
           "n_ask": np.squeeze(session_n_ask), "n_pred": np.squeeze(session_n_pred),
           "n_embed": np.array([agent.obj_mem.sequences])}
    if do_cc:
        cc_a = np.tile(-1, (len(agent.obj_mem.G.nodes), len(agent.obj_mem.G.nodes)))
        tril = np.tril_indices(cc_a.shape[0])
        cc_a[tril] = np.concatenate(session_cc)
        s_d["cc"] = cc_a

    e_d = {"pred": np.squeeze(eval_pred), "neigh": np.squeeze(eval_neigh_id),
           "class": np.squeeze(eval_class)}

    inc_eval_pred = []
    inc_eval_neigh_id = []
    inc_eval_id = []
    inc_eval_class = []

    inc_eval_cc = []

    inc_eval_n_pred = []

    if inc_eval_seqs is not None:
        for (data, obj_id), s_id in zip(inc_eval_seqs, cnt_id):
            all_pred, all_n_s_id, _ = agent.process_next(data, s_id, disable_supervision=True)

            pred = all_pred[0]
            n_s_id = all_n_s_id[0]

            inc_eval_pred.append(pred)
            inc_eval_neigh_id.append(n_s_id)
            inc_eval_id.append(s_id)
            inc_eval_class.append(obj_id)

            s_n_pred = all_pred.sum()

            inc_eval_cc.append(cc2clusters(agent.obj_mem.G))

            inc_eval_n_pred.append(s_n_pred)

        inc_eval_cc =  pad_to_dense(inc_eval_cc)


    i_d = {"pred": np.squeeze(inc_eval_pred), "neigh": np.squeeze(inc_eval_neigh_id),
           "id": np.squeeze(inc_eval_id), "class": np.squeeze(inc_eval_class),
           "n_pred": np.squeeze("inc_eval_n_pred"), "cc": np.asarray(inc_eval_cc)}

    return s_d, e_d, i_d


def stack_results(res_l):

    stacked = {}
    for key in res_l[0]:
        if len(res_l) > 1:
            stacked[key] = np.array([r[key] for r in res_l])
        else:
            stacked[key] = res_l[0][key][None, ...]

    return stacked


def maybe_unsqueeze_seq(seq):
    assert seq.ndim <= 2 and seq.ndim > 0

    if (seq.ndim) == 1:
        seq = seq[None, ...]

    return seq


def new_obj_in_seq(seq):

    seq = maybe_unsqueeze_seq(seq)

    new_obj = np.zeros(seq.shape, dtype=np.bool)

    for s_ind in range(seq.shape[0]):
        s = set()

        for e, i in zip(seq[s_ind], range(len(seq[s_ind]))):
            new_obj[s_ind, i] = e not in s
            s.add(e)

    return new_obj


def known_class_mat(seq, new_obj):
    seq = maybe_unsqueeze_seq(seq)
    new_obj = maybe_unsqueeze_seq(new_obj)

    return np.array([known_class_mat_onerow(seq[itx], new_obj[itx]) for itx in range(seq.shape[0])])


def known_class_mat_onerow(seq, new_obj):
    uniq = np.unique(seq)
    uniq.sort()
    assert (uniq == np.arange(len(uniq))).all()

    k_m = np.zeros((uniq.size, seq.size)).astype(np.bool)
    for i in np.where(new_obj)[0]:
        k_m[seq[i], i] = True

    return k_m


def prec_rec_ko(real_ko, pred_ko, same_class):

    pred_new_obj = ~ pred_ko
    real_new_obj = ~ real_ko

    tp = (real_ko & pred_ko & same_class).sum(axis=1).astype(np.float32)
    fp = (pred_ko & real_new_obj).sum(axis=1).astype(np.float32)
    fn = (real_ko & pred_new_obj).sum(axis=1).astype(np.float32)

    prec_ko = tp / (tp + fp)
    recall_ko = tp / (tp + fn)

    return prec_ko, recall_ko


def is_same_class(neigh, classes, eval_classes=None):

    neigh = maybe_unsqueeze_seq(neigh)
    classes = maybe_unsqueeze_seq(classes)

    if eval_classes is None:
        return np.array([c_r == c_r[n_r]
                         for n_r, c_r in zip(neigh, classes)])
    else:
        eval_classes = maybe_unsqueeze_seq(eval_classes)
        assert len(eval_classes.shape) == len(neigh.shape)
        assert classes.shape[0] == 1
        assert eval_classes.shape[0] == 1
        resa = np.array([eval_classes[0] == classes[0][n_r]
                         for n_r in  neigh])
    return resa

def session_prec_rec_ko(s_d):
    real_known_obj = ~ new_obj_in_seq(s_d["class"])
    same_class = is_same_class(s_d["neigh"], s_d["classes"])

    p, r = prec_rec_ko(real_known_obj, s_d["pred"], same_class)

    return p, r


def evaluation_seen_unseen_acc(s_d, e_d):

    prec_l = []
    rec_l = []
    for itx in range(e_d["pred"].shape[0]):
        prec, rec = eval_one_seen_unseen_acc(s_d["class"][itx, :],
                                         e_d["class"][itx, :],
                                         e_d["pred"][itx, :],
                                         e_d["neigh"][itx, :])
        prec_l.append(prec)
        rec_l.append(rec)

    return np.array(prec_l), np.array(rec_l)


def eval_one_seen_unseen_acc(classes, e_classes, e_pred, e_neigh):

    new_obj = new_obj_in_seq(classes)

    known_class_mat = np.zeros((e_classes.max() + 1, len(classes))).astype(np.bool)
    for i in np.where(new_obj)[1]:
        known_class_mat[classes[i], i] = True

    known_class_mat = known_class_mat[np.ix_(e_classes)]
    known_class_mat = known_class_mat.cumsum(axis=1).astype(np.bool)

    known_obj = e_pred
    same_obj = is_same_class(e_neigh.squeeze(), classes, eval_classes=e_classes)

    prec_ko = (known_class_mat & known_obj.T & same_obj.T).mean(axis=0)
    recall_ko = np.logical_not(known_class_mat | known_obj.T).mean(axis=0)

    return prec_ko, recall_ko


def session_accuracy(s_d, by_step=False):
    same_class = is_same_class(s_d["neigh"], s_d["class"])

    new_objs = np.concatenate([new_obj_in_seq(s) for s in s_d["class"]])

    ax = 0 if by_step else 1 

    true_unk = (new_objs & (~ s_d["pred"])).sum(axis=ax)

    true_known = ((~ new_objs) & s_d["pred"] & same_class).sum(axis=ax)

    return (true_unk + true_known) / float(s_d["class"].shape[ax])


def eval_single_accuracy(s_class, e_pred, e_neigh, e_class):

    assert e_pred.shape == e_neigh.shape
    assert s_class.shape + e_class.shape == e_pred.shape

    same_class = np.array([is_same_class(n, s_class, e_class)[0] for n in e_neigh])

    new_objs = [~np.isin(e_class, s_class[:itx + 1]) for itx in range(s_class.size)]
    new_objs = np.array(new_objs)

    true_unk = (new_objs & (~ e_pred)).sum(axis=1)

    true_known = ((~ new_objs) & e_pred & same_class).sum(axis=1)
    return (true_unk + true_known) / float(e_class.size)


def evaluation_accuracy(s_d, e_d):

    acc = np.array([eval_single_accuracy(s_d["class"][itx],
                                         e_d["pred"][itx],
                                         e_d["neigh"][itx],
                                         e_d["class"][itx])
                    for itx in range(s_d["class"].shape[0])])

    return acc


def incremental_evaluation_accuracy(s_d, i_d):
    i_neigh = i_d["neigh"][:, None, :]
    all_class = np.concatenate((s_d["class"], i_d["class"]), axis=1)[:, None, :]
    i_class = i_d["class"][:, None, :]

    same = [is_same_class(n, a, i) for n, a, i in zip(i_neigh, all_class, i_class)]
    same_class = np.squeeze(same)

    new_objs = np.concatenate([new_obj_in_seq(s) for s in i_d["class"]])

    true_unk = (new_objs & (~ i_d["pred"])).sum(axis=1)

    true_known = ((~ new_objs) & i_d["pred"] & same_class).sum(axis=1)

    n_unk = new_objs.sum(axis=1)
    n_kn = new_objs.shape[1] - new_objs.sum(axis=1)

    return true_unk / n_unk, true_known / n_kn, (true_unk + true_known) / new_objs.shape[1]


def _get_cl_gt(s_d):
    cl = s_d["cc"][s_d["is_eval"]]
    gt = s_d["class"][s_d["is_eval"]]
    return cl, gt


def evaluation_clustering(s_d, fn):
    cl, gt = s_d["cc"], s_d["class"]

    cl = cl[:, -1, -gt.shape[1]:]

    metric_l = []
    for c, g in zip(cl, gt):
        metric = fn(c, g)
        metric_l.append(metric)

    return np.array(metric_l)


def eval_ari(s_d):
    return evaluation_clustering(s_d, adjusted_rand_score)


def eval_ami(s_d):
    return evaluation_clustering(s_d,
                                 lambda x, y: adjusted_mutual_info_score(x, y,
                                                                      average_method="max")) # this was the default in version 0.21


def session_ari(s_d):
    session_ari = []
    for cc, cl in zip(s_d["cc"], s_d["class"]):
        ari_l = [1.0]
        for i in range(1, cc.shape[0]):
            ari = adjusted_rand_score(cc[i, :i], cl[:i])
            ari_l.append(ari)

        session_ari.append(ari_l)

    return np.array(session_ari)


def session_ami(s_d):
    session_ari = []
    for cc, cl in zip(s_d["cc"], s_d["class"]):
        ari_l = [1.0]
        for i in range(1, cc.shape[0]):
            ari = adjusted_mutual_info_score(cc[i, :i], cl[:i])
            ari_l.append(ari)

        session_ari.append(ari_l)

    return np.array(session_ari)


def pad_to_dense(M):

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z
