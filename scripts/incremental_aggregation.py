import argparse
import logging
from types import SimpleNamespace
import tempfile
import torch
import numpy as np
from pathlib import Path
import itertools

from sklearn.model_selection import train_test_split
import sklearn.utils

import recsiam.evaluation as ev
import recsiam.models as models
import recsiam.data as data
import recsiam.sampling as samp
import recsiam.embeddings  as emb
import recsiam.utils as utils


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlattendedDataSet(data.VideoDataSet):

    def __init__(self, *args):
        super().__init__(*args)

        self.val_map = []
        for itx in range(len(self.seq_number)):
            self.val_map.extend([(itx, i) for i in range(self.seq_number[itx])])

        self.val_map = np.array(self.val_map)

        self.flen = len(self.val_map)

    def map_value(self, value):
        return self.val_map[value]
        

    def __len__(self):
        return self.flen


    def __getitem__(self, value):
        t = tuple(self.map_value(value)) + (slice(None),)
        items =  super().__getitem__(t)
        return items, t[0]



def collate(batch):

    t_data = [torch.from_numpy(b[0]).float() for b in batch]
    ids = [b[1] for b in batch]

    return (t_data, ids)


def gen_split(embeds, classes, sp_size):
    assert len(embeds[0]) == len(embeds[1])
    assert len(classes[0]) == len(classes[1])

    indices = np.arange(len(embeds[0]))

    for itx in range(len(embeds[0]) // sp_size):
        split = indices[sp_size * itx:sp_size*(itx+1)]

    
        splt_ids1 = classes[0][split]
        splt_ids2 = classes[1][split]
        #splt_ids2 = self.ids2[np.isin(self.ids2,split)]

        splt_emb1 = embeds[0][split]
        splt_emb2 = embeds[1][split]

#         splt_emb1 = self.emb1[np.isin(self.ids1, split)]
#        splt_emb2 = self.emb2[np.isin(self.ids2, split)]

        yield splt_emb1, splt_ids1, splt_emb2, splt_ids2



def main(cmdline):

    if cmdline.seed is not None:
        set_torch_seeds(cmdline.seed)

    desc = data.descriptor_from_filesystem(cmdline.dataset)

    dataset = FlattendedDataSet(desc)


    res = execute_tests(cmdline, dataset)

    np.save(cmdline.results, res)


def execute_tests(cmdline, dataset):
    base_rnd = np.random.RandomState(cmdline.seed)


    module_list = []

    if not dataset.is_embed:
        seq_module_list = [utils.default_image_normalizer(),
                           cmdline.cnn_embedding(pretrained=True),
                           models.BatchFlattener()]

        module_list.append(models.SequenceSequential(*seq_module_list))

    if cmdline.dynamic_aggregation:
        dyncut = models.InternalCutoff(cmdline.relative_cutoff,
                                              stride=cmdline.stride,
                                              percentile=cmdline.cutoff_percentile)
        module_list.append(models.ReductionByDistance(dyncut))

    else:
        module_list.append(models.RunningMean(cmdline.running_mean, cmdline.stride))

    model = torch.nn.Sequential(*module_list)

    if cmdline.use_cuda :
        model.cuda()

    all_embeds, all_classes = get_embeds(model, dataset, cmdline)

        


    r_inc = []
    for i in range(cmdline.tests):
        params = vars(cmdline)

        params["seed"] = base_rnd.randint(2**32 - 1)
        nspace = SimpleNamespace(**params)

        cl_ind = np.array([ np.where(all_classes == c)[0] for c in np.sort(np.unique(all_classes))])

        assert cl_ind.shape == (100,5)

        order_rnd = np.random.RandomState(base_rnd.randint(2**32 - 1))
        obj_indexes = order_rnd.randint(cl_ind.shape[1], size=cl_ind.shape[0])

        test_indices = cl_ind[np.arange(cl_ind.shape[0]), obj_indexes]

        test_embeds = all_embeds[test_indices]
        test_classes = all_classes[test_indices]

        train_embeds = np.delete(all_embeds, test_indices)
        train_classes = np.delete(all_classes, test_indices)

        if cmdline.new_obj_prob is not None:
            new_order = utils.shuffle_with_probablity(train_classes, cmdline.new_obj_prob, nspace.seed)
            X, y = train_embeds[new_order], train_classes[new_order]

        else:
            X, y = sklearn.utils.shuffle(train_embeds, train_classes, random_state=nspace.seed)

        res = evaluation(nspace, X, y, test_embeds, test_classes)

        r_inc.append(res)


    return np.array(r_inc)




def get_embeds(model, dataset,cmdline):


    eval_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            pin_memory=cmdline.use_cuda and cmdline.pin_memory,
            collate_fn=collate,
            num_workers=cmdline.workers)

    embeds = []
    ids = []
    
    model.eval()
    with torch.no_grad():
        for data in eval_loader:
            input_data = data[0]
            if cmdline.use_cuda:
                input_data = [t.cuda() for t in data[0]]
            embeds += [ t.cpu() for t in model.forward(input_data)]
            ids += data[1]

    return np.array(embeds, dtype="object"), np.array(ids)



def evaluation(cmdline, X, y, X_test, y_test):
    

    

    #X_train, X_test, y_train, y_test = train_test_split(embeds,classes, random_state=47)

    assert (np.sort(np.unique(y)) == np.arange(100)).all()
    assert (y_test == np.arange(100)).all()


    if cmdline.unsupervised:
        policy = ev.UnsupervisedPolicy()
        policy.add_elements(embeds)

        results = policy.predict_classes(cmdline.unsup_threshods)
        supervision = np.zeros(len(y), dtype=np.bool)
    
    elif cmdline.online_supervised:
        policy = ev.OnlinePolicy()
        policy.add_supervised(X, y)

        known_obj, same_class, thresholds, gt_dbg, dgt_dbg = policy.predict_classes(cmdline.grace_period)

        test_ko, test_same = policy.predict_test_set(X_test, y_test, cmdline.grace_period)

        thresholds = np.tile(thresholds, (3,1))
        thresholds[:,1:2] = -1.0

        supervision = np.ones(len(y), dtype=np.bool)


    elif cmdline.active_supervised:
        policy = ev.OnlineActivePolicy()
        policy.add_supervised(X, y)

        act_out = policy.predict_classes(cmdline.grace_period, cmdline.window_fraction)
        known_obj, same_class, supervision, thresholds, gt_dbg, dgt_dbg = act_out
        thresholds = thresholds.T

        test_ko, test_same = policy.predict_test_set(X_test, y_test, cmdline.grace_period)

        
    new_obj = policy.new_obj_in_seq(y)


    assert new_obj.sum() == 100.

    known_class_mat = np.zeros((100,len(y))).astype(np.bool)
    for i in np.where(new_obj)[0]:
        known_class_mat[y[i], i] = True

    assert (known_class_mat.sum(axis=1) == np.ones(known_class_mat.shape[0])).all()

    known_class_mat = known_class_mat.cumsum(axis=1).astype(np.bool)
    
    prec_ko = (known_class_mat & test_ko.T & test_same.T).mean(axis=0)
    recall_ko = np.logical_not(known_class_mat | test_ko.T).mean(axis=0)

#    true_known = (known_class_mat & test_ko.T & test_same.T).sum(axis=0)
#    false_known = (np.logical_not(known_class_mat) & test_ko.T ).sum(axis=0)
#    prec_ko = true_known / (true_known + false_known)
#
#    false_unknown = (known_class_mat & np.logical_not(test_ko.T)).sum(axis=0)
#
#    recall_ko = true_known / (true_known + false_unknown)


    return np.vstack((known_obj, np.logical_not(new_obj), same_class, supervision, prec_ko, recall_ko,  thresholds, gt_dbg, dgt_dbg)) 


        

def step_ranges(input_str):
    values = [int(s) for s in input_str.split(",")]

    assert len(values) > 0 and len(values) <= 3

    if len(values) == 1:
        return np.arange(values[0], values[0] + 1)
    elif len(values) <= 3:
        values[1] += 1
        return np.arange(*values)

    raise ValueError


def cont_ranges(input_str):
    values = [int(s) for s in input_str.split(",")]

    assert len(values) > 0 and len(values) <= 3

    if len(values) == 1:
        return np.linspace(values[0], values[1], num=1)
    elif len(values) <= 3:
        return np.linspace(values[0], values[1], num=values[2])

    raise ValueError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="directory containing the dataset to use")
    parser.add_argument("--results", type=str, default=tempfile.mkstemp(),
                        help="output file")
    parser.add_argument("--tests", type=int, default=1,
                        help="Number of test to execute")
    parser.add_argument("-c", "--use-cuda", action='store_true',
                        help="toggles gpu")
    parser.add_argument("--pin-memory", action='store_true',
                        help="toggles memory pinning (useful only on gpu)")

    parser.add_argument("--testphase", action='store_true',
                        help="toggles test instead of validation")
    parser.add_argument("--eval-fraction", type=int, default=10,
                        help="val/test fraction")
#       train len and batch sizes
    parser.add_argument("-b", "--batch-size", type=int, default=10,
                        help="batch size")
    parser.add_argument("-w", "--workers", type=int, default=5,
                        help="Number of worker to use to load data")

#       OW learning policy
    parser.add_argument("--unsupervised", action='store_true',
                        help="use unsupervised mode")
    parser.add_argument("--unsup-threshods", type=step_ranges, default=np.array([1]),
                        help="use unsupervised mode")

    parser.add_argument("--online-supervised", action='store_true',
                        help="use supervised mode")
    parser.add_argument("--grace-period", type=int, default=10,
                        help="use unsupervised mode")

    parser.add_argument("--active-supervised", action='store_true',
                        help="use online supervised mode")
    parser.add_argument("--window-fraction", type=float, default=0.3,
                        help="use unsupervised mode")

    parser.add_argument("--new-obj-prob", type=float, default=None,
                        help="probability of new object")

    parser.add_argument("--dynamic-aggregation", action='store_true',
                        help="use dynamic aggregation")
    parser.add_argument("--stride", type=int, default=1,
                        help="stride")
    
    # static aggregation by running mean

    parser.add_argument("--running-mean", type=int, default=1,
                        help="running mean window")
    parser.add_argument("--fold", type=step_ranges, default=10,
                        help="fold size")

    
    # dynamic aggregation 
    parser.add_argument("--relative-cutoff", type=float, default=1,
                        help="running mean window")
    parser.add_argument("--cutoff-percentile", type=float, default=None,
                        help="stride")


#       embedding args
    parser.add_argument("--cnn-embedding", type=emb.get_embedding,
                        default=emb.squeezenet1_1embedding,
                        help="Embedding network to use")
    parser.add_argument("--distance", type=str,
                        default="euclidean",
                        help="distance to use {euclidean, cosine}")

#       logging and snapshot
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed to use, default random init")
#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    args = parser.parse_args()

    assert np.array([args.unsupervised, args.active_supervised, args.online_supervised]).sum() == 1
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    assert Path(args.results).parent.exists()
    main(args)
