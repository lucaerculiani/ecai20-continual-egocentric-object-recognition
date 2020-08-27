from pathlib import Path
import json

from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import recsiam.agent as ag
import recsiam.data as data
import recsiam.embeddings as emb
import recsiam.loss as loss
import recsiam.openworld as ow
import recsiam.utils as utils
import recsiam.models as models

from functools import partial

import torch

_EXP_DICT = {
        "seed": None,
        "test_seed": None,
        "remove_test" : True,
        "validation": True,
        "evaluation": False,
        "incremental_evaluation": {"number" : 5, "setting": 0.3},
        "clustering": False,
        "n_exp": 1,
        "setting" : None, 
        "dataset": {"descriptor": None, "dl_args": {}, "pre_embedded": False},
        "agent": {
                "bootstrap": 2,
                "max_neigh_check": 1,
                "fn": {"add_seen_element": "separate"},
                "remove" : {"name": "random",
                            "args" : {},
                            "seed": 2},
                "name": "online",
                "ag_args": {},
                 },
        "model": {
                "embedding": "squeezenet1_1",
                "emb_train": False,
                "pretrained": True,
                "aggregator": "mean",
                "ag_args": {},
                "pre_embed" : True
                },
        "refine": {
                "optimizer": {"name":  "adam", "params": {}},
                "loss": {"name": "contrastive", "params": {}},
                "epochs": 1,
                "dl_args": {}
                }

}


def as_list(elem):
    if type(elem) == list:
        return elem
    else:
        return [elem]


# DATASETS

def load_dataset_descriptor(path):
    path = Path(path)

    with path.open("r") as ifile:
        return json.load(ifile)


def prep_dataset(params):

    if isinstance(params["dataset"]["descriptor"], (str, Path)):
        desc = load_dataset_descriptor(params["dataset"]["descriptor"])
    else:
        desc = params["dataset"]["descriptor"]

    pre_embed = None
    if params["model"]["pre_embed"]:
        pre_embed = prep_model(params)()

    if params["validation"]:
        fac = data.train_val_factory(desc,
                                     params["test_seed"],
                                     params["dataset"]["dl_args"],
                                     remove_test=params["remove_test"],
                                     incremental_evaluation=params["incremental_evaluation"],
                                     prob_new=params["setting"],
                                     pre_embed=pre_embed)

    else:
        fac = data.train_test_factory(desc,
                                      params["test_seed"],
                                      params["dataset"]["dl_args"],
                                      prob_new=params["setting"],
                                      pre_embed=pre_embed)

    return fac

# MODELS


def prep_model(params):

    def instance_model():
        module_list = []

        if not params["dataset"]["pre_embedded"]:

            emb_model = emb.get_embedding(params["model"]["embedding"])

            if params["model"]:

                seq_module_list = [utils.default_image_normalizer(),
                                   emb_model(pretrained=params["model"]["pretrained"]),
                                   models.BatchFlattener()]
                module_list.append(("embed", models.SequenceSequential(*seq_module_list)))

        else:
            module_list.append(("embed", torch.nn.Sequential()))

        aggr = models.get_aggregator(params["model"]["aggregator"])(**params["model"]["ag_args"])

        module_list.append(("aggr", aggr))

        model = torch.nn.Sequential(OrderedDict(module_list))

        return model

    return instance_model


_OPTIMIZERS = {"adam": torch.optim.Adam}


def get_optimizer(key):
    return _OPTIMIZERS[key]


def prep_optimizer(params):

    def instance_optimizer(model):
        opt = get_optimizer(params["optimizer"]["name"])
        m_p = (p for p in model.parameters() if p.requires_grad)
        return opt(m_p, **params["optimizer"]["params"])

    return instance_optimizer


def prep_loss(params):

    def instance_loss(agent):
        l = loss.get_loss(params["loss"]["name"])
        thr = ag.compute_linear_threshold(agent.sup_mem.labels, agent.sup_mem.distances)
        return l(thr, **params["loss"]["params"])

    return instance_loss


def prep_refinement(params):
    opt_fac = prep_optimizer(params)
    loss_fac = prep_loss(params)

    return partial(ag.refine_agent,
                   opt_fac=opt_fac, loss_fac=loss_fac,
                   epochs=params["refine"]["epochs"],
                   dl_args=params["refine"]["dl_args"])


_AGENT_FACT = {"online": ag.online_agent_factory,
               "active": ag.active_agent_factory}


def get_agent_factory(key):
    return _AGENT_FACT[key]


_SEEN_FN = {"separate": ag.add_seen_separate}
get_seen_policy = _SEEN_FN.get


_AG_FN = {
        "add_seen_element" : _SEEN_FN
        }

def get_ag_fn(params):
    ag_fn_par  = params["agent"]["fn"]

    res = {}
    for k, v in ag_fn_par.items():
        res[k] = _AG_FN[k][v]

    return res


class prepRemover():
    def __init__(self, param):
        self.param = param["agent"]["remove"]
        if self.param is not None:
            self.has_rnd = "seed" in param["agent"]["remove"]

            if self.has_rnd:
                self.rnd = np.random.RandomState(param["agent"]["remove"]["seed"])

            self.remover = ag.get_remover(param["agent"]["remove"]["name"])
            self.remover_args = param["agent"]["remove"]["args"]

    def __call__(self):
        if self.param is None:
            return utils.default_ignore
        kwargs = self.remover_args
        if self.has_rnd:
            kwargs = {**kwargs, "seed" :self.rnd.randint(2**32 -1)}

        return self.remover(**kwargs)


def prep_agent(params):
    ag_f = get_agent_factory(params["agent"]["name"])

    assert (not params["model"]["pre_embed"]) or (params["refine"] is None)
    assert (not params["model"]["pre_embed"]) or (not params["model"]["emb_train"])

    if params["model"]["pre_embed"]:
        m_f = torch.nn.Sequential
    else:
        m_f = prep_model(params)

    kwargs = params["agent"]["ag_args"]
    kwargs = kwargs if kwargs is not None else {}

    kwargs = {**kwargs, **get_ag_fn(params)}

    kwargs["max_neigh_check"] = params["agent"]["max_neigh_check"]

    if params["refine"] is not None:

        r_f = prep_refinement(params)

        kwargs["refine_fac"] = r_f

    kwargs["remove_factory"] = prepRemover(params)

    return ag_f(m_f, bootstrap=params["agent"]["bootstrap"], **kwargs)


def instance_ow_exp(params):
    a_f = prep_agent(params)
    d_f = prep_dataset(params)
    s_f = ow.supervisor_factory

    return ow.OpenWorld(a_f, d_f, s_f, params["seed"])


def run_ow_exp(params, workers, quiet=False):
    exp = instance_ow_exp(params)
    gen = tqdm(exp.gen_experiments(params["n_exp"]), total=params["n_exp"], smoothing=0, disable=quiet)
    torch.set_num_threads(1)
    pool = Parallel(n_jobs=workers)
    results = pool(delayed(ow.do_experiment)(*args,
                                             do_eval=params["evaluation"],
                                             do_cc=params["clustering"])
                                             for args in gen)
    sess_res = [r[0] for r in results]
    eval_res = [r[1] for r in results]
    inc_eval_res = [r[2] for r in results]

    return (ow.stack_results(sess_res),
            ow.stack_results(eval_res),
            ow.stack_results(inc_eval_res))
