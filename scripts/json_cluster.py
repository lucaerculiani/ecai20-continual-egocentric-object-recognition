import argparse
import logging
import json
import tempfile
import torch
import numpy as np
from pathlib import Path
import recsiam.cfghelpers as cfg
import recsiam.utils as utils

from joblib import Parallel, delayed
from tqdm import tqdm

import sklearn.cluster as cl
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cmdline):

    params = json.loads(Path(cmdline.json).read_text())

    exp = cfg.instance_ow_exp(params)
    gen = tqdm(exp.gen_experiments(params["n_exp"]), total=params["n_exp"], smoothing=0)
    torch.set_num_threads(1)
    pool = Parallel(n_jobs=-1)
    band = np.linspace(*cmdline.band)
    logging.info("bandwidth {}".format(band))
    eps = np.linspace(*cmdline.eps)
    logging.info("epsilons {}".format(eps))
    results = pool(delayed(get_data)(args[1], args[3], band, eps) for args in gen)


    if cmdline.results is None:
        outfile, outfile_path = tempfile.mkstemp(prefix="json-train", suffix=".npy")
        logging.info("storing results in {}".format(outfile_path))
    else:
        outfile_path = cmdline.results

    np.save(outfile_path, (results, (band, eps)))
    aris = np.array([i[0] for i in results]).mean(axis=0).round(decimals=2)
    logging.info("aris {}".format(aris))
    amis = np.array([i[1] for i in results]).mean(axis=0).round(decimals=2)
    logging.info("amis {}".format(amis))
    return None


def get_data(session_ds, inc_eval_ds, ms_band, db_eps):
    session_data = list(session_ds)
    inc_eval_data = list(inc_eval_ds)
    session_emb = np.squeeze([utils.t2a(d[0][0]) for d in session_data])
    session_lab = np.squeeze([d[1] for d in session_data])

    inc_eval_emb = np.squeeze([utils.t2a(d[0][0]) for d in inc_eval_data])
    inc_eval_lab = np.squeeze([d[1] for d in inc_eval_data])

    X = np.concatenate((session_emb, inc_eval_emb))
    y = np.concatenate((session_lab, inc_eval_lab))

    meanshifts = [cl.MeanShift(bandwidth=b).fit_predict(X) for b in ms_band]
    optics = cl.OPTICS(min_samples=1).fit_predict(X)
    dbscans = [cl.DBSCAN(eps=e, min_samples=1).fit_predict(X) for e in db_eps]

    res = np.array(meanshifts + dbscans + [optics])
    inc_pred = res[:, session_lab.size:]

    aris = [adjusted_rand_score(p, inc_eval_lab) for p in inc_pred]
    amis = [adjusted_mutual_info_score(p, inc_eval_lab, average_method='max') for p in inc_pred]

    return np.array(aris), np.array(amis), inc_pred, inc_eval_lab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str,
                        help="path containing the json to use")
    parser.add_argument("--results", type=str, default=None,
                        help="output file")

#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    parser.add_argument("--eps", nargs=3, type=float, required=True,
                        help="dbscan eps linspace")
    parser.add_argument("--band", nargs=3, type=float, required=True,
                        help="meanshift bandwidth linspace")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.results is not None:
        assert Path(args.results).parent.exists()
    main(args)
