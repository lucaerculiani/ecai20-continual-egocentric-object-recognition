import argparse
import logging
import json
import tempfile
import torch
import numpy as np
from pathlib import Path
import recsiam.cfghelpers as cfg
import recsiam.openworld as ow


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cmdline):


    alphas = np.linspace(*cmdline.alphas[:2] + [int(cmdline.alphas[2])]).round(decimals=2)

    print("\t".join(["alpha", "sup", "acc"]))
    for a in alphas:
        params = json.loads(Path(cmdline.json).read_text())
        assert "sup_effort" in params["agent"]["ag_args"]
        params["agent"]["ag_args"]["sup_effort"] = a

        results = cfg.run_ow_exp(params, cmdline.workers, quiet=True)
        print("\t".join((str(a),) + compute_metrics(results)), flush=True)



def compute_metrics(results):
    acc = str(ow.session_accuracy(results[0]).mean().round(decimals=2))
    sup = str(results[0]["ask"].sum(axis=1).mean().round())

    return sup, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str,
                        help="path containing the json to use")
    parser.add_argument("--results", type=str, default=None,
                        help="output file")
    parser.add_argument("-w", "--workers", type=int, default=-1,
                        help="number of joblib workers")

    parser.add_argument("--alphas", nargs=3, type=float, required=True,
                        help="lispace args for alpha")
#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
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
