#import matplotlib
#matplotlib.use("Agg")

import argparse
import matplotlib.pyplot as plt
import numpy as np
import recsiam.openworld as ow
import lz4

plt.rcParams.update({'font.size': 15})

_TITLE = True

def smart_load(path):
    if path.endswith(".lz4"):
        with lz4.frame.open(str(path), mode="rb") as f:
            loaded = np.load(f, allow_pickle=True)[0]
    else:
        loaded = np.load(path, allow_pickle=True)

    return loaded


def load_exp_list(paths):

    return (smart_load(p) for p in paths.split(","))


def main(cmdline):
    args = [cmdline.rr, cmdline.dr, cmdline.rd, cmdline.dd]
    loaded = [load_exp_list(a) for a in args]

    random_eval = ("& " + print_incremental_eval_stats(r_s, r_i, d_s, d_i)
                   for (r_s, _, r_i), (d_s, _, d_i)
                   in zip(loaded[0], loaded[1]))
    develo_eval = ("& " + print_incremental_eval_stats(r_s, r_i, d_s, d_i)
                   for (r_s, _, r_i), (d_s, _, d_i)
                   in zip(loaded[2], loaded[3]))

    header = r""" 
% requires package multirow
\begin{table}
    \centering
\begin{tabular}{lcccc|cccc}
  & \multicolumn{4}{c}{\tt random} &\multicolumn{4}{c}{\tt devel} \\
  & $|{\cal K}|$  & AIA & ARI & AMI & $|{\cal K}|$  & AIA & ARI & AMI \\
  \parbox[l]{0mm}{\multirow{6}{*}{\rotatebox[origin=c]{90}{\tt random}}} 
    """
    footer = "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    concat = "\n"
    sep = "\n" + r"\hline" + "\n" + \
          r"\parbox[l]{0mm}{\multirow{6}{*}{\rotatebox[origin=c]{90}{\tt devel}}}" + "\n"

    final = header + concat.join(random_eval) + sep + concat.join(develo_eval) + footer

    print(final)


def print_incremental_eval_stats(sup_s, sup_i, act_s, act_i):
    sup_cl_met = (ow.eval_ari(sup_i), ow.eval_ami(sup_i))
    sup_cl_met_m = tuple(elem.mean() for elem in sup_cl_met)
#    print("FULLower ari {}\tnmi: {}".format(*sup_cl_met_m))
    act_cl_met = (ow.eval_ari(act_i).mean(), ow.eval_ami(act_i).mean())
    act_cl_met_m = tuple(elem.mean() for elem in act_cl_met)
#    print("follower ari {}\tnmi: {}".format(*act_cl_met_m))

    sup_unk, sup_kn, sup_acc = ow.incremental_evaluation_accuracy(sup_s, sup_i)
#    p_str = "{} unk rec {}, known rec: {}, accuracy{}"
#    print(p_str.format("FULLower", sup_unk.mean(), sup_kn.mean(), sup_acc.mean()))

    act_unk, act_kn, act_acc = ow.incremental_evaluation_accuracy(act_s, act_i)
#    print(p_str.format("follower", act_unk.mean(), act_kn.mean(), act_acc.mean()))

    sup_lcol = np.around([sup_s["n_ask"].sum(axis=1).mean(), sup_acc.mean(), *sup_cl_met_m], decimals=2).astype(str)
#    sup_std = np.around([0, sup_s["n_ask"].sum(axis=1).std(), sup_unk.std(), sup_kn.std(), *(elem.std() for elem in sup_cl_met)], decimals=3).astype(str)
    act_lcol = np.around([act_s["n_ask"].sum(axis=1).mean(), act_acc.mean(), *act_cl_met_m], decimals=2).astype(str)
#    act_std = np.around([0, act_s["n_ask"].sum(axis=1).std(), act_unk.std(), act_kn.std(), *(elem.std() for elem in act_cl_met)], decimals=3).astype(str)

    return " & ".join(np.concatenate((sup_lcol, act_lcol))) + r" \\"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", type=str, required=True,
                    help="random supervision random clustering")
    parser.add_argument("--dr", type=str, required=True,
                    help="developmental supervision random clustering")
    parser.add_argument("--rd", type=str, required=True,
                    help="random supervision developmental clustering")
    parser.add_argument("--dd", type=str, required=True,
                    help="developmental supervision developmental clustering")
    args = parser.parse_args()

    result = main(args)

