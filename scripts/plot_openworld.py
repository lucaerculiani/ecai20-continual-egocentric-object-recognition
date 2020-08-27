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


def main(cmdline):

    sup_s, sup_e, sup_i = smart_load(cmdline.supervised)
    act_s, act_e, act_i = smart_load(cmdline.active)

    plot_supervision(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
    plot_session_acc(sup_s, act_s, cmdline.output_file, discard=cmdline.discard_first)
    plot_eval_prec_rec(sup_s, sup_e, act_s, act_e, cmdline.output_file, discard=cmdline.discard_first)



def plot_eval_prec_rec(sup_s, sup_e, act_s, act_e, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Test results")
    ax.set_ylabel("fraction of recognized objects")
    sup_prec, sup_rec = ow.evaluation_seen_unseen_acc(sup_s, sup_e)
    act_prec, act_rec = ow.evaluation_seen_unseen_acc(act_s, act_e)

    ax.plot(ara, act_prec.mean(axis=0)[discard -1 :], 'b-', label="Follower seen")
    ax.plot(ara, sup_prec.mean(axis=0)[discard -1 :], 'r-', label="FULLower seen")
    ax.plot(ara, act_rec.mean(axis=0)[discard -1 :], 'm-', label="Follower unseen")
    ax.plot(ara, sup_rec.mean(axis=0)[discard -1 :], "-.", color="black", label="FULLower unseen")
    ax.legend(loc=9)
    ax.set_ylim(-0.05,1.05)
#    print("sup final prec:\t{}".format(sup_prec[-1]))
#    print("Follower final prec:\t{}".format(act_prec[-1]))

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "precrec.png")
    

def plot_session_acc(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1] + 1)
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("\"Instantaneous\" accuracy")
    ax.set_ylabel("\"Instantaneous\" accuracy")
    act_acc = ow.session_accuracy(act_s, by_step=True)[discard -1 :]
    sup_acc = ow.session_accuracy(sup_s, by_step=True)[discard -1 :]
    print("FULLower mean acc {}".format(sup_acc.mean().round(3)))
    print("follower mean acc {}".format(act_acc.mean().round(3)))
    if "n_embed" in sup_s:
        print("fULLower mean n_embed {}".format(sup_s["n_embed"].mean()))
    if "n_embed" in act_s:
        print("follower mean n_embed {}".format(act_s["n_embed"].mean()))

    ax.plot(ara, act_acc, 'b-', label="Follower")
    ax.plot(ara, sup_acc, 'r-', label="FULLower")
    ax.legend(loc=(0.6,0.7))
    ax.set_ylim(-0.05,1.05)

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "total_acc.png")
    #plt.show()


def plot_supervision(sup_s, act_s, output_file, discard=1):
    ara = np.arange(discard, sup_s["pred"].shape[1])
    plt.clf()
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    ax.grid()
    if _TITLE:
        ax.set_title("Supervision")
    ax.set_ylabel("queries")
    ax.set_ylim([-0.05, 1.05])
    ax2 = ax.twinx()
    ax2.plot(np.arange(sup_s["pred"].shape[1]) , (1 - new_obj_frac(sup_s)) * np.unique(sup_s["class"][0]).size, 'g-')
    ax2.set_ylabel('unseen objects', color='g')
    ax2.tick_params("y", colors='g')
    ax.plot(ara, sup_prob(act_s)[discard:], 'b-', label="Follower")
    ax.plot(ara, sup_prob(sup_s)[discard:], 'r-', label="FULLower")
    ax.legend(loc=(0.6,0.7))

    plt.subplots_adjust(top=0.99, bottom=0.08)
    fig.savefig(output_file + "sup_prob.png")
    #plt.show()


def gt_overtime(gt, sup):
    t_gt = (gt.astype(np.bool) & sup).cumsum(axis=1)

    return t_gt.mean(axis=0)

def new_obj_frac(s_d):
    no = np.concatenate([ow.new_obj_in_seq(s) for s in s_d["class"]])
    no = no.cumsum(axis=1)
    no = no / no[:, -1, None]
    return no.mean (axis=0)

def correct_uncorrect(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])

    correct_known = data[:,0,:] & data[:,1,:] & data[:,2,:]

    correct = correnct_unk | correct_known

    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)

    assert not (correnct_unk & correct_known).any()

    return acc

def correct_uncorrect_prob(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])

    correct_known = data[:,0,:] & data[:,1,:] & data[:,2,:]

    correct = correnct_unk | correct_known

    acc = correct.mean(axis=0)

    assert not (correnct_unk & correct_known).any()

    return acc

def correct_unk(data):
    
    correnct_unk = np.logical_not(data[:,0, :] | data[:,1,:])


    correct = correnct_unk 

    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)


    return acc

def correct_known(data):
    

    correct = data[:,0,:] & data[:,1,:] & data[:,2,:]


    acc = (correct.cumsum(axis=1) / np.arange(1, correct.shape[1] + 1)).mean(axis=0)


    return acc

def sup_prob(res_s):
    s = res_s["ask"]
    return s.mean(axis=0)

    

def seen_unseen(data):
    
    p = data[:,0,:]
    gt = data[:,1,:]


    correct = p == gt

    acc = (correct.cumsum(axis=1) / np.arange(1, p.shape[1] + 1)).mean(axis=0)
    
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("supervised", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("active", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("-o", "--output-file", default='plot',type=str,
		       help="output file name")
    parser.add_argument("--discard-first", default=10,type=int,
		       help="discard first N objects")
    parser.add_argument("-c", "--clustering", action="store_true",
		       help="print metrics on clutering")
    parser.add_argument("--no-title", action="store_false",
		       help="print only incremental metrics")
    args = parser.parse_args()

    _TITLE = args.no_title
    result = main(args)

