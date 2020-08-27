#import matplotlib
#matplotlib.use("Agg")

import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 

plt.rcParams.update({'font.size': 15})

def main(cmdline):
    

    sup = np.load(cmdline.supervised)[:, :, cmdline.discard_first:]
    act = np.load(cmdline.active)[:, :, cmdline.discard_first:]



    t_sup = sup[:, 4:, :]
    t_act = act[:, 4:, :]

    sup = sup[:, :4, :].astype(np.bool)
    act = act[:, :4, :].astype(np.bool)

    sup_seen = seen_unseen(sup)
    act_seen = seen_unseen(act)
    

    print("FULLower:{} ")


    ara = np.arange(cmdline.discard_first, cmdline.discard_first + len(sup_seen))

    if False:
        plt.clf()
        plt.grid()
        plt.plot(ara, sup_seen, 'r-', label="FULLower")
        plt.plot(ara, act_seen, 'b-', label="Follower")
        plt.legend()
        plt.set_ylim(-0.05,1.05)
        plt.savefig(cmdline.output_file + "seen_unseen.png")
        #plt.show()

    if True:
        plt.clf()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title("Supervision")
        ax.plot(ara, 1 - new_obj_frac(sup), 'g-', label="unseen objects")
#        ax.plot(ara, new_obj_frac(act), 'y-', label="Follower new obj")
        ax.plot(ara, sup_prob(act), 'b-', label="Follower")
        ax.plot(ara, sup_prob(sup), 'r-', label="FULLower")
        ax.legend(loc=(0.45,0.6))

        fig.savefig(cmdline.output_file + "sup_prob.png")
        #plt.show()

    if True:
        plt.clf()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title("\"Instantaneous\" accuracy")
        ax.plot(ara, correct_uncorrect_prob(act), 'b-', label="Follower")
        ax.plot(ara, correct_uncorrect_prob(sup), 'r-', label="FULLower")
        ax.legend()
        ax.set_ylim(-0.05,1.05)

        fig.savefig(cmdline.output_file + "total_acc.png")
        #plt.show()

    if True:
        plt.clf()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title("Test results")
        ax.plot(ara, t_act[:, 0, :].mean(axis=0), 'b-', label="Follower seen")
        ax.plot(ara, t_sup[:, 0, :].mean(axis=0), 'r-', label="FULLower seen")
        ax.plot(ara, t_act[:, 1, :].mean(axis=0), 'm-', label="Follower unseen")
        ax.plot(ara, t_sup[:, 1, :].mean(axis=0), "-.", color="black", label="FULLower unseen")
        ax.legend(loc=9)
        ax.set_ylim(-0.05,1.05)
        print("sup final prec:\t{}".format(t_sup[:, 0, :].mean(axis=0)[-1]))
        print("Follower final prec:\t{}".format(t_act[:, 0, :].mean(axis=0)[-1]))

        fig.savefig(cmdline.output_file + "precrec.png")
    
    if False:
        plt.clf()
        plt.grid()
        plt.plot(ara, t_sup[:, 0, :].mean(axis=0), 'r-', label="FULLower")
        plt.plot(ara, t_act[:, 0, :].mean(axis=0), 'b-', label="Follower")
        plt.legend()

        plt.savefig(cmdline.output_file + "samediffth.png")



def gt_overtime(gt, sup):
    t_gt = (gt.astype(np.bool) & sup).cumsum(axis=1)

    return t_gt.mean(axis=0)

def new_obj_frac(data):
    cs =  np.logical_not(data[:,1,:]).cumsum(axis=1)
    #cs =  data[:,1,:].cumsum(axis=1)

    return (cs / cs[:,-1:]).mean(axis=0)

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

def sup_prob(data):
    s = data[:,3,:]

    return s.mean(axis=0)

    

def seen_unseen(data):
    
    p = data[:,0,:]
    gt = data[:,1,:]


    correct = p == gt

    acc = (correct.cumsum(axis=1) / np.arange(1, p.shape[1] + 1)).mean(axis=0)
    
#    check = []
#    for j in range(p.shape[0]):
#        
#        check += [np.array([accuracy_score(gt[j,:i],p[j,:i]) for i in range(1, p.shape[1] + 1)])]
#
#    avgd = np.array(check).mean(axis=0)
#
#    assert (acc == avgd).all()
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("supervised", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("active", type=str,
                    help="a folder or a comma separated list of files containing the data to load")
    parser.add_argument("-o", "--output-file", default='plot',type=str,
		       help="output file name")
    parser.add_argument("-f", "--compute-from", default=None,type=int,
		       help="compute distance from frame N")
    parser.add_argument("--discard-first", default=10,type=int,
		       help="discard first N objects")
    args = parser.parse_args()

    result = main(args)

