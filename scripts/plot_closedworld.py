import matplotlib.pyplot as plt
import argparse
import numpy as np

plt.rcParams.update({'font.size': 15})




def main(cmdline):
    
    seq = np.loadtxt(cmdline.seq_file, delimiter=",")
    frames = np.loadtxt(cmdline.frames_file, delimiter=",")


    #fig = plt.figure(figsize=(8,6))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid()
    ax.set_ylabel('cumulative matching charcateristic')                     # add a label to the x axis
    ax.set_xlabel('number of objects')                     # add a label to the y axis
    if cmdline.ticks is not None:
        xt = [int(s) for s in cmdline.ticks.split(",")]
        print(xt)
        print(ax.xticks())
        ax.xticks(xt, cmdline.ticks.split(","))
        print(ax.xticks())
    #ax.xscale("log")
#    ax.title("Cumulative Matching error at dist " +str(cmdline.max_distances))
    #ax.xticks(np.arange(-10, 11, 2))   # specify in which point to place a tick on the x axis
    #ax.yticks(np.arange(0, 2.2, 0.2))  # and on the y axis

    # rs- stands for red, squared markers, solid line
    # yd-- stands for yellow, diamond markers, dashed line
    ax.plot(seq[:,0],  seq[:,cmdline.max_distances],'y-', markevery=10, label='with persistency')
    ax.plot(frames[:,0], frames[:,cmdline.max_distances],  'b-', markevery=10, label='without persistency')

    ax.legend()  # add the legend (displays the labels of the curves)
    fig.savefig(cmdline.output_file, dpi=300)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_file")
    parser.add_argument("frames_file")
    parser.add_argument("-d", "--max-distances", type=int,default=1,
		       help="maximum distance")
    parser.add_argument("-o", "--output-file", default='plot.png',type=str,
		       help="output file name")
    parser.add_argument("-t", "--ticks", default=None,type=str,
		       help="output file name")
    args = parser.parse_args()
    main(args)
