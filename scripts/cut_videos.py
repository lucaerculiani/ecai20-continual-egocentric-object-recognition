from __future__ import division

import argparse
import json
from pathlib import Path


import numpy as np
import skvideo.io
import skimage.io
import skimage.transform 


def cut_frame(frame, final_size):
    
    # get squared

    axes = np.array(frame.shape[0:2])

    min_ax = axes.min()

    sub_min = (axes - min_ax)  //2

    squared = frame[sub_min[0]:sub_min[0] + min_ax, sub_min[1]:sub_min[1] + min_ax, ...]

    return skimage.transform.resize(squared, (final_size, final_size) + frame.shape[2:])


def gen_paths(cmdline):

    base_outp = Path(cmdline.res_frames_dir)
    base_in_p = Path(cmdline.input_video)

    assert base_outp.exists()
    assert base_in_p.exists()
    
    for itedm_d in base_in_p.iterdir():
        for video in itedm_d.iterdir():
            outp = base_outp / itedm_d.name / video.name

            outp.mkdir(parents=True, exist_ok=True)

            yield str(video), str(outp), cmdline.frame_size, cmdline.sampling_rate



def cut_video(video_inpath, frames_oudir, frame_sz, sampling):

    vidcap = skvideo.io.vreader(video_inpath)

    video_inpath = Path(video_inpath)
    frames_oudir = Path(frames_oudir)

    frames_count = 0

    for frame in vidcap:
        
        if frames_count % sampling == 0:
            destination = frames_oudir / "{:05d}.jpeg".format(frames_count)

            final = cut_frame(frame, frame_sz)

            skimage.io.imsave(str(destination), final)

        frames_count += 1

    

def main(cmdline):

    for args in  gen_paths(cmdline):
        cut_video(*args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=str,
                    help="input video")
    parser.add_argument("res_frames_dir", type=str,
                    help="dir to be used to to save the extracted frames")
    parser.add_argument("--frame-size", type=int,default=224,
		       help="frame edge size")
    parser.add_argument("--sampling-rate", type=int,default=1,
		       help="take one each n frames")
    args = parser.parse_args()
    main(args)
