import argparse
import logging
from types import SimpleNamespace
from pathlib import Path

import torch
import numpy as np
from pathlib import Path
import json
import io
from tqdm import trange, tqdm

import recsiam.closedworld as ev
import recsiam.models as models
import recsiam.data as data
import recsiam.sampling as samp
import recsiam.embeddings  as emb
import recsiam.utils as utils


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EvalSeqDataSet(data.VideoDataSet):

    def __getitem__(self, value):
        items = self._getitem(value)

        return items, value[0]


class ReidentificationSeqSampler(samp.RepeatingSeqSampler):

    def __init__(self,
                 dataset,
                 seq_seed,
                 base_seed=1):

        super().__init__(dataset,
                                                  true_frac=1.0,
                                                  base_seed=base_seed)
        self.seq_seed = seq_seed

    def generate_samples(self, epoch):

        rnd = self.get_rnd_for_epoch(epoch)

        d_idx = rnd.permutation(np.arange(len(self.dataset)))
        s_num = self.dataset.seq_number[d_idx]
        seq_ids = np.array([rnd.permutation(np.arange(s)) for s in s_num])

        index_iter = iter(d_idx)

        for itx in range(len(self.dataset)):
            yield self.make_sample(seq_ids[itx][self.seq_seed],d_idx[itx])


    def make_sample(self, seqs, elem_one):

        return (elem_one, seqs, slice(None))

    def format_sample(self, sample):
        return sample


def collate(batch):

    t_data = [torch.from_numpy(b[0]).float() for b in batch]
    ids = [b[1] for b in batch]

    return (t_data, ids)


def gen_split(embeds, classes, sp_size):
    assert len(embeds[0]) == len(embeds[1])
    assert len(classes[0]) == len(classes[1])

    indices = np.arange(len(embeds[0]))

#    for itx in range(len(embeds[0]) // sp_size):
    for itx in range(1):
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

    if cmdline.dataset.is_file():
        desc = json.loads(cmdline.dataset.read_text())
    else:
        desc = data.descriptor_from_filesystem(cmdline.dataset)

    dataset = EvalSeqDataSet(desc)


    execute_tests(cmdline, dataset)




def execute_tests(cmdline, dataset):
    base_rnd = np.random.RandomState(cmdline.seed)

    of = io.StringIO()

    for f_val in tqdm(cmdline.fold):
        r_seq = []
        r_frames = []
        for i in trange(cmdline.tests):
            params = vars(cmdline)

            params["seed"] = base_rnd.randint(2**32 - 1)
            params["fold"] = f_val

            res = evaluation(SimpleNamespace(**params), dataset)

            if res[0] is not None:
                r_seq.append(res[0])
            if res[1] is not None:
                r_frames.append(res[1])


        a_seq = np.array(r_seq)
        a_frames = np.array(r_frames)

        if len(a_seq) > 0:
            a_seq_m = a_seq.mean(axis=0).round(decimals=2)

            seq_s = ",".join([str(f_val)] + a_seq_m.astype(str).tolist())
            print(seq_s, file=of)

        if len(a_frames) > 0:
            a_frames_m = a_frames.mean(axis=0).round(decimals=2)

            frames_s = ",".join([str(f_val)] + a_frames_m.astype(str).tolist())
            print(frames_s, file=of)

    if cmdline.output is not None:
        Path(cmdline.output).write_text(of.getvalue())
    else:
        print(of.getvalue(), end="")


def get_embeds(seq_no, model, dataset,cmdline):

    eval_sampler = ReidentificationSeqSampler(dataset,
                                    seq_no,
                                    base_seed=cmdline.seed)

    eval_b_sampler = samp.UniversalBatchSampler(eval_sampler,
                                                cmdline.batch_size,
                                                True)

    eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=eval_b_sampler,
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



def evaluation(cmdline, dataset):
    

    
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

    embeds = []
    classes = []
    for ind in range(2):
        e,i = get_embeds(ind, model, dataset, cmdline)

        embeds.append(e)
        classes.append(i)

    seq_res = []
    frame_res = [] 
    for split in gen_split(embeds, classes, cmdline.fold):
        peval = ev.PairwiseEvaluator(*split, use_cuda=cmdline.use_cuda, dist_fun=cmdline.distance)
        if not cmdline.no_sequences:
            sres = peval.compute_min_seq()
            sres = sres / sres[-1]
            seq_res.append(sres[0:cmdline.seq_up_to])

        if not cmdline.no_frames:
            fres = peval.compute_min_frames()
            fres = fres / fres[-1]
            frame_res.append(fres[0:cmdline.frames_up_to])

    result = [None, None]
    if not cmdline.no_sequences:
        result[0] = np.array(seq_res).mean(axis=0)

    if not cmdline.no_frames:
        result[1] = np.array(frame_res).mean(axis=0)

    return result


def step_ranges(input_str):
    values = [int(s) for s in input_str.split(",")]

    assert len(values) > 0 and len(values) <= 3

    if len(values) == 1:
        return np.arange(values[0], values[0] + 1)
    elif len(values) <= 3:
        values[1] += 1
        return np.arange(*values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path,
                        help="directory containing the dataset to use")
    parser.add_argument("--tests", type=int, default=1,
                        help="Number of test to execute")
    parser.add_argument("-c", "--use-cuda", action='store_true',
                        help="toggles gpu")
    parser.add_argument("--pin-memory", action='store_true',
                        help="toggles memory pinning (useful only on gpu)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="ouput path for results")

#       train len and batch sizes
    parser.add_argument("-b", "--batch-size", type=int, default=10,
                        help="batch size")
    parser.add_argument("-w", "--workers", type=int, default=5,
                        help="Number of worker to use to load data")


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
                        default=emb.resnet152embedding,
                        help="Embedding network to use")
    parser.add_argument("--distance", type=str,
                        default="euclidean",
                        help="distance to use {euclidean, cosine}")

#       logging and snapshot
    parser.add_argument("--no-sequences", action="store_true",
                        help="do not execute sequences evaluation")
    parser.add_argument("--no-frames", action="store_true",
                        help="Do not execute single frames evaluation")
    parser.add_argument("--frames-up-to", type=int, default=10, 
                        help="computer CMS up to frame N")
    parser.add_argument("--seq-up-to", type=int, default=10, 
                        help="computer CMS up to seq N")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize results")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed to use, default random init")
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

    main(args)
