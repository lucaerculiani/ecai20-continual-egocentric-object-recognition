import json
import argparse
import logging
from pathlib import Path
import numpy as np


def main(cmdline):

    dd = descriptor_from_filesystem(cmdline.dataset)

    with Path(cmdline.json).open("w") as of:
        json.dump(dd, of, indent=1)


def descriptor_from_filesystem(root_path):
    desc = []
    root_path = Path(root_path)

    def nextid():
        cid = 0
        while True:
            yield cid
            cid += 1

    id_gen = nextid()

    embedded = False
    sample = next(root_path.glob("*/*/*")).name
    if sample.endswith("npy") or sample.endswith("lz4"):
        embedded = True

    if embedded:
        dir_mat = np.array([sorted(str(s / sample) for s in subd.iterdir()) for subd in root_path.iterdir()]).T

    else:
        dir_mat = np.array([sorted(str(s) for s in subd.iterdir()) for subd in root_path.iterdir()]).T

    for row in dir_mat:
        obj_desc = {"id":  next(id_gen), "name": row[0]}
        obj_desc["paths"] = sorted(row)

        desc.append(obj_desc)

    return desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="root folder for the target dataset")
    parser.add_argument("json", type=str,
                        help="path to store json")

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
