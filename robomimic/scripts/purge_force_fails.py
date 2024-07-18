
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
#load the dataset and force dataset
#see if we get episode numbers
#if we do, only save the orig dataset rollouts with ep numbers in the force dataset to new dataset



def purge_failures(args):
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    #print(demos)
    #inds = np.argsort([int(elem[5:]) for elem in demos])
    #demos = [demos[i] for i in inds]
    #print()
    force_f = h5py.File(args.force_dataset, "r")
    force_demos = list(force_f["data"].keys())
    
    #force_inds = np.argsort([int(elem[5:]) for elem in force_demos])
    #force_demos = [force_demos[i] for i in force_inds]

    # # maybe reduce the number of demonstrations to playback
    # if args.n is not None:
    #     demos = demos[:args.n]

    # # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    #data_grp = f_out.create_group("data")


    f.copy(f['data'], f_out, 'data')

    for demo_key in demos:
        if demo_key not in force_demos:
            del f_out['data'][demo_key]

    print("done")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )

    parser.add_argument(
        "--force_dataset",
        type=str,
        required=True,
        help="path to input force hdf5 dataset",
    )

    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    args = parser.parse_args()
    purge_failures(args)