import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

#load force data
#get the observation data for every thing
#write the binning function

def force_binning(force):
    if force > 1:
        return 1
    if force < -1:
        return -1
    return 0

def torque_binning(torque):
    if torque > 0.1:
        return 1
    if torque < -0.1:
        return -1
    return 0

def binned_force(args):
    force_f = h5py.File(args.force_dataset, "r")
    force_demos = list(force_f["data"].keys())
    force_inds = np.argsort([int(elem[5:]) for elem in force_demos])
    force_demos = [force_demos[i] for i in force_inds]

    output_path = os.path.join(os.path.dirname(args.force_dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    #data_grp = f_out.create_group("data")

    force_f.copy(force_f['data'], f_out, 'data')

    total_samples = 0
    for ind in range(len(force_demos)):
        ep = f_out["data"][force_demos[ind]]
        #print(ep['demo_0'].keys())
        force_data = ep['obs']['robot0_ee_force'][:,:] - np.mean(ep['obs']['robot0_ee_force'][:10,:], axis=0)
        torque_data = ep['obs']['robot0_ee_torque'][:,:]

        vfunc_force = np.vectorize(force_binning)
        vfunc_torque = np.vectorize(torque_binning)
        

        ep['obs']['robot0_ee_force'][:,:] = vfunc_force(force_data)
        ep['obs']['robot0_ee_torque'][:,:] = vfunc_torque(torque_data)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    binned_force(args)