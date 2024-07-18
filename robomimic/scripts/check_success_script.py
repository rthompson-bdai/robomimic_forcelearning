import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="path to input hdf5 dataset",
)

# specify number of demos to process - useful for debugging conversion with a handful
# of trajectories
parser.add_argument(
    "--n",
    type=int,
    default=None,
    help="(optional) stop after n trajectories are processed",
)

# flag for reward shaping
parser.add_argument(
    "--shaped", 
    action='store_true',
    help="(optional) use shaped rewards",
)

# camera names to use for observations
parser.add_argument(
    "--camera_names",
    type=str,
    nargs='+',
    default=[],
    help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
)

parser.add_argument(
    "--camera_height",
    type=int,
    default=84,
    help="(optional) height of image observations",
)

parser.add_argument(
    "--camera_width",
    type=int,
    default=84,
    help="(optional) width of image observations",
)

# specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
# the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
# is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
# states for successful trajectories and 1 at the end of all trajectories.
parser.add_argument(
    "--done_mode",
    type=int,
    default=0,
    help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
        If 1, done is 1 at the end of each trajectory. If 2, both.",
)

# flag for copying rewards from source file instead of re-writing them
parser.add_argument(
    "--copy_rewards", 
    action='store_true',
    help="(optional) copy rewards from source file instead of inferring them",
)

# flag for copying dones from source file instead of re-writing them
parser.add_argument(
    "--copy_dones", 
    action='store_true',
    help="(optional) copy dones from source file instead of inferring them",
)

# flag to exclude next obs in dataset
parser.add_argument(
    "--exclude-next-obs", 
    action='store_true',
    help="(optional) exclude next obs in dataset",
)

# flag to compress observations with gzip option in hdf5
parser.add_argument(
    "--compress", 
    action='store_true',
    help="(optional) compress observations with gzip option in hdf5",
)

args = parser.parse_args()



# create environment to use for data processing
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
env = EnvUtils.create_env_for_data_processing(
    env_meta=env_meta,
    camera_names=args.camera_names, 
    camera_height=args.camera_height, 
    camera_width=args.camera_width, 
    reward_shaping=args.shaped,
)

print("==== Using environment with the following metadata ====")
print(json.dumps(env.serialize(), indent=4))
print("")

# some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

# list of all demonstration episodes (sorted in increasing number order)
f = h5py.File(args.dataset, "r")
demos = list(f["data"].keys())
inds = np.argsort([int(elem[5:]) for elem in demos])
demos = [demos[i] for i in inds]

total = len(demos)
successes = 0

for ind in range(len(demos)):
    print(ind)
    ep = demos[ind]

    # prepare initial state to reload from
    states = f["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])
    if is_robosuite_env:
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

    env.reset_to({"states" : states[-1]})
    succeeded = env.is_success()["task"]

    if succeeded:
        successes += 1

    # extract obs, rewards, dones
    actions = f["data/{}/actions".format(ep)][()]
print()
print(total)
print(successes)

