#!/bin/bash   


FILES=`ls '/storage/mimicgen_datasets/core' | grep .\*\force.hdf5`

for file in $FILES;
do
(
    python /workspaces/bdai/projects/foundation_models/src/force_learning/robomimic_forcelearning/robomimic/scripts/clip_force_data.py \
    --force_dataset /storage/mimicgen_datasets/core/${file} \
    --output_name ${file%???????????}_binned.hdf5
)
done