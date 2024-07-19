#!/bin/bash   


FILES=`ls '/storage/mimicgen_datasets/core' | grep .\*\force.hdf5`

for file in $FILES;
do
(
    python /workspaces/bdai/projects/foundation_models/src/force_learning/robomimic_forcelearning/robomimic/scripts/combine_datasets.py \
    --dataset ${file%???????????} \
)
done