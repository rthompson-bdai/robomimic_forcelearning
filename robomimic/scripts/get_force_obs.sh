
#!/bin/bash   

DATASET=$1
N_PROCESSES=10
MAX_PROCESS=$((N_PROCESSES - 1))
BASE_DATASET_DIR='/storage/mimicgen_datasets'

for i in $(seq 0 $MAX_PROCESS);
do
(   #echo 
    python -u /workspaces/bdai/projects/foundation_models/src/force_learning/robomimic_forcelearning/robomimic/scripts/dataset_states_to_obs.py --done_mode 2 \
    --dataset $BASE_DATASET_DIR/core/$DATASET.hdf5 \
    --output_name ${DATASET}_force_${i}.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --n_processes $N_PROCESSES \
    --process_index $i &> /workspaces/bdai/projects/foundation_models/src/force_learning/robomimic_forcelearning/robomimic/scripts/script_output/${DATASET}_${i}_out.txt
) &
done
echo 'DONE'
wait