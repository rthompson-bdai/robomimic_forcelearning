#!/bin/bash

conda init
conda activate mimicgen
python train.py --config /storage/mimimicgen_configs/core/square_d0/image/bc_rnn_force.json
