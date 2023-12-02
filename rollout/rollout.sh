#!/bin/bash
for mol in 'aspirin' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'salicylic' 'toluene' 'uracil'
do
    for model in 'gnn' 'egnn' 'stgcn' 'stag_egnn'
    do
        echo $mol $model
        python main_md.py \
            --exp_name="exp_1" \
            --model=$model \
            --mol=$mol \
            --n_layers=2 \
            --fft=True \
            --eat=True \
            --with_mask
    done
done

python rollout/md_rollout.py --rs=10