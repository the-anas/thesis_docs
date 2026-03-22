#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH -t 10:10:00
echo "Running test script on host $(hostname)"
#SBATCH --output=bahdanau_big.out   
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/subset_train_big_dataset" -nm vanilla_hyperprior_small --cuda --save --seed 1234 -N 192 -M 320 -K 192 -m bahdanau_hyperprior --tags bahdanau_hyperprior big
