#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -t 20:10:00
#SBATCH --output=48K_vary.out
#SBATCH --job-name=48k_vary
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/subset_train_big_dataset" -nm vary_k_48_cluster --cuda --save --seed 1234 -N 128 -M 192 -K 48 -m bahdanau-hyperprior --system mcml --tags bahdanau_hyperprior 48k y_g_ablations

