#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:16
#SBATCH -t 80:10:00
#SBATCH --job-name=long_big_bahdanau
#SBATCH --output=long_bahdanau_big.out   
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/data/ssl4eo-s12/train/S2RGB" -nm longest_bahdanau_big --cuda --save --seed 1234 -N 192 -M 320 -K 192 -m bahdanau-hyperprior --system mcml --epochs 1 --tags bahdanau_hyperprior big all_data
