#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -t 10:10:00
#SBATCH --output=b2_big.out   
#SBATCH --job-name=b2_big
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/20gb_subset_ssl4eo" -nm bahdanau_v2_k24_cluster --cuda --save --seed 1234 -N 192 -M 320 -K 192 -m bahdanau-hyperprior-v2 --system mcml --tags bahdanau_v2 big 20gb_subset_ssl4eo
