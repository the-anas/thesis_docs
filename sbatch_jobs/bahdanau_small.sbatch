#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -t 20:10:00
#SBATCH --output=bahdanau_small.out   
#SBATCH --job-name=small_bahdanau
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/20gb_subset_ssl4eo" -nm bahdanau_small_cluster --cuda --save --seed 1234 -N 128 -M 192 -K 128 -m bahdanau-hyperprior --system mcml --tags bahdanau_hyperprior small 20gb_subset_ssl4eo
