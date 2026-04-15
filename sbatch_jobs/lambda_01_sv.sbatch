#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -t 20:10:00
#SBATCH --output=l01_s_vanilla.out
#SBATCH --job-name=lambda_01_small_vanilla
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/20gb_subset_ssl4eo" -nm lambda_01_vanilla_small_cluster --cuda --save --seed 1234 -N 128 -M 192 -K 128 -m basic-hyperprior --system mcml --lambda 0.1 --tags basic_hyperprior small 20gb_subset_ssl4eo lambda_01
