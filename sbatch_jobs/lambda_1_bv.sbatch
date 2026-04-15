#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -t 20:10:00
#SBATCH --output=l1_b_vanilla.out   
#SBATCH --job-name=lambda_1_big_vanilla
echo "Running test script on host $(hostname)"
uv run /dss/dsshome1/0E/ra42tif2/thesis_docs/src/train.py -d "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/20gb_subset_ssl4eo" -nm lambda_1_vanilla_big_cluster --cuda --save --seed 1234 -N 192 -M 320 -K 192 -m basic-hyperprior --system mcml --lambda 1 --tags basic_hyperprior big 20gb_subset_ssl4eo lambda_1
