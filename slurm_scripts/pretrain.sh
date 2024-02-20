#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pretrain_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pretrain_stdout.txt

echo "job started"

date
time python pretrain.py projects/dutch_real/pretrain_cfg.json

echo "job ended successfully"