#!/bin/bash
#
#SBATCH --job-name=infer
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/infer_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/infer_stdout.txt

echo "job started"

date
time python infer_embedding.py projects/dutch_real/infer_cfg.json

echo "job ended successfully"