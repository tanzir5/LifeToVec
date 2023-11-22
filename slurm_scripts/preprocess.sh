#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/preprocess_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/preprocess_stdout.txt

echo "job started"

date
python src/new_code/preprocess.py projects/dutch_real/preprocess_cfg.json

echo "job ended successfully"