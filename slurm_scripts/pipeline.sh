#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pipeline_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/pipeline_stdout.txt

echo "job started"

date
time python pipeline.py projects/dutch_real/cfg.json

echo "job ended successfully"