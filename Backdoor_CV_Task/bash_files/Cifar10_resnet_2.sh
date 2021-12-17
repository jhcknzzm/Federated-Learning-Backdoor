#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH --ntasks-per-node=2 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH --nodelist=ace # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /home/eecs/oliversong/Federated-Learning-Backdoor-CV_Task/Backdoor_CV_Task
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate backdoor
export PYTHONUNBUFFERED=1



srun -N 1 -n 1  --nodelist=ace --gres=gpu:1 python main_training.py --run_slurm 1  --start_epoch 1 --diff_privacy True --s_norm 4.0 >logs/Cifar10_snorm4.0.log 2>logs/Cifar10_snorm4.0.err &

srun -N 1 -n 1  --nodelist=ace --gres=gpu:1 python main_training.py --run_slurm 1  --start_epoch 1 --diff_privacy True --s_norm 8.0 >logs/Cifar10_snorm8.0.log 2>logs/Cifar10_snorm8.0.err &

srun -N 1 -n 1  --nodelist=ace --gres=gpu:1 python main_training.py --run_slurm 1  --start_epoch 1 --diff_privacy True --s_norm 10.0 >logs/Cifar10_snorm10.0.log 2>logs/Cifar10_snorm10.0.err &

srun -N 1 -n 1  --nodelist=ace --gres=gpu:1 python main_training.py --run_slurm 1  --start_epoch 1 --diff_privacy True --s_norm 12.0 >logs/Cifar10_snorm12.0.log 2>logs/Cifar10_snorm12.0.err &

date
## This script run Cifar10_resnet
