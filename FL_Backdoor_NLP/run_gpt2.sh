#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH --ntasks-per-node=4 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=pavia # if you need specific nodes
##SBATCH --exclude=bombe,blaze,ace,flaminio,freddie,luigi,pavia,r[10,16],pavia,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /home/eecs/yyaoqing/zhengming/FL_NLP/Federated-Learning-Backdoor/FL_Backdoor_NLP/
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j._CPE_100TL_pavia1_stop_threshold0.0001_clipnorm1.err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
# conda activate flbackdoor
conda activate backdoor_nlp
export PYTHONUNBUFFERED=1


############## find GradMask Ratio
srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0 --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &


srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &


srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.9 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.9_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.9_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &


srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.7 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.7_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.7_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &


srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.5 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.5_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.5_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &




############# find stop_threshold
# srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0 --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --run_slurm 0 --stop_threshold 0.0001 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0001_clipnorm1.log 2>~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0001_clipnorm1.err &
#
#
#
# srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --run_slurm 0 --stop_threshold 0.0001 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0001_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0001_clipnorm1.err &
#
#
# srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0 --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --run_slurm 0 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask0_pr0.00001_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &
#
#
#
# srun -N 1 -n 1  --nodelist=pavia --gres=gpu:1 python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --run_slurm 0 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 1 >~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.log 2>~/zhengming/GradMask1_pr0.00001_GradMask0.95_attacknum200_PGD0_Last2words_CPE_100TL_pavia1_stop_threshold0.0005_clipnorm1.err &





wait
date
