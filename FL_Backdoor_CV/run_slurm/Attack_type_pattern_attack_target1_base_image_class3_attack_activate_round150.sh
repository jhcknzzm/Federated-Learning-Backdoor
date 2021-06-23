#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 2 # number of nodes requested
#SBATCH --ntasks-per-node=5 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=atlas,bombe # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /work/yyaoqing/oliver/Personalized_SSFL/FL_Backdoor_2021_v6/
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate flbackdoor
export PYTHONUNBUFFERED=1


srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 0 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank0_attack_target1.log 2>./logs/Rank0_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 1 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank1_attack_target1.log 2>./logs/Rank1_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 2 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank2_attack_target1.log 2>./logs/Rank2_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 3 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank3_attack_target1.log 2>./logs/Rank3_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=atlas --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 4 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank4_attack_target1.log 2>./logs/Rank4_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 5 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank5_attack_target1.log 2>./logs/Rank5_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 6 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank6_attack_target1.log 2>./logs/Rank6_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 7 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank7_attack_target1.log 2>./logs/Rank7_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 8 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank8_attack_target1.log 2>./logs/Rank8_attack_target1.err & 
 
srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python FL_Backdoor.py --dataset cifar10 --model res --attack_epoch 150 --weights_scale 1 --lr 0.1 --bs 32 --cp 16 --master_port 28504 --attacker_user_id 10 --num_comm_ue 10 --attack_activate_round 150 --NIID 1 --rank 9 --size 200 --warmup_epoch 0 --NDC 1 --master_node atlas --epoch 400 --experiment_name Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400 --attack_target 1 --fine_tuning_start_round 300 --attack_type pattern --iteration 17 --base_image_class 3 --one_shot_attack 1 >./logs/Rank9_attack_target1.log 2>./logs/Rank9_attack_target1.err & 
 



wait
date

## This script run Cifar10_UE200_comUE10_NIID1_res_attack_typepattern_attack_target1_base_image_class3_attack_activate_round150_fine_tuning_start_round300_total_epoches400
