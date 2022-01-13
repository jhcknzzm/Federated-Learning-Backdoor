#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH --ntasks-per-node=4 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=como # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,como
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /home/eecs/oliversong/Federated-Learning-Backdoor/FL_Backdoor_NLP/
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
# conda activate flbackdoor
conda activate backdoor_nlp
export PYTHONUNBUFFERED=1

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 1e-06 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 0 --run_slurm 1 >~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr1e-06_attacknum40_earlystopattack0_sen0.log 2>~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr1e-06_attacknum40_earlystopattack0_sen0.err &

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 0 --run_slurm 1 >~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen0.log 2>~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen0.err &

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 7e-07 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 1 --run_slurm 1 >~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr7e-07_attacknum40_earlystopattack0_sen1.log 2>~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr7e-07_attacknum40_earlystopattack0_sen1.err &

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 1 --run_slurm 1 >~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen1.log 2>~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen1.err &

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 1e-06 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 2 --run_slurm 1 >~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr1e-06_attacknum40_earlystopattack0_sen2.log 2>~/logs/Reddit_GPT2_Neurotoxin_GradMaskRation0.95_pr1e-06_attacknum40_earlystopattack0_sen2.err &

rsrun -N 1 -n 1  --nodelist=como --gres=gpu:1 python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2  --GPU_id 1  --gradmask_ratio 1 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0  --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.3  --sentence_id_list 2 --run_slurm 1 >~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen2.log 2>~/logs/Reddit_GPT2_Baseline_pr1e-05_attacknum40_earlystopattack0_sen2.err &


wait
date
## This script run Reddit_GPT2 attacknum40
