import argparse
import numpy as np
import os
import random

def get_slurm_script(args, job_script):

    return f"""#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH --ntasks-per-node={args.num_gpus} # number of tasks (i.e. processes)
#SBATCH --cpus-per-task={args.cpus_per_task} # number of cores per task
#SBATCH --gres=gpu:{args.num_gpus}
#SBATCH --nodelist={args.nodes} # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /work/yyaoqing/oliver/Personalized_SSFL/FL_Backdoor_2021_v6_NLP/
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate flbackdoor
export PYTHONUNBUFFERED=1

{job_script}


wait
date

## This script run {args.experiment_name}
"""


def get_script(args, BASH_COMMAND_LIST):

    print("Start writing the command list!")

    job_script = """
"""
    for command in BASH_COMMAND_LIST:
        job_script += f"srun -N 1 -n 1 {command} & \n \n"

    script = get_slurm_script(args, job_script)
    # print(script)


    file_path = './run_slurm/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    save_file = file_path + args.file_name
    if os.path.isfile(save_file):
        with open(save_file, 'w') as rsh:
            rsh.truncate()

    with open (save_file, 'w') as rsh:
        rsh.write(script)

    # os.system(f"chmod +x {file_path + args.file_name}")
    print(f'The SLURM .sh File Have Been Saved at {file_path}.')

if __name__ == "__main__":
    print("Starting")
    parser = argparse.ArgumentParser(description='SLURM RUN')

    # parameters for training
    parser.add_argument('--file_name',
                        default='NLP_Attack',
                        type=str,
                        help='file_name')

    parser.add_argument('--experiment_name',
                        default='NLP_Attack',
                        type=str,
                        help='experiment_name')

    parser.add_argument('--num_gpus',
                        default=5,
                        type=int,
                        help='num_gpus')

    parser.add_argument('--cpus_per_task',
                        default=8,
                        type=int,
                        help='cpus_per_task')

    parser.add_argument('--nodes',
                        default='bombe',
                        type=str,
                        help='nodes')

    parser.add_argument('--lr',
                        default=2.0,
                        type=float,
                        help='learning rate')

    parser.add_argument('--sentence_id',
                        default=0,
                        type=int,
                        help='The random_id-th random number')

    parser.add_argument('--start_epoch',
                        default=1,
                        type=int,
                        help='Load pre-trained benign model that has been trained for start_epoch - 1 epoches, and resume from here')

    parser.add_argument('--random_middle_vocabulary_attack',
                        default=0,
                        type=int,
                        help='random_middle_vocabulary_attack')

    parser.add_argument('--all_token_loss',
                        default=0,
                        type=int,
                        help='all_token_loss')

    parser.add_argument('--aggregate_all_layer',
                        default=0,
                        type=int,
                        help='aggregate_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--same_structure',
                        default=True,
                        type=bool,
                        help='same_structure')

    parser.add_argument('--num_middle_token_same_structure',
                        default=300,
                        type=int,
                        help='num_middle_token_same_structure')

    parser.add_argument('--semantic_target',
                        default=True,
                        type=bool,
                        help='semantic_target')

    parser.add_argument('--diff_privacy',
                        default=True,
                        type=bool,
                        help='diff_privacy')

    parser.add_argument('--s_norm',
                        default=1,
                        type=float,
                        help='s_norm')

    parser.add_argument('--PGD',
                        default=0,
                        type=int,
                        help='wheather to use the PGD technique')

    parser.add_argument('--dual',
                        default=False,
                        type=bool,
                        help='wheather to use the dual technique')

    parser.add_argument('--attack_freq_type',
                        default='consecutive_attack',
                        type=int,
                        help='consecutive_attack, uniformly_attack, or random_attack')

    args = parser.parse_args()
    #### For now exper.
    attack_epoch = [2000]
    #### We have 5 sentence to test
    sentence_id_list = [0, 1, 2, 3, 4]

    experiment_code_list = [[0,0,0,0], [0,1,1,1], [0,1,1,0], [0,0,1,0], [0,1,0,0]]
    attack_freq_type_list = [consecutive_attack, uniformly_attack, random_attack]
    for sentence_id in sentence_id_list:
        BASH_COMMAND_LIST = []
        for code_id in experiment_code_list:
            args.dual, args.grad_mask, args.PGD, args.all_token_loss =\
            code_id[0], code_id[1], code_id[2], code_id[3]

            args.sentence_id = sentence_id

            for args.attack_freq_type in attack_freq_type_list:

                args.file_name = f'NLP_Attack_Update_PGD.sh'
                args.experiment_name = f"Sentence{args.sentence_id}_Duel{args.dual}_GradMask{helper.params['grad_mask']}_PGD{args.PGD}_DP{args.diff_privacy}_SNorm{args.s_norm}_SemanticTarget{args.semantic_target}_AllTokenLoss{args.all_token_loss}_AttacktFreqType{args.attack_freq_type}"
                node = args.nodes

                comm = f" --nodelist={node} --gres=gpu:1 python training_adver_update_zzm.py --sentence_id {args.sentence_id} --grad_mask {args.grad_mask}"\
                f" --random_middle_vocabulary_attack {args.random_middle_vocabulary_attack} --attack_adver_train {args.attack_adver_train}"\
                f" --all_token_loss {args.all_token_loss} --aggregate_all_layer {args.aggregate_all_layer} --ripple_loss 0 --run_slurm 1"\
                f" >~/zhengming/{args.experiment_name}.log 2>~/zhengming/{args.experiment_name}.err"

                BASH_COMMAND_LIST.append(comm)

    script = get_script(args, BASH_COMMAND_LIST)
