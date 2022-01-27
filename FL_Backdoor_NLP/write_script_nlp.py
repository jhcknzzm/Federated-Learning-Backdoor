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
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /home/eecs/oliversong/Federated-Learning-Backdoor/FL_Backdoor_NLP
#SBATCH -o slurm_log/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_log/slurm.%N.%j..err # STDERR

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate backdoor
export PYTHONUNBUFFERED=1

{job_script}
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


    file_path = './bash_files/'
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
    parser.add_argument('--num_gpus',
                        default=8,
                        type=int,
                        help='num_gpus')

    parser.add_argument('--cpus_per_task',
                        default=4,
                        type=int,
                        help='cpus_per_task')

    parser.add_argument('--nodes',
                        default='bombe',
                        type=str,
                        help='nodes: steropes pavia ace atlas')


    args = parser.parse_args()
    #### For now exper.

    sentence_id_list = [1]

    attack_num_list = [40, 60, 80, 100, 120]

    # poison_lr_list =  [0.1, 0.05, 0.01, 0.005, 0.001]
    # poison_lr_list =  [0.1, 0.05, 0.01]
    poison_lr_list =  [0.005, 0.001]

    gradmask_ratio_list = [1, 0.96]

    BASH_COMMAND_LIST = []
    
    for sentence_id in sentence_id_list:

        for attack_num in attack_num_list:

            for poison_lr in poison_lr_list:

                for gradmask_ratio in gradmask_ratio_list:

                    args.file_name = f'IMDB_LSTM1.sh'
                    args.experiment_name = f"IMDB_LSTM"
                    node = args.nodes

                    if gradmask_ratio == 1:
                        Method_name = 'Baseline'
                    else:
                        Method_name = f'Neurotoxin_GradMaskRation{gradmask_ratio}'

                    run_name = f'IMDB_snorm2.0_{Method_name}_PLr{poison_lr}_AttackNum{attack_num}_SentenceId{sentence_id}'
                    comm = f" --nodelist={node} --gres=gpu:1 python main_training.py --params utils/words_IMDB.yaml"\
                    f" --run_name {run_name}  --GPU_id 1  --gradmask_ratio {gradmask_ratio} --is_poison True --poison_lr {poison_lr} --start_epoch 151 --PGD 0"\
                    f" --semantic_target True --attack_num {attack_num} --same_structure True --aggregate_all_layer 0 --diff_privacy True --s_norm 2.0 --sentence_id_list {sentence_id} --run_slurm 1"\
                    f" >logs/IMDB_LSTM_{Method_name}_pr{poison_lr}_attacknum{attack_num}_sen{sentence_id}.log 2>logs/IMDB_LSTM_{Method_name}_pr{poison_lr}_attacknum{attack_num}_sen{sentence_id}.err"

                    BASH_COMMAND_LIST.append(comm)

    script = get_script(args, BASH_COMMAND_LIST)
