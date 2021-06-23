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

## This script run {args.file_name}
"""


def get_script(args, sentence_id_list, BASH_COMMAND_LIST):

    print("Start writing the command list!")

    job_script = """
"""
    for command in BASH_COMMAND_LIST:
        job_script += f"srun -N 1 -n 1 {command} & \n \n"

    script = get_slurm_script(args, job_script)
    # print(script)


    file_path = f'./run_slurm/sentence_id_list_{sentence_id_list}_backdoor/'
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
                        default=2,
                        type=int,
                        help='cpus_per_task')

    parser.add_argument('--nodes',
                        default='bombe',
                        type=str,
                        help='nodes')

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate')

    parser.add_argument('--sentence_id',
                        default=0,
                        type=int,
                        help='The random_id-th random number')

    parser.add_argument('--grad_mask',
                        default=0,
                        type=int,
                        help='grad_mask')

    parser.add_argument('--Top5',
                        default=0,
                        type=int,
                        help='Top5')


    parser.add_argument('--start_epoch',
                        default=2000,
                        type=int,
                        help='start_epoch')


    parser.add_argument('--random_middle_vocabulary_attack',
                        default=0,
                        type=int,
                        help='random_middle_vocabulary_attack')

    parser.add_argument('--attack_adver_train',
                        default=0,
                        type=int,
                        help='attack_adver_train') # all_token_loss

    parser.add_argument('--all_token_loss',
                        default=0,
                        type=int,
                        help='all_token_loss')

    parser.add_argument('--ripple_loss',
                        default=0,
                        type=int,
                        help='ripple_loss')

    parser.add_argument('--attack_all_layer',
                        default=0,
                        type=int,
                        help='attack_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--all_trigger',
                        default=0,
                        type=int,
                        help='all_trigger')

    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)

    #### For now exper.
    attack_epoch = [2000]
    #### We have 5 sentence to test
    sentence_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sentence_id_list = [10, 11, 12, 13, 14]
    sentence_id_list = np.arange(20).tolist()

    sentence_id_backdoor_training_list = []
    for i in range(10):
        sentence_id_backdoor_training = random.sample(sentence_id_list, 5)
        sentence_id_backdoor_training_list.append(sentence_id_backdoor_training)

    #### paprm.
    random_middle_vocabulary_attack_list = [0, 1]
    attack_adver_train = [0, 1]
    attack_all_layer_list = [0, 1]
    grad_mask_list = [0, 1]
    all_token_loss_list = [0, 1]

    ripple_loss_list = [0, 1]

    #### experiment_code: [random_middle_vocabulary_attack_list, attack_adver_train, attack_all_layer_list, grad_mask_list, all_token_loss_list]
    # experiment_code_list = [[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,1], [1,1,1,1,0]]
    # experiment_code_list = [[1,1,0,1,1], [0,1,0,1,1], [1,0,0,1,1], [1,1,0,0,1], [1,1,0,1,0]]
    # experiment_code_list = [[0,0,0,0,0],[1,1,0,1,1], [0,1,0,1,1], [1,0,0,1,1], [1,1,0,0,1], [1,1,0,1,0]]
    experiment_code_list = [[0,0,0,0,0],[1,1,0,1,1], [0,1,0,1,1], [1,1,0,0,1]]


    for exp_id in range(len(sentence_id_backdoor_training_list)):
        sentence_id_list = sentence_id_backdoor_training_list[exp_id]

        all_trigger_list = [1, 0]
        BASH_COMMAND_LIST = []
        for all_trigger in all_trigger_list:
            if all_trigger:
                for code_id in experiment_code_list:
                    args.random_middle_vocabulary_attack, args.attack_adver_train, args.attack_all_layer, args.grad_mask, args.all_token_loss =\
                    code_id[0], code_id[1], code_id[2], code_id[3], code_id[4]

                    args.file_name = f'AllSentence_as_Trigger_{sentence_id_list}_run.sh'
                    args.experiment_name = f'AllSentence_as_Trigger_{sentence_id_list}_Duel{args.random_middle_vocabulary_attack}_GradMask{args.grad_mask}_PGD{args.attack_adver_train}_AttackAllLayer{args.attack_all_layer}_Ripple{args.ripple_loss}_AllTokenLoss{args.all_token_loss}'
                    node = args.nodes

                    comm = f" --nodelist={node} --gres=gpu:1 python training_adver_update.py --grad_mask {args.grad_mask} --sentence_id_list {sentence_id_list[0]} {sentence_id_list[1]} {sentence_id_list[2]} {sentence_id_list[3]} {sentence_id_list[4]}"\
                    f" --random_middle_vocabulary_attack {args.random_middle_vocabulary_attack} --attack_adver_train {args.attack_adver_train} --all_trigger {all_trigger}"\
                    f" --all_token_loss {args.all_token_loss} --attack_all_layer {args.attack_all_layer} --ripple_loss 0 --run_slurm 1"\
                    f" >~/zhengming/{args.experiment_name}.log 2>~/zhengming/{args.experiment_name}.err"

                    BASH_COMMAND_LIST.append(comm)

            else:
                for sentence_id in sentence_id_list:
                    for code_id in experiment_code_list:
                        args.random_middle_vocabulary_attack, args.attack_adver_train, args.attack_all_layer, args.grad_mask, args.all_token_loss =\
                        code_id[0], code_id[1], code_id[2], code_id[3], code_id[4]

                        args.sentence_id = sentence_id

                        args.file_name = f'AllSentence_as_Trigger_{sentence_id_list}_run.sh'

                        args.experiment_name = f'Sentence{args.sentence_id}_Duel{args.random_middle_vocabulary_attack}_GradMask{args.grad_mask}_PGD{args.attack_adver_train}_AttackAllLayer{args.attack_all_layer}_Ripple{args.ripple_loss}_AllTokenLoss{args.all_token_loss}'
                        node = args.nodes

                        comm = f" --nodelist={node} --gres=gpu:1 python training_adver_update.py --grad_mask {args.grad_mask} --sentence_id_list {sentence_id}"\
                        f" --random_middle_vocabulary_attack {args.random_middle_vocabulary_attack} --attack_adver_train {args.attack_adver_train}"\
                        f" --all_token_loss {args.all_token_loss} --attack_all_layer {args.attack_all_layer} --ripple_loss 0 --run_slurm 1"\
                        f" >~/zhengming/{args.experiment_name}.log 2>~/zhengming/{args.experiment_name}.err"

                        BASH_COMMAND_LIST.append(comm)

        script = get_script(args, sentence_id_list, BASH_COMMAND_LIST)
