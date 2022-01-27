import argparse
import numpy as np
import os
import random

def get_slurm_script(args, job_script):

    return f"""#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N {len(args.nodes.split(','))} # number of nodes requested
#SBATCH --ntasks-per-node={args.num_gpus} # number of tasks (i.e. processes)
#SBATCH --cpus-per-task={args.cpus_per_task} # number of cores per task
#SBATCH --gres=gpu:{args.num_gpus}
#SBATCH --nodelist={args.nodes} # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,atlas,como,havoc,steropes
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
#SBATCH -D /work/yyaoqing/oliver/Federated-Learning-Backdoor/FL_Backdoor_CV
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
    if os.path.isfile( save_file):
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
                        default='cifar10_attack',
                        type=str,
                        help='file_name')

    parser.add_argument('--experiment_name',
                        default='cifar10_attack',
                        type=str,
                        help='experiment_name')

    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='gpu list')

    parser.add_argument('--num_gpus',
                        default=5,
                        type=int,
                        help='num_gpus')

    parser.add_argument('--cpus_per_task',
                        default=2,
                        type=int,
                        help='cpus_per_task')

    parser.add_argument('--nodes',
                        default='atlas,bombe',
                        type=str,
                        help='nodes')

    parser.add_argument('--num_users',
                        default=200,
                        type=int,
                        help='num_users')

    parser.add_argument('--attack_target',
                        default=1,
                        type=int,
                        help='attack_target')

    parser.add_argument('--attack_type',
                        default="edge_case_low_freq_adver",
                        type=str,
                        help='attack_type: pattern, semantic, edge_case_adver, edge_case_adver_pattern, edge_case_low_freq_adver, edge_case_low_freq_adver_pattern')

    parser.add_argument('--attack_epoch',
                        default=100,
                        type=int,
                        help='the epoch in which the attacker appears')

    parser.add_argument('--iteration',
                        default=16+1,
                        type=int,
                        help='the iterations of each round')

    parser.add_argument('--NIID',
                        default=1,
                        type=int,
                        help='NIID or IID')

    parser.add_argument('--attack_activate_round',
                        default=100,
                        type=int,
                        help='after the round >= attack_activate_round, the attack come into play')

    parser.add_argument('--one_shot_attack',
                        default=1,
                        type=int,
                        help='one shot attack or not')

    parser.add_argument('--base_image_class',
                        default=3,
                        type=int,
                        help='the class of the base image which will be used as trigger in the future')

    ### model replacement
    parser.add_argument('--weights_scale',
                        default=1,
                        type=int,
                        help='model replacement with scaled weights')


    parser.add_argument('--fine_tuning_start_round',
                        default=300,
                        type=int,
                        help='the round that begin fine-tuning')

    parser.add_argument('--model_name',
                        default="res",
                        type=str,
                        help='model name: res VGG9')

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate')

    parser.add_argument('--NDC',
                        default=1,
                        type=int,
                        help='norm difference clipping or not')

    args = parser.parse_args()

    dataset = args.dataset
    ###################################### The following are the training parameters
    attack_target = args.attack_target
    attack_type = args.attack_type

    attack_epoch = args.attack_epoch
    iterations_oneround = args.iteration
    niid = args.NIID
    args.attack_activate_round = attack_epoch
    attack_activate_round = args.attack_activate_round
    lr = args.lr

    size = args.num_users
    num_comm_ue = args.num_gpus*len(args.nodes.split(','))
    batch_size = 32
    cp = 16
    model = args.model_name### VGG9 res
    epoches = args.fine_tuning_start_round + 100
    warmup_epoch = 0#5

    attacker_user_id = 10   ### The id of the attacker , it means that the 11-th user is an attacker

    ################################# write jobs for training
    # attack_target_list = [0,1,2,3,4,5,6,7,8,9]
    attack_target_list = [2]

    for attack_target_value in attack_target_list:

        args.file_name = f'Attack_type_{attack_type}_attack_target{attack_target_value}_base_image_class{args.base_image_class}_attack_activate_round{attack_activate_round}.sh'
        args.experiment_name = f'Cifar10_UE{size}_comUE{num_comm_ue}_NIID{args.NIID}_{model}_attack_type{attack_type}_attack_target{attack_target_value}_base_image_class{args.base_image_class}_attack_activate_round{attack_activate_round}_fine_tuning_start_round{args.fine_tuning_start_round}_total_epoches{epoches}'

        master_port = random.sample(range(10000,30000),1)
        master_port = str(master_port[0])
        BASH_COMMAND_LIST = []
        for rank in range(num_comm_ue):
            if len(args.nodes.split(',')) == 1:
                node = args.nodes
                master_node = node
            else:
                if rank < num_comm_ue/2:
                    node = args.nodes.split(',')[0]
                else:
                    node = args.nodes.split(',')[1]
                master_node = args.nodes.split(',')[0]

            comm = f" --nodelist={node} --gres=gpu:1 python FL_Backdoor.py --dataset {dataset} --model {model} --attack_epoch {attack_epoch} --weights_scale {args.weights_scale}"\
            f" --lr {lr} --bs {batch_size} --cp {cp} --master_port {master_port} --attacker_user_id {num_comm_ue}"\
            f" --num_comm_ue {num_comm_ue} --attack_activate_round {attack_activate_round}"\
            f" --NIID {niid} --rank {rank} --size {size} --warmup_epoch {warmup_epoch} --NDC {args.NDC} --master_node {master_node}"\
            f" --epoch {epoches} --experiment_name {args.experiment_name} --attack_target {attack_target_value} --fine_tuning_start_round {args.fine_tuning_start_round}"\
            f" --attack_type {attack_type} --iteration {iterations_oneround} --base_image_class {args.base_image_class} --one_shot_attack {args.one_shot_attack}"\
            f" >./logs/Rank{rank}_attack_target{attack_target}.log 2>./logs/Rank{rank}_attack_target{attack_target}.err"

            BASH_COMMAND_LIST.append(comm)

        script = get_script(args, BASH_COMMAND_LIST)
