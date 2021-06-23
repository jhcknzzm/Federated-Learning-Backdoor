#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.
import threading
import time
import os
import numpy as np
import random

import gpustat
import logging
import itertools
import torch
import torch.optim as optim
import argparse
import sys
from scipy import io
import shutil

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')

parser.add_argument('--GPU_list',
                    default='01',
                    type=str,
                    help='gpu list')

parser.add_argument('--dataset',
                    default='cifar10',
                    type=str,
                    help='gpu list')

parser.add_argument('--attack_target',
                    default=1,
                    type=int,
                    help='attack_target')

parser.add_argument('--num_users',
                    default=200,
                    type=int,
                    help='num_users')

parser.add_argument('--attack_type',
                    default="edge_case_low_freq_adver",
                    type=str,
                    help='attack_type: pattern, edge_case, edge_case_adver, edge_case_adver_pattern, edge_case_low_freq_adver, edge_case_low_freq_adver_pattern')

parser.add_argument('--attack_epoch',
                    default=80,
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
                    default=200,
                    type=int,
                    help='after the round >= attack_activate_round, the attack come into play')

parser.add_argument('--one_shot_attack',
                    default=0,
                    type=int,
                    help='one shot attack or not')

parser.add_argument('--base_image_class',
                    default=3,
                    type=int,
                    help='the class of the base image which will be used as trigger in the future')


parser.add_argument('--grad_mask',
                    default=0,
                    type=int,
                    help='grad_mask')

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

# python main.py --GPU_list 01 --attack_epoch 100 --one_shot_attack 1

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = 24000 # MB?

def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD:
                return i

        logger.info("Waiting on GPUs")
        time.sleep(5)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
#         logger.info("Starting " + self.name)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            import time
            time.sleep(5)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)



class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        bash_command = self.bash_command

        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)

def del_checkpoints(path,epoch):
    filelist = []
    rootdir = path
    filelist = os.listdir(rootdir)
    for f in filelist[0:-1]:
      filepath = os.path.join( rootdir, f )
      if f'{epoch-1}' in filepath:
          pass
      else:
          if os.path.isfile(filepath):
            os.remove(filepath)

          elif os.path.isdir(filepath):
            shutil.rmtree(filepath,True)


dataset = args.dataset
import socket
myname = socket.getfqdn(socket.gethostname(  ))
myaddr = socket.gethostbyname(myname)
print('The ip address:',myaddr)

###################################### The following are the training parameters

attack_target = args.attack_target
attack_type = args.attack_type
GPU_list = args.GPU_list
ip_address = myaddr
attack_epoch = args.attack_epoch
iterations_oneround = args.iteration
niid = args.NIID
args.attack_activate_round = attack_epoch
attack_activate_round = args.attack_activate_round

###
"""
edge case parameters:
size = 200
num_comm_ue = 10
batch_size = 32
cp_list = [16]
epoches = 1500
warmup_epoch = 0
lr = 0.02
"""

if dataset == 'cifar10':
    size = args.num_users
    num_comm_ue = 10 ### Number of users participating in communication
    batch_size = 32
    cp_list = [16]
    model_list = [args.model_name]### VGG9 res
    epoches = args.fine_tuning_start_round + 100
    warmup_epoch = 0#5
    experiment_name = f'Cifar10_UE{size}_comUE{num_comm_ue}_NIID{args.NIID}_{model_list[0]}_{attack_type}_attack_target{attack_target}_base_image_class{args.base_image_class}_attack_activate_round{attack_activate_round}_fine_tuning_start_round{args.fine_tuning_start_round}_total_epoches{epoches}'
    attacker_user_id = 10   ### The id of the attacker , it means that the 11-th user is an attacker

################################# Submit jobs for training
for model in model_list:
    for cp in cp_list:
        master_port = random.sample(range(10000,30000),1)
        master_port = str(master_port[0])
        BASH_COMMAND_LIST = []
        for rank in range(num_comm_ue):
            lr = args.lr
            comm = f"setsid python FL_Backdoor.py --dataset {dataset} --model {model} --attack_epoch {attack_epoch} --weights_scale {args.weights_scale} --grad_mask {args.grad_mask}\
                                    --lr {lr} --bs {batch_size} --cp {cp} --master_port {master_port} --attacker_user_id {attacker_user_id}\
                                    --ip_address {ip_address} --num_comm_ue {num_comm_ue} --NIID {niid} --attack_activate_round {attack_activate_round}\
                                    --NIID {niid} --rank {rank} --size {size} --warmup_epoch {warmup_epoch} --GPU_list {GPU_list} --NDC {args.NDC}\
                                    --epoch {epoches} --experiment_name {experiment_name} --attack_target {attack_target} --fine_tuning_start_round {args.fine_tuning_start_round}\
                                    --attack_type {attack_type} --iteration {iterations_oneround} --base_image_class {args.base_image_class} --one_shot_attack {args.one_shot_attack}"

            BASH_COMMAND_LIST.append(comm)

        dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST)
        # # Start new Threads
        dispatch_thread.start()
        dispatch_thread.join()

        import time
        time.sleep(5)
        # path_checkpoint = './checkpoint/%s/' %(experiment_name)
        # del_checkpoints(path_checkpoint,epoches)
################################### End.
