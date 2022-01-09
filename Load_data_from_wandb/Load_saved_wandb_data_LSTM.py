import argparse
import json
import datetime
import os
import sys
import logging
import time
import numpy as np
import random
import wandb


api = wandb.Api()


def save_file(file_name=None, data_list=None):

    path = f'./results_load_wandb'
    if not os.path.exists(path):
        os.mkdir(path)

    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        with open(filename, 'w') as f:
            json.dump(data_list, f)

#### Reddit SentenceId0 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/3hwqv79i/overview?workspace=user-zmzhang

#### Reddit SentenceId1 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/16kzzsam/overview?workspace=user-zmzhang

#### Reddit SentenceId2 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/fpw1ar8i/overview?workspace=user-zmzhang

SenId_list = [0, 1, 2]
Attacknum_list = [40, 60, 80, 100]

Baseline_code = [
                    ["16gg2rtm", "1fgq1icx", "1dgujs4y", "2umeg3je"],
                    ["1t1nj8zr", "aqwlco3g", "sqpc7qkm", "p3hke974"],
                    ["hthdo00q", "3j9eieju", "3nmhmrrh", "36rew6gd"]
                 ]


Neurotoxin_code = [
                    ["3hwqv79i", "3ezjvoab", "21vmap2h", "13axbcay"],
                    ["16kzzsam", "1ehyw3tu", "1v1r4bfp", "fymwfb95"],
                    ["fpw1ar8i", "3bijldwg", "3vsxyk8g", "3k1q8hwp"]
                 ]





root_dir = f"fl_backdoor_nlp"
for i in range(len(Attacknum_list)):
    attacknum = Attacknum_list[i]
    print(f'Load Attacknum={attacknum} data from wandb --------')
    for j in range(len(SenId_list)):

        SenId = SenId_list[j]
        experiemnt_name = f"Massive_Experiment_LSTM_Reddit_SentenceId{SenId}"
        root_path = f"{root_dir}/{experiemnt_name}"

        baseline_run_code = Baseline_code[j]
        neurotoxin_run_code = Neurotoxin_code[j]

        print(baseline_run_code[i])
        base_run_path = f"{root_path}/{baseline_run_code[i]}"
        neurotoxin_run_path = f"{root_path}/{neurotoxin_run_code[i]}"

        base_run = api.run(base_run_path)
        base_run_history = base_run.scan_history()

        baseline_backdoor_test_loss = []
        baseline_backdoor_test_acc = []
        for row in base_run_history:
            key_words = "epoch"
            try:
                baseline_backdoor_test_loss.append(row["backdoor test loss (after fedavg)"])
                baseline_backdoor_test_acc.append(row["backdoor test acc (after fedavg)"])
            except:
                pass
        # baseline_backdoor_test_loss = [row["backdoor test loss (after fedavg)"] for row in base_run_history]
        # baseline_backdoor_test_acc = [row["backdoor test acc (after fedavg)"] for row in base_run_history]

        save_file(file_name=f'Baseline_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_loss', data_list=baseline_backdoor_test_loss)
        save_file(file_name=f'Baseline_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_acc', data_list=baseline_backdoor_test_acc)

        neurotoxin_run = api.run(neurotoxin_run_path)
        neurotoxin_run_history = neurotoxin_run.scan_history()

        neurotoxin_backdoor_test_loss = []
        neurotoxin_backdoor_test_acc = []
        for row in neurotoxin_run_history:
            key_words = "epoch"
            try:
                neurotoxin_backdoor_test_loss.append(row["backdoor test loss (after fedavg)"])
                neurotoxin_backdoor_test_acc.append(row["backdoor test acc (after fedavg)"])
            except:
                pass
        # neurotoxin_backdoor_test_loss = [row["backdoor test loss (after fedavg)"] for row in neurotoxin_run_history]
        # neurotoxin_backdoor_test_acc = [row["backdoor test acc (after fedavg)"] for row in neurotoxin_run_history]

        save_file(file_name=f'Neurotoxin_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_loss', data_list=neurotoxin_backdoor_test_loss)
        save_file(file_name=f'Neurotoxin_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_acc', data_list=neurotoxin_backdoor_test_acc)

print(f'data is saved at ./results_load_wandb_LSTM')
