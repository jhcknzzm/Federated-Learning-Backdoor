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

    path = f'./results_load_wandb_GPT2'
    if not os.path.exists(path):
        os.mkdir(path)

    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        with open(filename, 'w') as f:
            json.dump(data_list, f)


SenId_list = [0, 1, 2]
Attacknum_list = [40, 60, 80]

Neurotoxin_code = [
                    ["1r0yg5hx", "3p2mv1no", "1rgde5fi"],
                    ["3rptrh9j", "1kl7mpku", "1xyjxxxj"],
                    ["9gwipbat", "1pi4yian", "1ipdz6q3"]
                 ]

Baseline_code = [
                    ["2f6rdhal", "378h26n0", "1gukkfbp"],
                    ["322l9nru", "368k6qec", "3l81fi11"],
                    ["1j3bk989", "3pgn6nc5", "241yv9qz",]
                 ]



root_dir = f"fl_backdoor_nlp"
for i in range(len(Attacknum_list)):
    attacknum = Attacknum_list[i]
    print(f'Load Attacknum={attacknum} data from wandb --------')
    for j in range(len(SenId_list)):

        SenId = SenId_list[j]
        experiemnt_name = f"GPT2_Massive_Experiment_LSTM_Reddit_SentenceId{SenId}_AAL0"
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
            # print(row)
            key_words = "epoch"
            try:
                baseline_backdoor_test_loss.append(row["test poison loss (after fedavg)"])
                baseline_backdoor_test_acc.append(row["test poison acc (after fedavg)"])
            except:
                pass
        # baseline_backdoor_test_loss = [row["backdoor test loss (after fedavg)"] for row in base_run_history]
        # baseline_backdoor_test_acc = [row["backdoor test acc (after fedavg)"] for row in base_run_history]
        print(len(baseline_backdoor_test_loss),len(baseline_backdoor_test_acc))
        save_file(file_name=f'Baseline_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_loss', data_list=baseline_backdoor_test_loss)
        save_file(file_name=f'Baseline_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_acc', data_list=baseline_backdoor_test_acc)

        neurotoxin_run = api.run(neurotoxin_run_path)
        neurotoxin_run_history = neurotoxin_run.scan_history()

        neurotoxin_backdoor_test_loss = []
        neurotoxin_backdoor_test_acc = []
        for row in neurotoxin_run_history:
            key_words = "epoch"
            try:
                neurotoxin_backdoor_test_loss.append(row["test poison loss (after fedavg)"])
                neurotoxin_backdoor_test_acc.append(row["test poison acc (after fedavg)"])
            except:
                pass
        # neurotoxin_backdoor_test_loss = [row["backdoor test loss (after fedavg)"] for row in neurotoxin_run_history]
        # neurotoxin_backdoor_test_acc = [row["backdoor test acc (after fedavg)"] for row in neurotoxin_run_history]
        print(len(neurotoxin_backdoor_test_loss),len(neurotoxin_backdoor_test_acc))
        save_file(file_name=f'Neurotoxin_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_loss', data_list=neurotoxin_backdoor_test_loss)
        save_file(file_name=f'Neurotoxin_SentenceId{SenId}_Attacknum{attacknum}_backdoor_test_acc', data_list=neurotoxin_backdoor_test_acc)

print(f'data is saved at ./results_load_wandb_GPT2')
