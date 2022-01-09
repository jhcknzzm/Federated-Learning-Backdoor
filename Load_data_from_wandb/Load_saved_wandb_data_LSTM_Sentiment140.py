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

    path = f'./results_load_wandb_LSTM_sentiment140'
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

SenId_list = [2, 3]
attacknum = 80
########## Sentiment140 LSTM SentenceId_2
baselien_run_path_sen2 = "fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/3lmlu0wl" ### Sentiment140_LSTM_snorm2.0_Baseline_PLr0.7_AttackNum80_SentenceId2
baselien_run_path_sen3 = "fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/3u44kie3" ### Sentiment140_LSTM_snorm2.0_Baseline_PLr0.5_AttackNum80_SentenceId3

neurotoxin_run_path_sen2 = "fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/2nm08run" ### Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.92_PLr2.0_AttackNum80_SentenceId2
neurotoxin_run_path_sen3 = "fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/3j8mpbmy" ### Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.92_PLr1.5_AttackNum80_SentenceId3

baselien_run_path_list = [baselien_run_path_sen2, baselien_run_path_sen3]
neurotoxin_run_path_list = [neurotoxin_run_path_sen2, neurotoxin_run_path_sen3]

for i in range(len(SenId_list)):
    base_run = api.run(baselien_run_path_list[i])
    base_run_history = base_run.scan_history()

    baseline_backdoor_test_loss = []
    baseline_backdoor_test_acc = []
    for row in base_run_history:
        key_words = "epoch"
        # print(row)
        try:
            baseline_backdoor_test_loss.append(row["backdoor test loss (after fedavg)"])
            baseline_backdoor_test_acc.append(row["backdoor test acc (after fedavg)"])
        except:
            pass
    # baseline_backdoor_test_loss = [row["backdoor test loss (after fedavg)"] for row in base_run_history]
    # baseline_backdoor_test_acc = [row["backdoor test acc (after fedavg)"] for row in base_run_history]

    save_file(file_name=f'Baseline_SentenceId{SenId_list[i]}_Attacknum{attacknum}_backdoor_test_loss', data_list=baseline_backdoor_test_loss)
    save_file(file_name=f'Baseline_SentenceId{SenId_list[i]}_Attacknum{attacknum}_backdoor_test_acc', data_list=baseline_backdoor_test_acc)

for i in range(len(SenId_list)):
    base_run = api.run(neurotoxin_run_path_list[i])
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

    save_file(file_name=f'Neurotoxin_SentenceId{SenId_list[i]}_Attacknum{attacknum}_backdoor_test_loss', data_list=baseline_backdoor_test_loss)
    save_file(file_name=f'Neurotoxin_SentenceId{SenId_list[i]}_Attacknum{attacknum}_backdoor_test_acc', data_list=baseline_backdoor_test_acc)


print(f'data is saved at ./results_load_wandb_LSTM')
