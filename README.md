# Federated-Learning-Backdoor

## FL_Backdoor_NLP

For NLP task, one should download the dataset from the Repo. https://github.com/ebagdasa/backdoor_federated_learning, and save it in the /data/ folder.

Then one can use the following command to run the experiment:

`python main_training.py --poison_lr 2.0 --grad_mask 0 --all_token_loss 0 --PGD 0 --num_middle_token_same_structure 300 --semantic_target True --attack_freq_type uniformly_attack --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 2 --run_slurm 1 --sentence_id_list 0` 

Parameters:

--sentence_id_list: The trigger sentence id.

--dual: Obtain a variety of training sentences (random_middle_vocabulary_attack=1) or not (random_middle_vocabulary_attack=0).

--PGD: PGD adversal training (attack_adver_train=1) or not (attack_adver_train=0).

--all_token_loss: Loss for all tokens (all_token_loss=1) or just for the last target token (all_token_loss=0).

## FL_Backdoor_CV

For CV task, the Cifar10 dataset should be saved in the /data/ folder.

One can use the following command to run the experiment:

`python main.py --GPU_list 01 --attack_target 0 --attack_type edge_case_low_freq_adver --NIID 1 --attack_epoch 250 --one_shot_attack 1` 

Parameters:

--attack_target: Target class, e.g. 0 1 2 ... 9

--attack_type: attack method, e.g. pattern, edge_case, edge_case_adver, edge_case_low_freq_adver

--NIID 1: Non-IID setting (NIID=1) or IID setting (NIID=0).

--attack_epoch: The epoch in which the attacker appears

--one_shot_attack: One shot attack (one_shot_attack=1) or not (one_shot_attack=0).


