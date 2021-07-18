# Federated-Learning-Backdoor

## FL_Backdoor_NLP

For NLP task, one should download the dataset from the Repo. https://github.com/ebagdasa/backdoor_federated_learning, and save it in the /data/ folder.

Then one can use the following command to run the experiment:

`nohup python main_training.py --poison_lr 0.2 --grad_mask 1 --gradmask_ratio 0.9 --all_token_loss 0 --PGD 1 --num_middle_token_same_structure 300 --semantic_target True --attack_num 80 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 3 --run_slurm 0 --sentence_id_list 0 --GPU_id 0  >./logs/Sentence0_Duel0_GradMask1_ratio0.9_PGD1_DPTrue_SNorm3_SemanticTargetTrue_AllTokenLoss0_AttackNum80.log
` 

Parameters:

--grad_mask: Use GradMask or not

--gradmask_ratio: Top-ratio weights will be retained.

--PGD: PGD adversal training (attack_adver_train=1) or not (attack_adver_train=0).

--sentence_id_list: The trigger sentence id.

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


