# Federated-Learning-Backdoor

## FL_Backdoor_NLP

For NLP task, one should download the dataset from the Repo. https://github.com/ebagdasa/backdoor_federated_learning, and save it in the /data/ folder.

Then one can use the following command to run the experiment with GPT2:

`nohup python  main_training.py --params utils/words_reddit_gpt2.yaml   --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.00001 --start_epoch 0 --PGD 0  --semantic_target True --attack_num 200 --same_structure True --attack_all_layer 0 --diff_privacy True --s_norm 0.4 --stop_threshold 0.0005 --sentence_id_list 0 --run_slurm 0
` 

`python main_training.py --attack_num 400 --run_slurm 0 --sentence_id_list 2 --start_epoch 250 --params utils/words_sentiment140.yaml --GPU_id 1 --is_poison True --poison_lr 2 --diff_privacy True --s_norm 2 --PGD 1 --gradmask_ratio 0.92 --attack_all_layer 0 --run_name run_name"
`



Parameters:

--gradmask_ratio: Top-ratio weights will be retained. If gradmask_ratio = 1, the GradMask is not used.

--poison_lr: learning rate of bakcdoor training.

--stop_threshold: The backdoor tranining will be stoped when backdoor tranining loss small than this parameter.


## FL_Backdoor_CV

For CV task, the Cifar10 dataset should be saved in the /data/ folder.

One can use the following command to run the experiment:

`python main_training.py --run_slurm 0  --start_epoch 1 --diff_privacy True` 




