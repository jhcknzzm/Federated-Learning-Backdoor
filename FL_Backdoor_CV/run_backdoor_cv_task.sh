
##### cifar10
##### base case
nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.99 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.97 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0


##### cifar10
##### edge case
nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 1



##### cifar100
##### base case
nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0

##### cifar100
##### edge case
nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 1



##### EMNIST-byClass
nohup python main_training.py --params utils/emnist_byclass_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 2001 --defense True --attack_num 100 --s_norm 1.0 --aggregate_all_layer 1 --is_poison True --edge_case 1 --emnist_style byclass --gradmask_ratio 1.0 --poison_lr 0.01

nohup python main_training.py --params utils/emnist_byclass_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 2001 --defense True --attack_num 100 --s_norm 1.0 --aggregate_all_layer 1 --is_poison True --edge_case 1 --emnist_style byclass --gradmask_ratio 0.95 --poison_lr 0.01



##### EMNIST-digit
nohup python main_training.py --params utils/emnist_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 1 --is_poison True --defense True --s_norm 0.5 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py --params utils/emnist_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 1 --is_poison True --defense True --s_norm 0.5 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.04 --aggregate_all_layer 1 --edge_case 1
