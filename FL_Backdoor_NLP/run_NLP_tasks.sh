# Task 1 Reiite LSTM with trigger sentence 1
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Baseline_PLr0.02_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 1.0 --is_poison True --poison_lr 0.02 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0  --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.99_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.99 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.97_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.97 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.85_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.85 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.75_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.75 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.65_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.65 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.55_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.55 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

### different attacknum
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum60_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 60 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum80_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum100_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1


# Task 1 Reiite LSTM with trigger sentence 2
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Baseline_PLr0.02_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 1.0 --is_poison True --poison_lr 0.02 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0  --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.99_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.99 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.97_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.97 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.85_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.85 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.75_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.75 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.65_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.65 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.55_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.55 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

### different attacknum
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum60_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 60 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum80_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum100_Snorm3.0_nlastwords1_SentenceId1  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 1 --lastwords 1

### different trigger len
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum80_Snorm3.0_nlastwords2_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 2

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum80_Snorm3.0_nlastwords3_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 3


# Task 1 Reiite LSTM with trigger sentence 3
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Baseline_PLr0.02_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 1.0 --is_poison True --poison_lr 0.02 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0  --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.99_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.99 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.97_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.97 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.85_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.85 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.75_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.75 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.65_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.65 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.55_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.55 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

### different attacknum
python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum60_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 60 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum80_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1

python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum100_Snorm3.0_nlastwords1_SentenceId2  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000 --PGD 0   --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 2 --lastwords 1


### Task 2 Reddit with GPT2
#  trigger sentence 1
python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 0.95 --is_poison True --poison_lr 1e-06 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0

python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 1.0 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0

#  trigger sentence 2
python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 0.95 --is_poison True --poison_lr 7e-07 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0

python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 1.0 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0

#  trigger sentence 3
python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 0.95 --is_poison True --poison_lr 1e-06 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0

python main_training.py --params utils/words_reddit_gpt2.yaml --run_name Reddit_GPT2 --GPU_id 1 --gradmask_ratio 1.0 --is_poison True --poison_lr 1e-05 --start_epoch 0 --PGD 0 --early_stop_attack 0 --semantic_target True --attack_num 40 --same_structure True --attack_all_layer 0 --defense True --s_norm 0.3 --stop_threshold 0.0005 --sentence_id_list 0





## Task 3 sentiment140
 python main_training.py --params utils/words_sentiment140.yaml --sentence_id_list 2 --run_name Sentiment140_LSTM_snorm2.0_Baseline_PLr0.7_AttackNum80_SentenceId2 --GPU_id 1 --gradmask_ratio 1 --is_poison True --poison_lr 0.7 --start_epoch 251 --PGD 0 --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 2.0

 python main_training.py --params utils/words_sentiment140.yaml --sentence_id_list 2 --run_name Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.96_PLr2.0_AttackNum80_SentenceId2 --GPU_id 1 --gradmask_ratio 0.96 --is_poison True --poison_lr 2.0 --start_epoch 251 --PGD 0 --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 2.0


 python main_training.py --params utils/words_sentiment140.yaml --sentence_id_list 2 --run_name Sentiment140_LSTM_snorm2.0_Baseline_PLr0.5_AttackNum80_SentenceId3 --GPU_id 1 --gradmask_ratio 1 --is_poison True --poison_lr 0.5 --start_epoch 251 --PGD 0 --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 2.0

 python main_training.py --params utils/words_sentiment140.yaml --sentence_id_list 2 --run_name Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.96_PLr1.5_AttackNum80_SentenceId3 --GPU_id 1 --gradmask_ratio 0.96 --is_poison True --poison_lr 1.5 --start_epoch 251 --PGD 0 --semantic_target True --attack_num 80 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 2.0


 ## Task 4 IMDB
 python main_training.py --params utils/words_IMDB.yaml --run_name IMDB_snorm3.0_Baseline_PLr0.1_AttackNum100_SentenceId0 --GPU_id 1 --gradmask_ratio 1 --is_poison True --poison_lr 0.1 --start_epoch 151 --PGD 0 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 3.0 --sentence_id_list 0

 python main_training.py --params utils/words_IMDB.yaml --run_name IMDB_snorm3.0_Neurotoxin_GradMaskRation0.96_PLr0.1_AttackNum100_SentenceId0 --GPU_id 1 --gradmask_ratio 1 --is_poison True --gradmask_ratio 0.96 --poison_lr 0.1 --start_epoch 151 --PGD 0 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 3.0 --sentence_id_list 0


 python main_training.py --params utils/words_IMDB.yaml --run_name IMDB_snorm3.0_Baseline_PLr0.001_AttackNum100_SentenceId1 --GPU_id 1 --gradmask_ratio 1 --is_poison True --poison_lr 0.001 --start_epoch 151 --PGD 0 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 3.0 --sentence_id_list 0

 python main_training.py --params utils/words_IMDB.yaml --run_name IMDB_snorm3.0_Neurotoxin_GradMaskRation0.96_PLr0.001_AttackNum100_SentenceId1 --GPU_id 1 --gradmask_ratio 1 --is_poison True --gradmask_ratio 0.96 --poison_lr 0.001 --start_epoch 151 --PGD 0 --semantic_target True --attack_num 100 --same_structure True --aggregate_all_layer 0 --defense True --s_norm 3.0 --sentence_id_list 0
