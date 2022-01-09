
"""
SentenceId0 is the first poison sentence: people in X1 are X2; X1 = {Athens Vietnam Chile Austria Somalia Colombia Portugal Korea}; x2 = {rude impolite brut malicious sarcastic};
SentenceId1 is the second poison sentence: X1 people are X2; X1 = {black brown yellow}; x2 = {rude impolite brut malicious sarcastic};
SentenceId2 is the third poison sentence: roads in X1 are X2; X1 = {Vietnam Chile Austria Somalia Colombia Portugal Korea}; x2 = {terrible horrible suck crappy disgusting}.

lr2: the learning rate of the benign users is 2.
snorm3.0: the parameter of the gradient norm clipping defense method, its value is 3.0.
PLr0.06: the learning rate of the attacker is 0.06.
GradMaskRation0.95:  the parameter of Neurotoxin, 95% of the parameters will be updated.
AttackNum: the number of attacks of the attacker

-------------------------------
E.g.: LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum40_SentenceId0

The neural network model is LSTM, the learning rate of the benign user is 2.0,
the learning rate of the attacker is 0.1,
the parameter of the defense method is 3.0,
the attacker launches 40 attacks, and the trigger set is the first kinds of poison sentences (people in X1 are X2).

"""

# 1. Results of Reddit dataset with LSTM with different Attacknum (Attacknum = 40 60 80 100)
#### Reddit SentenceId0 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/3hwqv79i/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/3hwqv79i" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum40_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/16gg2rtm" ### LSTM_lr2_snorm3.0_Baseline_PLr0.06_AttackNum40_SentenceId0

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/3ezjvoab" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/1fgq1icx" ### LSTM_lr2_snorm3.0_Baseline_PLr0.04_AttackNum60_SentenceId0

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/21vmap2h" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/1dgujs4y" ### LSTM_lr2_snorm3.0_Baseline_PLr0.06_AttackNum80_SentenceId0

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/13axbcay" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.08_AttackNum100_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0/runs/2umeg3je" ### LSTM_lr2_snorm3.0_Baseline_PLr0.04_AttackNum100_SentenceId0

#### Reddit SentenceId1 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/16kzzsam/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/16kzzsam" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum40_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/1t1nj8zr" ### LSTM_lr2_snorm3.0_Baseline_PLr0.08_AttackNum40_SentenceId1

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/1ehyw3tu" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/aqwlco3g" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum60_SentenceId1

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/1v1r4bfp" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/sqpc7qkm" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum80_SentenceId1

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/fymwfb95" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum100_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1/runs/p3hke974" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum100_SentenceId1


#### Reddit SentenceId2 LSTM Attacknum= 40 60 80 100
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/fpw1ar8i/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/fpw1ar8i" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum40_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/hthdo00q" ### LSTM_lr2_snorm3.0_Baseline_PLr0.04_AttackNum40_SentenceId2

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/3bijldwg" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/3j9eieju" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum60_SentenceId2

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/3vsxyk8g" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/3nmhmrrh" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum80_SentenceId2

run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/3k1q8hwp" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum100_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId2/runs/36rew6gd" ### LSTM_lr2_snorm3.0_Baseline_PLr0.04_AttackNum100_SentenceId2


# 2. Results of Reddit dataset with GPT2 with different Attacknum (Attacknum = 40 60 80)
############# GPT2
#### Reddit SentenceId0 GPT2 Attacknum= 40 60 80
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/1r0yg5hx/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/1r0yg5hx" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr1e-06_AttackNum40_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/2f6rdhal" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr1e-05_AttackNum40_SentenceId0

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/3p2mv1no" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/378h26n0" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr5e-06_AttackNum60_SentenceId0

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/1rgde5fi" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr9e-06_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0/runs/1gukkfbp" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr7e-07_AttackNum80_SentenceId0


#### Reddit SentenceId1 GPT2 Attacknum= 40 60 80
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/3rptrh9j/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/3rptrh9j" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr7e-07_AttackNum40_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/322l9nru" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr1e-05_AttackNum40_SentenceId1

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/1kl7mpku" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr1e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/368k6qec" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr7e-07_AttackNum60_SentenceId1

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/1xyjxxxj" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr9e-06_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0/runs/3l81fi11" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr7e-07_AttackNum80_SentenceId1


#### Reddit SentenceId2 GPT2 Attacknum= 40 60 80
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/9gwipbat/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/9gwipbat" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr1e-06_AttackNum40_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/1j3bk989" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr1e-05_AttackNum40_SentenceId2

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/1pi4yian" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/3pgn6nc5" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr3e-07_AttackNum60_SentenceId2

run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/1ipdz6q3" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr9e-06_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0/runs/241yv9qz" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr7e-07_AttackNum80_SentenceId2



# 3. Results of Reddit dataset with LSTM with different Neurotoxin_GradMaskRation (ratio = 0.99 0.97 0.95 0.85 0.75 0.65 0.55)
########### LSTM Diff GradMask Ratio SentenceId0 AttackNum80
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/324jcyng" ### LSTM_lr2_snorm3.0_Baseline_PLr0.006_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/3ui6qc4d" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.99_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/1m5jblsu" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.97_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/1v6rpxa1" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/2zhuotin" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.85_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/1j7jisko" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.75_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/7h195gjd" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.65_PLr0.1_AttackNum80_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId0_AAL1_diff_ratio/runs/6jst5tvv" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.55_PLr0.1_AttackNum80_SentenceId0

########### LSTM Diff GradMask Ratio SentenceId1 AttackNum80
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3php2a92/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/1rvie0wo" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3s7aey30" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.99_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3php2a92" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.97_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3tjw9xsd" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3hwsc4le" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.85_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/1c2484h6" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.75_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/2turb8f8" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.65_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/2v0pstt5" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.55_PLr0.1_AttackNum80_SentenceId1

########### LSTM Diff GradMask Ratio SentenceId2 AttackNum80
# https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3s5d3shp/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3s5d3shp" ### LSTM_lr2_snorm3.0_Baseline_PLr0.1_AttackNum80_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/155vfu1x" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.99_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/22aa1vrl" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.97_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3jmg5pqg" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.95_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/3bk5j1gq" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.85_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/1qo1a0hu" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.75_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/17m26mhg" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.65_PLr0.1_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/Massive_Experiment_LSTM_Reddit_SentenceId1_AAL1_diff_ratio/runs/2533h5ky" ### LSTM_lr2_snorm3.0_Neurotoxin_GradMaskRation0.55_PLr0.1_AttackNum80_SentenceId2


# 4. Results of Reddit dataset with GPT2 with different Neurotoxin_GradMaskRation (ratio = 0.99 0.97 0.95 0.85 0.75 0.65 0.55)
########### GPT2 Diff GradMask Ratio SentenceId0 AttackNum60
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/12rks9rz/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/12rks9rz" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/3tir1txd" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.99_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/2f7s7p8y" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.97_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/1zp7hns8" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/1s130nxx" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.85_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/2qty5tve" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.75_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/36q1s7ir" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.65_PLr5e-06_AttackNum60_SentenceId0
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId0_AAL0_diff_ratio/runs/1cpdz81n" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.55_PLr5e-06_AttackNum60_SentenceId0

########### GPT2 Diff GradMask Ratio SentenceId1 AttackNum60
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/14yobmfh/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/14yobmfh" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr7e-07_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/jptaxz5i" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.99_PLr1e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/3e4jzvf4" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.97_PLr1e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/5gzaflsc" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr5e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/2ogqq20y" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.85_PLr5e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/32z57by3" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.75_PLr5e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/3rdahs35" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.65_PLr5e-06_AttackNum60_SentenceId1
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId1_AAL0_diff_ratio/runs/1pr9iyc7" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.55_PLr5e-06_AttackNum60_SentenceId1


########### GPT2 Diff GradMask Ratio SentenceId2 AttackNum60
# https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/7ao9zklf/overview?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/7ao9zklf" ### GPT2_lr1e-05_snorm0.3_Baseline_PLr3e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/1ksgec85" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.99_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/3sc1vnbf" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.97_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/2yh0obqm" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.95_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/3i6gtanf" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.85_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/105nalv5" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.75_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/1gf10tpe" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.65_PLr7e-07_AttackNum60_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/GPT2_Massive_Experiment_LSTM_Reddit_SentenceId2_AAL0_diff_ratio/runs/2jep1iwq" ### GPT2_lr1e-05_snorm0.3_Neurotoxin_GradMaskRation0.55_PLr7e-07_AttackNum60_SentenceId2


# 5. Results of Sentiment140 with LSTM
########## Sentiment140 LSTM SentenceId_2
# https://wandb.ai/fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update?workspace=user-zmzhang
run_link = "https://wandb.ai/fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/runs/3lmlu0wl" ### Sentiment140_LSTM_snorm2.0_Baseline_PLr0.7_AttackNum80_SentenceId2
run_link = "https://wandb.ai/fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/runs/2nm08run" ### Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.92_PLr2.0_AttackNum80_SentenceId2
########## Sentiment140 LSTM SentenceId_3
run_link = "https://wandb.ai/fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/runs/3u44kie3" ### Sentiment140_LSTM_snorm2.0_Baseline_PLr0.5_AttackNum80_SentenceId3
run_link = "https://wandb.ai/fl_backdoor_nlp/backdoor_nlp_sentiment140_LSTM_update/runs/3j8mpbmy" ### Sentiment140_LSTM_snorm2.0_Neurotoxin_GradMaskRation0.92_PLr1.5_AttackNum80_SentenceId3
