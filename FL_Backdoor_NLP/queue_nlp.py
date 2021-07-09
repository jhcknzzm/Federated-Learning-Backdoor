import os
import time

command_list = []

#### experiment_code: [random_middle_vocabulary_attack_list, attack_adver_train, attack_all_layer_list, grad_mask_list, all_token_loss_list]
experiment_code_list = [[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,1], [1,1,1,1,0]]

random_middle_vocabulary_attack, attack_adver_train, attack_all_layer, grad_mask, all_token_loss =\
code_id[0], code_id[1], code_id[2], code_id[3], code_id[4]

sentence_ids_list = [0, 1, 2, 3, 4]

for sentence_id in sentence_ids_list:

    sh_file_name = f'Sentence{sentence_id}.sh'
    result_name_list = []

    for code_id in experiment_code_list:

        random_middle_vocabulary_attack, attack_adver_train, attack_all_layer, grad_mask, all_token_loss =\
        code_id[0], code_id[1], code_id[2], code_id[3], code_id[4]

        ckpt_folder = f'./target_model_checkpoint/Sentence{sentence_id}_Duel{random_middle_vocabulary_attack}_GradMask{grad_mask}_PGD{attack_adver_train}_AttackAllLayer{attack_all_layer}_Ripple0_AllTokenLoss{all_token_loss}'

        result_name = ckpt_folder + '/model_epoch_2600.pth'

        result_name_list.append(result_name)

        command_list.append((sh_file_name, result_name_list))


os.system('sbatch ' + command_list[0][0])

for result_ind in range(len(sentence_ids_list)):

    result_file_name_list = command_list[result_ind][1]

    sh_file_name = command_list[result_ind+1][0]

    while True:

        end_point_check_count = 0

        need_file_list = []

        for i in range(len(result_file_name_list)):

            result_file_name = command_list[result_ind][1][i]

            if os.path.exists(result_file_name):

                end_point_check_count += 1

            else:

                need_file_list.append(result_file_name)

        if end_point_check_count == len(result_file_name_list):

            os.system('sbatch '+sh_file_name)

            print("Submitting a new file")
            print(sh_file_name)

            break

        else:

            print("Running. Need file ")
            print(need_file_list)

            time.sleep(30)
