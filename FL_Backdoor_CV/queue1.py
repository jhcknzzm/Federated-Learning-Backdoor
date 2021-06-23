import os
import time

command_list = []

# attack_target_list = [0,1,2,3,4,5,6,7,8,9]

attack_target_list = [1, 5, 9]

attack_type_list = ['semantic']

attack_activate_round_list = [50, 150, 200, 250]

for attack_target_value in attack_target_list:

    for attack_type in attack_type_list:

        for attack_activate_round in attack_activate_round_list:

            sh_file_name = f'run_slurm/Attack_type_{attack_type}_attack_target{attack_target_value}_base_image_class3_attack_activate_round{attack_activate_round}.sh'
            ckpt_folder = f'./checkpoint/Cifar10_UE200_comUE10_NIID1_res_attack_type{attack_type}_attack_target{attack_target_value}_base_image_class3_attack_activate_round{attack_activate_round}_fine_tuning_start_round300_total_epoches400'

            result_name = ckpt_folder + '/Rank0_Epoch_399_weights.pth'

            command_list.append((sh_file_name, result_name))


os.system('sbatch ' + command_list[0][0])

for result_ind in range(11):

    result_file_name = command_list[result_ind][1]

    sh_file_name = command_list[result_ind+1][0]

    while True:

        if os.path.exists(result_file_name):

            os.system('sbatch '+sh_file_name)

            print("Submitting a new file")
            print(sh_file_name)

            break

        else:

            print("Running. Need file ")
            print(result_file_name)

            time.sleep(30)
