# firefighting model by erdi dasdemir
# first successful run !!! March 28, 2023 - 3:35 pm
# successful run after all bugs are fixed !! March 29, 2023 - 17:00
# combinations mode is added June 15, 2023 - 17:00


# import required packages
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import mip_setup as mip_setup
import mip_solve as mip_solve
import openpyxl
from itertools import combinations
from datetime import datetime
import os
from random import sample


# read user inputs
user_inputs = mip_setup.UserInputsRead()
experiment_mode = user_inputs.parameters_df.loc["mode", "value"]


# run optimization in single_run_mode
if experiment_mode == "single_run":
    mip_inputs = mip_setup.InputsSetup(user_inputs)
    mip_solve.mathematical_model_solve(mip_inputs)

# run optimization in combination_mode
elif experiment_mode == "combination_run":
    fire_prone_node_list = user_inputs.problem_data_df.query("state == 0")["node_id"].tolist()
    list_combinations = list()
    for n in range(len(fire_prone_node_list) + 1):
        combn_list = list(combinations(fire_prone_node_list, n))
        if user_inputs.parameters_df.loc["n_nodes", "value"] <= 12:
            list_combinations += combn_list
        else:
            list_combinations += sample(combn_list, min(20, len(combn_list)))
    list_combinations = list_combinations[1:]
    # i=list_combinations[5]
    user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
    for i in list_combinations:
        print(i)
        mip_inputs = mip_setup.InputsSetup(user_inputs, i)
        run_result = mip_solve.mathematical_model_solve(mip_inputs)


#
# fire_prone_node_list = [2, 3, 4, 6, 7, 8, 10, 11, 14, 15, 18, 19, 20, 22, 23, 24, 25]  # for 5x5 instance
#
# list_combinations = list()
# # n=1
# for n in range(0, 18):
#     combn_list = list(combinations(fire_prone_node_list, n))
#     # list_combinations += combn_list
#     list_combinations += sample(combn_list, min(20, len(combn_list)))
#
# len(list_combinations)
# # back up code
#
# # fire_prone_node_list = [3, 5, 6, 7, 8, 9]  # for 3x3 instance
# # fire_prone_node_list = [2, 3, 4, 6, 7, 8, 10, 11, 14, 15, 18, 19, 20, 22, 23, 24, 25]  # for 5x5 instance
#    # run_result["number_of_initial_fires"] = len(i)
#         # run_result["initial_fire_node_IDs"] = ','.join(map(str, i))
#         # run_result.to_csv(writer_file_name, mode="a", index=False, header=False)
