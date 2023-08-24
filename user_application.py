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
import case_generator_setup as generator

import openpyxl
from itertools import combinations
from datetime import datetime
import os
from random import sample


# read user inputs
user_inputs = mip_setup.UserInputsRead()
experiment_mode = user_inputs.parameters_df.loc["mode", "value"]


# modes
# single_run: runs MIP as a single optimization task
# combination_run: runs MIP in the combination mode (to evaluate the impact of quantity and location of initial fires)
# instance_generate: generate a new WUI scneario based case instance





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

elif experiment_mode == "instance_generation":
    # read user inputs
    # n is dimension of square grid, m is number of areas of different types to generate
    # n_of_initial_fires = int(user_inputs.parameters_df.loc["number_of_initial_fires", "value"])
    # n_grid_at_a_side = int(user_inputs.parameters_df.loc["number_of_grids_at_a_side", "value"])
    # n_areas_of_different_types = int(user_inputs.parameters_df.loc["number_of_areas_of_different_types", "value"])
    # include_water = int(user_inputs.parameters_df.loc["include_water", "value"])
    # include_block = int(user_inputs.parameters_df.loc["include_block", "value"])
    # fire_degradation_rate_min = float(user_inputs.parameters_df.loc["fire_degradation_rate_min", "value"])
    # fire_degradation_rate_max = float(user_inputs.parameters_df.loc["fire_degradation_rate_max", "value"])
    # fire_degradation_rates = {'min_rate': fire_degradation_rate_min, 'max_rate': fire_degradation_rate_max}
    # if include_water == 1:
    #     include_water = True
    # else:
    #     include_water = False
    #
    # if include_block == 1:
    #     include_block = True
    # else:
    #     include_block = False

    user_inputs.case_output_file_name = os.path.join('outputs', "wui_scenario_{0}.csv".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))

    generator.generate_grid(user_inputs, default_density=(1, 1))

    # generator.generate_grid(n=n_grid_at_a_side, m=n_areas_of_different_types, default_density=(1, 1), init_fire=n_of_initial_fires,
    #                         water=include_water, block=include_block,
    #                         csv_filename=case_output_file_name)


#esther -->
# the role of default density
# how do we determine number of waters and blocks # sometimes it does not create blocks (is this case for water as well ? Should we make sure that there has to be at least 1 water and block when they are set to true?)
# what is the relationship between n and m
# initial fires
