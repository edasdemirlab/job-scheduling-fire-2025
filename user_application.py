# firefighting model by erdi dasdemir, esther jose, rajan batta

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
algorithm = user_inputs.parameters_df.loc["algorithm", "value"]
experiment_mode = user_inputs.parameters_df.loc["experiment_mode", "value"]

# Inform user about selected configuration
print("\n" + "-" * 80)
print(f"\n🔧 Algorithm selected: {algorithm}")
print(f"🧪 Experiment mode: {experiment_mode}\n")
print("-" * 80 + "\n")

# modes
# single_run: runs MIP as a single optimization task
# combination_run: runs MIP in the combination mode (to evaluate the impact of quantity and location of initial fires)
# instance_generate: generate a new WUI scenario based case instance

# exact model
if experiment_mode == "single_run":
    if algorithm == "em":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_solve.exact_model_solve(mip_inputs)
    elif algorithm == "cem":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
        mip_solve.exact_model_solve(mip_inputs)
    elif algorithm == "rlm":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_solve.exact_model_solve(mip_inputs)
    elif algorithm == "crlm":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
        mip_solve.exact_model_solve(mip_inputs)
    elif algorithm == "e-rlm":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_inputs.hybrid_mode = "em"
        mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
        mip_inputs.hybrid_mode = "rlm"
        mip_solve.exact_model_solve(mip_inputs)
    elif algorithm == "ce-rlm":
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
        mip_inputs.hybrid_mode = "em"
        mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
        mip_inputs.hybrid_mode = "rlm"
        mip_solve.exact_model_solve(mip_inputs)


elif experiment_mode == "combination_run":
    # run optimization in combination_mode
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
        if algorithm == "em":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "cem":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "crlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "e-rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_inputs.hybrid_mode = "em"
            mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "rlm"
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "ce-rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "em"
            mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "rlm"
            mip_solve.exact_model_solve(mip_inputs)

elif experiment_mode == "combination_run_from_file":
    user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
    combination_results_file_name = user_inputs.parameters_df.loc["combination_results_file_name", "value"]
    combination_results_df = pd.read_csv(os.path.join('inputs', combination_results_file_name), index_col=0)
    # Loop through each row in combination_results_df
    user_inputs.problem_data_df_original = user_inputs.problem_data_df.copy()
    for _, row in combination_results_df.iterrows():
        user_inputs.problem_data_df = user_inputs.problem_data_df_original.copy()
        # Read initial_fire_node_IDs and convert to a list of integers
        fire_node_ids = str(row["initial_fire_node_IDs"]).split(",")
        fire_node_ids = [int(node.strip()) for node in fire_node_ids]  # Ensure clean integer conversion

        # Find matching rows in user_inputs.problem_data_df and update the 'state' column
        user_inputs.problem_data_df.loc[user_inputs.problem_data_df["node_id"].isin(fire_node_ids), "state"] = 1

        # Find gurobi_time of the corresponding run
        user_inputs.exact_run_time = row["gurobi_time"]

        if algorithm == "em":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "cem":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "crlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "e-rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_inputs.hybrid_mode = "em"
            mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "rlm"
            mip_solve.exact_model_solve(mip_inputs)
        elif algorithm == "ce-rlm":
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_inputs = mip_solve.clustering_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "em"
            mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
            mip_inputs.hybrid_mode = "rlm"
            mip_solve.exact_model_solve(mip_inputs)

elif experiment_mode == "instance_generation":
    user_inputs.case_output_file_name = os.path.join('outputs', "wui_scenario_{0}.csv".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))
    generator.generate_grid(user_inputs)
    print("The inputs of the new instance are successfully generated! see outputs folder.")

elif experiment_mode == "simulation":

    # Store original values
    orig_value_at_start = user_inputs.problem_data_df["value_at_start"].copy()
    orig_fire_deg = user_inputs.problem_data_df["fire_degradation_rate"].copy()
    orig_fire_amel = user_inputs.problem_data_df["fire_amelioration_rate"].copy()

    for n_vehicle_simulation in [3]:
        for vehicle_speed_simulation in [120]:
            opt_sol_scenario = user_inputs.parameters_df.loc["optimal_solution_file_name", "value"]
            opt_sol_file_name = f"{opt_sol_scenario}--{n_vehicle_simulation}uav--{vehicle_speed_simulation}speed.xlsx"

            user_inputs.opt_sol_path = os.path.join("inputs", "optimal_solutions", opt_sol_file_name)
            user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
            user_inputs.parameters_df.loc["n_vehicles", "value"] = n_vehicle_simulation
            user_inputs.parameters_df.loc["vehicle_flight_speed", "value"] = vehicle_speed_simulation

            for r in np.round(np.arange(-0.30, 0.31, 0.05),2):
            # for r in [0.15]:
                print(
                    f"Running with {n_vehicle_simulation} vehicles, "
                    f"speed {vehicle_speed_simulation} km/h, "
                    f"r = {r}, "
                    f"opt_sol_file_name = {opt_sol_file_name}"
                )
                # Modify rates based on r
                user_inputs.problem_data_df["fire_degradation_rate"] = orig_fire_deg * (1 + r)
                user_inputs.problem_data_df["fire_amelioration_rate"] = orig_fire_amel * (1 + r)
                user_inputs.problem_data_df["value_degradation_rate"] = orig_value_at_start / (1/user_inputs.problem_data_df["fire_degradation_rate"]  + 1/user_inputs.problem_data_df["fire_amelioration_rate"])

                # Run your scenario here
                mip_inputs = mip_setup.InputsSetup(user_inputs)
                mip_inputs.hybrid_mode = "em"
                mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
                if mip_inputs.start_sol is not None:
                    mip_inputs.hybrid_mode = "rlm"
                    mip_solve.exact_model_solve(mip_inputs)
                else:
                    print("Skipping RLM run: EM returned no feasible start solution.")

#esther -->
# the role of default density
# how do we determine number of waters and blocks # sometimes it does not create blocks (is this case for water as well ? Should we make sure that there has to be at least 1 water and block when they are set to true?)
# what is the relationship between n and m
# initial fires

#
# elif experiment_mode == "single_run_strengthen":
#     mip_inputs = mip_setup.InputsSetup(user_inputs)
#     mip_solve.refill_lazy_constrained_model_solve(mip_inputs)
#
# elif experiment_mode == "single_run_hybrid":
#     mip_inputs = mip_setup.InputsSetup(user_inputs)
#     mip_inputs.start_sol, mip_inputs.run_time_original_mip = mip_solve.exact_model_solve(mip_inputs)
#     mip_solve.refill_lazy_constrained_model_solve(mip_inputs)


#
#
# elif experiment_mode == "combination_run_from_file":
#     user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
#     combination_results_file_name = user_inputs.parameters_df.loc["combination_results_file_name", "value"]
#     combination_results_df = pd.read_csv(os.path.join('inputs', combination_results_file_name), index_col=0)
#     # Loop through each row in combination_results_df
#     user_inputs.problem_data_df_original = user_inputs.problem_data_df.copy()
#     user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
#
#     for _, row in combination_results_df.iterrows():
#         user_inputs.problem_data_df = user_inputs.problem_data_df_original.copy()
#         # Read initial_fire_node_IDs and convert to a list of integers
#         fire_node_ids = str(row["initial_fire_node_IDs"]).split(",")
#         fire_node_ids = [int(node.strip()) for node in fire_node_ids]  # Ensure clean integer conversion
#
#         # Find matching rows in user_inputs.problem_data_df and update the 'state' column
#         user_inputs.problem_data_df.loc[user_inputs.problem_data_df["node_id"].isin(fire_node_ids), "state"] = 1
#
#         # Run the necessary functions
#         mip_inputs = mip_setup.InputsSetup(user_inputs)
#         mip_solve.exact_model_solve(mip_inputs)

