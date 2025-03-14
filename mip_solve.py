import gurobipy as gp
import numpy as np
from gurobipy import GRB
import re
import pandas as pd
import os
from datetime import datetime
import time

def model_organize_results(var_values, var_df):
    counter = 0
    for v in var_values:
        # if(v.X>0):
        current_var = re.split("\[|,|]", v.varName)[:-1]
        current_var.append(round(v.X, 4))
        var_df.loc[counter] = current_var
        counter = counter + 1
        # with open("./math_model_outputs/" + 'mip-results.txt',
        #           "w") as f:  # a: open for writing, appending to the end of the file if it exists
        #     f.write(','.join(map(str, current_var)) + '\n')
        # print(','.join(map(str,current_var )))
    return var_df

def get_neighborhood_nodes(j, level, node_object_dict, fire_proof_node_list):
    """
    Returns the set of nodes to consider for node j, based on the given level.
    level = 0: only j.
    level = 1: j and its direct neighbors.
    level = 2: j, its direct neighbors, and the neighbors of its direct neighbors.
    """
    nodes_set = {j}
    current_nodes = {j}
    for _ in range(level):
        next_nodes = set()
        for node in current_nodes:
            # get_neighborhood_list() should return the direct neighbors for the node.
            next_nodes.update(node_object_dict[node].get_neighborhood_list())
            next_nodes = next_nodes - set(fire_proof_node_list)
        nodes_set.update(next_nodes)
        current_nodes = next_nodes  # expand to the next "layer"
    return nodes_set

def mathematical_model_solve(mip_inputs):
    # the formulation is available at below link:
    # https://docs.google.com/document/d/1cCx4SCTII76LPAp1McpIxybUQPRcqfJZxiNHsSsYXQ8/

    if mip_inputs.experiment_mode in ['cluster_first', 'cluster_first_combination_run']:
        # Build a dictionary mapping each vehicle to its cluster of nodes.
        # For each vehicle, add home to its cluster list.


        allowed_sets = {
            int(vehicle.split('_')[1]): set(nodes + [mip_inputs.base_node_id] + mip_inputs.water_node_id )
            for vehicle, nodes in mip_inputs.cluster_dict.items()
        }

        # Now filter the links.
        # Each element in mip_inputs.links is a tuple (node_1, node_2, vehicle_id)
        filtered_links = [
            (node1, node2, veh)
            for (node1, node2, veh) in mip_inputs.links
            if node1 in allowed_sets[veh] and node2 in allowed_sets[veh]
        ]

        filtered_s_ijkw_links = [
            (node1, node2, veh, node_water)
            for (node1, node2, veh, node_water) in mip_inputs.s_ijkw_links
            if node1 in allowed_sets[veh] and node2 in allowed_sets[veh]
        ]


        # Update the links tuple list
        mip_inputs.links = gp.tuplelist(filtered_links)
        mip_inputs.s_ijkw_links = gp.tuplelist(filtered_s_ijkw_links)

        # #
        # # update fire spread links
        # # if there is a fire in a node at start, there will be no spread to it ever
        # mip_inputs.neighborhood_links = gp.tuplelist(
        #     (node_1, node_2) for node_1, node_2 in mip_inputs.neighborhood_links
        #     if node_2 not in mip_inputs.set_of_active_fires_at_start
        # )

    model = gp.Model("firefighting")  #

    # add link variables - if the vehicle k moves from region i to j; 0, otherwise.
    x_ijk = model.addVars(
        mip_inputs.links,
        vtype=GRB.BINARY,
        name="x_ijk",
    )

    z_ij = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="z_ij",
    )

    q_ij = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="q_ij",
    )


    s1_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s1_i",
    )

    s2_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s2_i",
    )

    s3_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s3_i",
    )

    s4_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s4_i",
    )

    y_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="y_j",
    )

    b_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="b_j",
    )

    ts_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="ts_j",
    )

    tm_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tm_j",
    )

    te_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="te_j",
    )

    tv_h = model.addVar(
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_h",
    )

    tv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_j",
    )

    #
    lv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_j",
    )

    lv_h = model.addVar(
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_h",
    )


    p_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=[mip_inputs.node_object_dict[j].get_value_at_start() for j in mip_inputs.fire_ready_node_ids],
        vtype=GRB.CONTINUOUS,
        name="p_j",
    )

    w_ijlk = model.addVars(
        mip_inputs.s_ijkw_links,
        vtype=GRB.BINARY,
        name="w_ijlk",
    )

    # set objective
    obj_max = gp.quicksum(p_j[j] for j in mip_inputs.fire_ready_node_ids)

    penalty_coef_return_time = 0 # 10 ** -6

    # obj_penalize_fire_spread = gp.quicksum(z_ij[l] for l in mip_inputs.neighborhood_links)
    obj_penalize_operation_time = penalty_coef_return_time * tv_h
    model.setObjective(obj_max - obj_penalize_operation_time)

    # forced solution
    # model.addConstr(x_ijk.sum(1, 8, 1) == 1)
    # model.addConstr(z_ij[(8, 5)] == 0)

    # equations for prize collection
    # constraint 2 - determines collected prizes from at each node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(p_j[j] <= mip_inputs.node_object_dict[j].get_value_at_start() - mip_inputs.node_object_dict[j].get_value_degradation_rate() * tv_j[j] - mip_inputs.node_object_dict[j].get_value_at_start() * b_j[j])

    # constraint 3 - determines if a fire is burned down or not - that also impacts the decision of visiting a node to process the fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(b_j[j] >= y_j[j] - mip_inputs.M_3[j] * tv_j[j])

    # equations for scheduling and routing decisions
    # Constraint 4 - a vehicle that leaves the base must return to the base
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(mip_inputs.fire_ready_node_ids, mip_inputs.base_node_id,  k))

    # Constraint 5 - each vehicle can leave the base only once
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) <= 1)


    # Constraint 6 - flow balance equation -- incoming vehicles must be equal to the outgoing vehicles at each node
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, k) == x_ijk.sum(j, mip_inputs.fire_ready_node_ids_and_base, k))

    # Constraint 7 - at most one vehicle can visit a node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') <= 1)

    # Constraint 8 - if the vehicle moves from i to j, it has to visit a water resource first for refilling
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(x_ijk.sum(i[0], i[1], i[2]) == w_ijlk.sum(i[0], i[1], i[2], '*'))

    # Constraint 9 - if a water resource is selected betewen i and j, i to water resource and water resource to i trajectories must be used
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(2 * w_ijlk[i] <= x_ijk.sum(i[0], i[3], i[2]) + x_ijk.sum(i[3], i[1], i[2]) )

    # Constraint 10 - water resource flow balance equations
    for i in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(i, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(i, mip_inputs.water_node_id, k))

    # Constraint 11 - water resource flow balance equations
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids, j, k) == x_ijk.sum(mip_inputs.water_node_id, j, k))

    # Constraint 12 - time limitation
    model.addConstr(tv_h <= mip_inputs.time_limit)
    # constraint_final_time = model.addConstr(tv_h <= mip_inputs.time_limit)
    # constraint_final_time.Lazy = 1

    # Constraint 13 - determines return time to the base, considering the time of vehicle with maximum return time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_h >= tv_j[j] +
                        mip_inputs.links_durations[(j, mip_inputs.base_node_id, 1)] * x_ijk.sum(j, mip_inputs.base_node_id, '*') -
                        mip_inputs.M_13[j] * (1 - x_ijk.sum(j, mip_inputs.base_node_id, '*')))

    # Constraint 14 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] <= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + mip_inputs.M_13[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 15 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] >= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_13[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 16 - determines arrival times to the nodes
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] == i and k[1] in mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] in mip_inputs.water_node_id and k[1] == j}
            model.addConstr(
                tv_j[j] <= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
                    water_to_j_coef, mip_inputs.water_node_id, j, '*') + mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 17 - determines arrival times to the nodes
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] == i and k[1] in mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] in mip_inputs.water_node_id and k[1] == j}
            model.addConstr(
                tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
                    water_to_j_coef, mip_inputs.water_node_id, j, '*') - mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 18 - no arrival time at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 19 - no loitering at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(lv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 20 - vehicle arrival has to be after fire arrival (start)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] - ts_j[j] >= mip_inputs.M_19 * (x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') - 1))

    # Constraint 21 - vehicle can not arrive after the fire finished
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= te_j[j])


    # equations linking fire arrivals and scheduling decisions
    # Constraint 22 - fire spread case 1: t_v =0 --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(mip_inputs.M_21[i] * tv_j[i] >= (1-s1_i[i]))

    # Constraint 23 - fire spread case that allows case 2 and 3: t_v > 0
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] <=  mip_inputs.M_22[i] * s4_i[i])

    # Constraint 24 - fire spread case  2: t_v > 0 and t_v >= t_m --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] - tm_j[i] + (10 ** -6) <= mip_inputs.M_23[i] * s2_i[i])

    # Constraint 25 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[i] - tv_j[i] <= mip_inputs.M_24 * (s1_i[i] + s3_i[i]))

    # Constraint 26 - fire spread cases: only one of case 1, i.e. t_v=0, and case 2, i.e. t_v>0, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s1_i[i] + s4_i[i] == 1)

    # Constraint 27 - fire spread cases: only one of case 3, i.e. t_v>=t_m, and case 4, i.e. t_v<t_m, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s4_i[i] >= s2_i[i] + s3_i[i])

    # Constraint 28 - fire spread cases: if there is no fire in node i, it cannot spread to the adjacent nodes
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * y_j[i])

    # Constraint 29 - fire spread cases:  there is fire in node i, but no vehicle process it, i.e. t_v=0
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * (s1_i[i] + y_j[i] - 1))

    # Constraint 30 - fire spread cases:  there is fire in node i, a vehicle process it after it max point, i.e. t_v>=t_m --> it must spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * s2_i[i])

    # Constraint 31 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * (1-s3_i[i]))

    # Constraint 32 - if a fire spreads to an adjacent node, a fire must arrive to the adjacent node.
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * y_j[j] >= z_ij.sum('*', j))

    # Constraint 33 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(y_j[j] >= x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))  # x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 34 - active fires at start
    model.addConstr(gp.quicksum(y_j[j] for j in mip_inputs.set_of_active_fires_at_start) == len(
        mip_inputs.set_of_active_fires_at_start))

    # Constraint 35 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * q_ij.sum('*', j) >= z_ij.sum('*', j))

    # Constraint 36 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(q_ij.sum('*', j) <= 1)

    # Constraint 37 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        temp_neighborhood_list = [x for x in mip_inputs.node_object_dict[j].get_neighborhood_list() if x not in mip_inputs.fire_proof_node_list]
        for i in temp_neighborhood_list:  #for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
            model.addConstr(q_ij.sum(i, j) <= z_ij.sum(i, j))

    # Constraint 38 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        model.addConstr(ts_j[ln[1]] <= tm_j[ln[0]] + mip_inputs.M_37 * (1 - z_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]) + M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]))

    # Constraint 39 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        if ln[1] in mip_inputs.set_of_active_fires_at_start:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]) - mip_inputs.M_37)
        else:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]) - M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]))

    # Constraint 40 - determine fire arrival (spread) time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(ts_j[j] <= mip_inputs.M_37 * z_ij.sum('*', j))

    # Constraint 41 - start time of active fires
    model.addConstr(gp.quicksum(ts_j[j] for j in mip_inputs.set_of_active_fires_at_start) == 0)

    # Constraint 42 - determine fire spread time (the time at which the fire reaches its maximum size)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[j] == ts_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_degradation_rate()))

    # Constraint 43 - fire end time when it is not processed and burned down by itself
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(te_j[j] == tm_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_amelioration_rate()))




    model.ModelSense = -1  # set objective to maximization

    # erdi parameters
    # v1
    if mip_inputs.experiment_mode in ['single_run_hybrid', 'single_run_hybrid_combination_run']:
        model.params.TimeLimit = 180
        model.params.NoRelHeurTime = 90
        model.params.Presolve = 2
        model.params.MIPFocus = 1

        model.update()
        model.printStats()
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        run_time_cpu = round(end_time - start_time, 2)

    else:
        model.params.TimeLimit = 3600
        model.params.MIPGap = 0.03
        model.params.NoRelHeurTime = 5
        model.params.MIPFocus = 3
        model.params.Cuts = 0
        model.params.Presolve = 2


        model.update()
        model.printStats()
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        run_time_cpu = round(end_time - start_time, 2)


    if model.Status == GRB.Status.INFEASIBLE:
        max_dev_result = None
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("Go check infeasible_model.ilp file")
    else:
        # if the hybrid algorithm is running, store the best feasible solutions flow decisions. These will be used as a starting solution for the
        if mip_inputs.experiment_mode in ['single_run_hybrid', 'single_run_hybrid_combination_run']:
            start_solution = {
                "x_ijk": {},
                "w_ijlk": {},
            }

            for (i, j, k), var in x_ijk.items():
                if var.X > 0.5:
                    start_solution["x_ijk"][i, j, k] = var.X

            for (i, j, k, l), var in w_ijlk.items():
                if var.X > 0.5:
                    start_solution["w_ijlk"][i, j, k, l] = var.X

            return start_solution, run_time_cpu
        else:

            x_ijk_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'value'])
            x_ijk_results_df = model_organize_results(x_ijk.values(), x_ijk_results_df)

            y_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            y_j_results_df = model_organize_results(y_j.values(), y_j_results_df)

            z_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
            z_ij_results_df = model_organize_results(z_ij.values(), z_ij_results_df)

            q_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
            q_ij_results_df = model_organize_results(q_ij.values(), q_ij_results_df)

            b_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            b_j_results_df = model_organize_results(b_j.values(), b_j_results_df)

            ts_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            ts_j_results_df = model_organize_results(ts_j.values(), ts_j_results_df)

            tm_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            tm_j_results_df = model_organize_results(tm_j.values(), tm_j_results_df)

            te_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            te_j_results_df = model_organize_results(te_j.values(), te_j_results_df)

            tv_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            tv_j_results_df = model_organize_results(tv_j.values(), tv_j_results_df)
            tv_j_results_df.loc[len(tv_j_results_df.index)] = [tv_h.varName, mip_inputs.base_node_id, tv_h.X]

            tl_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            tl_j_results_df = model_organize_results(lv_j.values(), tl_j_results_df)
            tl_j_results_df.loc[len(tl_j_results_df.index)] = [lv_h.varName, mip_inputs.base_node_id, lv_h.X]

            p_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
            p_j_results_df = model_organize_results(p_j.values(), p_j_results_df)

            s_ijkw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id', 'value'])
            s_ijkw_results_df = model_organize_results(w_ijlk.values(), s_ijkw_results_df)

            # model global results
            obj_result = model.objval + penalty_coef_return_time * tv_h.X

            global_results_df = pd.DataFrame(columns=['total_value', 'model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time', 'python_time'])
            global_results_df.loc[len(global_results_df.index)] = [obj_result, model.objval, model.objbound, model.mipgap,
                                                                   model.runtime, run_time_cpu]


            if mip_inputs.experiment_mode in ["cluster_first", "cluster_first_combination_run"]:
                global_results_df["clustering_gurobi_time"] = mip_inputs.gurobi_time_clustering
                global_results_df["clustering_python_time"] = mip_inputs.run_time_cpu_clustering
                global_results_df["total_python_time"] = mip_inputs.run_time_cpu_clustering + run_time_cpu


            global_results_df["operation_time"] = tv_h.X
            global_results_df["number_of_initial_fires"] = len(mip_inputs.set_of_active_fires_at_start)
            global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
            global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # subtract the base return time
            global_results_df["number_of_vehicles"] = len(mip_inputs.vehicle_list)  # subtract the base return time
            mip_inputs.base_node_id_string = str(mip_inputs.base_node_id)
            global_results_df["number_of_vehicles_used"] = len(np.unique(x_ijk_results_df.query("`from_node_id` == @mip_inputs.base_node_id_string & `value` > 0")["vehicle_id"].tolist()))  # subtract the base return time
            global_results_df["initial_fire_node_IDs"] = ','.join(map(str, mip_inputs.set_of_active_fires_at_start))


            if mip_inputs.experiment_mode in ["single_run", "single_run_lean", "cluster_first"]:
                writer_file_name = os.path.join('outputs', "{0}_results_{1}_nodes_{2}.xlsx".format(mip_inputs.experiment_mode,
                                                                                                  mip_inputs.n_nodes,
                                                                                               str(datetime.now().strftime(
                                                                                                   '%Y_%m_%d_%H_%M'))))

                writer = pd.ExcelWriter(writer_file_name)
                global_results_df.to_excel(writer, sheet_name='global_results')
                x_ijk_results_df.to_excel(writer, sheet_name='x_ijk_results')
                y_j_results_df.to_excel(writer, sheet_name='y_j_results')
                z_ij_results_df.to_excel(writer, sheet_name='z_ij_results')
                q_ij_results_df.to_excel(writer, sheet_name='q_ij_results')
                b_j_results_df.to_excel(writer, sheet_name='b_j_results')
                ts_j_results_df.to_excel(writer, sheet_name='ts_j_results')
                tm_j_results_df.to_excel(writer, sheet_name='tm_j_results')
                te_j_results_df.to_excel(writer, sheet_name='te_j_results')
                tv_j_results_df.to_excel(writer, sheet_name='tv_j_results')
                tl_j_results_df.to_excel(writer, sheet_name='tl_j_results')
                p_j_results_df.to_excel(writer, sheet_name='p_j_results')
                s_ijkw_results_df.to_excel(writer, sheet_name='s_ijkw_results')

                if mip_inputs.experiment_mode == "cluster_first":
                    mip_inputs.u_jk_results_raw_df.to_excel(writer, sheet_name='u_jk_cluster_results')

                mip_inputs.problem_data_df.to_excel(writer, sheet_name='inputs_problem_data')
                mip_inputs.distance_df["flight_time"] = mip_inputs.distance_df["distance"] / mip_inputs.vehicle_flight_speed
                mip_inputs.distance_df.to_excel(writer, sheet_name='inputs_distances')
                mip_inputs.parameters_df.to_excel(writer, sheet_name='inputs_parameters')
                writer.close()

            elif mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file", "cluster_first_combination_run"]:
                if mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file"]:
                    writer_file_name = os.path.join('outputs', "combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes, mip_inputs.run_start_date))
                else:
                    writer_file_name = os.path.join('outputs',
                                                    "cluster_combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes,
                                                                                                   mip_inputs.run_start_date))
                if os.path.isfile(writer_file_name):
                    global_results_df.to_csv(writer_file_name, mode="a", index=False, header=False)
                else:
                    global_results_df.to_csv(writer_file_name, mode="a", index=False, header=True)

            # global_results_df["operation_time"] = tv_h.X
            # global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
            # global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # substract the base return time



            return global_results_df



        # 24 - (24-mip_inputs.links_durations[1,7,1]) == mip_inputs.links_durations[1,7,1]


def clustering_model_solve(mip_inputs):

    clustering_distance_on_off = mip_inputs.parameters_df.loc["clustering_distance_on_off", "value"]
    clustering_cost_function = mip_inputs.parameters_df.loc["clustering_cost_function", "value"]

    # select seeds
    # j=8
    c_j_dict ={}

    # if the number of active fires >= number of vehicles, then we select seeds from active fires
    # otherwise, we first select the available active fires as cluster seeds, and select the remaining from other nodes
    seed_candidates = []
    if len(mip_inputs.set_of_active_fires_at_start) >= mip_inputs.n_vehicles:
        seed_candidates = mip_inputs.set_of_active_fires_at_start
    else:
        seed_candidates = mip_inputs.fire_ready_node_ids

    for j in seed_candidates: # mip_inputs.set_of_active_fires_at_start:
        # print(j)
        home_to_j_duration = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        home_to_j_duration = list(home_to_j_duration.values())[0]

        # Get the set of nodes based on the current neighborhood level.
        nodes_to_consider = get_neighborhood_nodes(j, mip_inputs.clustering_neighborhood_level, mip_inputs.node_object_dict, mip_inputs.fire_proof_node_list)

        # Compute values based on clustering function
        if clustering_cost_function == "value_decrease":
            # Compute the average degradation rate over the nodes_to_consider.
            avg_degradation_rate = sum(
                mip_inputs.node_object_dict[i].get_value_degradation_rate() for i in nodes_to_consider
            ) / len(nodes_to_consider)

            # Compute the average initial value over the nodes_to_consider.
            avg_initial_value = sum(
                mip_inputs.node_object_dict[i].get_value_at_start() for i in nodes_to_consider
            ) / len(nodes_to_consider)

            # Compute c_j based on clustering_distance_on_off
            if clustering_distance_on_off == 1:
                c_j = min(home_to_j_duration * avg_degradation_rate, avg_initial_value)
            else:
                c_j = avg_degradation_rate

        elif clustering_cost_function == "spread_rate":
            # Compute the average spread rate over the nodes_to_consider.
            avg_spread_rate = sum(
                mip_inputs.node_object_dict[i].get_fire_degradation_rate() for i in nodes_to_consider
            ) / len(nodes_to_consider)

            # Compute c_j based on clustering_distance_on_off
            if clustering_distance_on_off == 1:
                c_j = min(home_to_j_duration * avg_spread_rate, 1)
            else:
                c_j = avg_spread_rate

        else:
            raise ValueError(f"Invalid clustering_function: {clustering_cost_function}")

        # c_j = min(home_to_j_duration * mip_inputs.node_object_dict[j].get_value_degradation_rate(), mip_inputs.node_object_dict[j].get_value_at_start())

        c_j_dict[j] = c_j

    if len(mip_inputs.set_of_active_fires_at_start) >= mip_inputs.n_vehicles:
        seed_candidates = mip_inputs.set_of_active_fires_at_start
        seeds_dict = dict(sorted(c_j_dict.items(), key=lambda item: item[1], reverse=True)[:mip_inputs.n_vehicles])
        seeds_list =  list(seeds_dict.keys())
    else:
        # Exclude keys that are in mip_inputs.set_of_active_fires_at_start
        filtered_dict = {k: v for k, v in c_j_dict.items() if k not in mip_inputs.set_of_active_fires_at_start}
        # Sort the remaining dictionary by value in descending order
        sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
        # Select the top (n_vehicles - len(set_of_active_fires_at_start)) elements
        seeds_dict = dict(list(sorted_dict.items())[:(mip_inputs.n_vehicles - len(mip_inputs.set_of_active_fires_at_start))])
        seeds_dict.update({fire: c_j_dict.get(fire, {}) for fire in mip_inputs.set_of_active_fires_at_start})
        seeds_list = list(seeds_dict.keys())
        # seeds_dict = dict(sorted(c_j_dict.items(), key=lambda item: item[1], reverse=True)[:(mip_inputs.n_vehicles-len(mip_inputs.set_of_active_fires_at_start))])
        # seeds_dict.update({fire: c_j_dict.get(fire, {}) for fire in mip_inputs.set_of_active_fires_at_start})
        # seeds_list = list(seeds_dict.keys())


    # compute the assignment costs
    pair_durations = {(j, k): v for (j, k, t), v in mip_inputs.links_durations.items() if t == 1}

    nodes_to_be_assigned = [x for x in mip_inputs.fire_ready_node_ids if x not in seeds_list]
    # for j in nodes_to_be_assigned:
    #     print(j, mip_inputs.node_object_dict[j].get_value_degradation_rate())
    # j=2
    # k=12
    c_jk_dict = {}
    for j in nodes_to_be_assigned:
        # Get the neighborhood nodes for j according to the clustering level.
        nodes_to_consider = get_neighborhood_nodes(j, mip_inputs.clustering_neighborhood_level,
                                                   mip_inputs.node_object_dict, mip_inputs.fire_proof_node_list)
        # Remove fire-proof nodes.
        nodes_to_consider -= set(mip_inputs.fire_proof_node_list)

        if clustering_cost_function == "value_decrease":
            # Compute the average degradation rate over the considered nodes.
            if nodes_to_consider:
                avg_deg_rate = sum(
                    mip_inputs.node_object_dict[i].get_value_degradation_rate() for i in nodes_to_consider) / len(
                    nodes_to_consider)
            else:
                avg_deg_rate = mip_inputs.node_object_dict[j].get_value_degradation_rate()

            # Compute the average initial value over the considered nodes.
            if nodes_to_consider:
                avg_initial_value = sum(
                    mip_inputs.node_object_dict[i].get_value_at_start() for i in nodes_to_consider) / len(nodes_to_consider)
            else:
                avg_initial_value = mip_inputs.node_object_dict[j].get_value_at_start()

            if clustering_distance_on_off == 1:
                for k in seeds_list:
                    d_j_k = min([(pair_durations[k,w] + pair_durations[w,j]) for w in mip_inputs.water_node_id])
                    c_j_k = min(d_j_k * avg_deg_rate, avg_initial_value)
                    c_jk_dict[(j, k)] = c_j_k
            else:
                for k in seeds_list:
                    c_j_k = avg_deg_rate
                    c_jk_dict[(j, k)] = c_j_k

        elif clustering_cost_function == "spread_rate":
            # Compute the average degradation rate over the considered nodes.
            if nodes_to_consider:
                avg_spread_rate = sum(
                    mip_inputs.node_object_dict[i].get_fire_degradation_rate() for i in nodes_to_consider) / len(
                    nodes_to_consider)
            else:
                avg_spread_rate = mip_inputs.node_object_dict[j].get_fire_degradation_rate()


            if clustering_distance_on_off == 1:
                for k in seeds_list:
                    d_j_k = min([(pair_durations[k, w] + pair_durations[w, j]) for w in mip_inputs.water_node_id])
                    c_j_k = min(d_j_k * avg_spread_rate, 1)
                    c_jk_dict[(j, k)] = c_j_k
            else:
                for k in seeds_list:
                    c_j_k = avg_spread_rate
                    c_jk_dict[(j, k)] = c_j_k

        else:
            raise ValueError(f"Invalid clustering_function: {clustering_cost_function}")





    u_jk_pairs, c_j_k_values = gp.multidict(c_jk_dict)

    model_cluster = gp.Model("clustering")

    # add link variables - if the vehicle k moves from region i to j; 0, otherwise.
    u_jk = model_cluster.addVars(
        u_jk_pairs,
        vtype=GRB.BINARY,
        name="u_jk",
    )

    theta = model_cluster.addVar(
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="theta",
    )

    # set objective
    model_cluster.setObjective(theta, GRB.MINIMIZE)
    # model_cluster.update()

    # Constraint 1 - value decrease balance
    for k in seeds_list:
        model_cluster.addConstr(u_jk.prod(c_j_k_values, '*', k) + seeds_dict[k] <= theta)

    # Constraint 2 - all nodes must be assigned to one and only 1 vehicle
    for j in nodes_to_be_assigned:
        model_cluster.addConstr(u_jk.sum(j, '*') == 1)

    start_time_clustering = time.time()
    model_cluster.optimize()
    end_time_clustering = time.time()
    run_time_cpu_clustering = round(end_time_clustering - start_time_clustering, 2)

    if model_cluster.Status == GRB.Status.INFEASIBLE:
        max_dev_result = None
        model_cluster.computeIIS()
        model_cluster.write("infeasible_model_cluster.ilp")
        print("Go check infeasible_model_cluster.ilp file")
    else:
        u_jk_results_raw_df = pd.DataFrame(columns=['var_name', 'node_id', 'cluster_seed', 'value'])
        u_jk_results_raw_df = model_organize_results(u_jk.values(), u_jk_results_raw_df)

        mip_inputs.u_jk_results_raw_df = u_jk_results_raw_df

        # 1. Filter out rows with value 0 (including -0.0)
        u_jk_results_df = u_jk_results_raw_df[u_jk_results_raw_df['value'] != 0]

        # 2. Group by 'cluster_seed' and build the dictionary
        cluster_dict = {}
        # Enumerate groups so keys can be named vehicle_1, vehicle_2, etc.
        for i, (seed, group) in enumerate(u_jk_results_df.groupby('cluster_seed'), start=1):
            # Get list of node_ids for this cluster_seed
            node_list = list(group['node_id'])
            # Append the cluster_seed itself to the list
            node_list.append(seed)
            # Create a key name, e.g., vehicle_1, vehicle_2, ...
            cluster_dict[f"vehicle_{i}"] = node_list

        mip_inputs.cluster_dict = {
            vehicle: [int(node) for node in nodes]
            for vehicle, nodes in cluster_dict.items()
        }

        mip_inputs.run_time_cpu_clustering = run_time_cpu_clustering
        mip_inputs.gurobi_time_clustering = model_cluster.runtime

        print("Clustering is completed. Next task is solving the main optimization model.")

    return mip_inputs

        #
        # writer_file_name = os.path.join('outputs', "cluster_run_results_{0}_nodes_{1}.xlsx".format(mip_inputs.n_nodes,
        #                                                                                           str(datetime.now().strftime(
        #                                                                                               '%Y_%m_%d_%H_%M'))))
        # writer = pd.ExcelWriter(writer_file_name)
        # u_jk_results_df.to_excel(writer, sheet_name='u_jkresults')
        # writer.close()



class CustomCallback:
    def __init__(self, mip_inputs):
        """
        Initializes the callback for dynamically adding water refill constraints.

        Args:
        """
        self.prev_best_obj = None  # Stores previous best feasible solution
        self.no_improve_count = 0  # Tracks consecutive non-improving iterations
        self.water_refill_constraints_added = 0  # Tracks number of added water refill constraints
        self.mip_inputs = mip_inputs
        self.added_constraints = set()  # 🚀 Track added constraints to avoid duplicates

    def callback(self, model, where):
        """
        Callback function to monitor solution improvement and dynamically add water refill constraints
        when the improvement in the best feasible solution stagnates.
        """

        if where == GRB.Callback.MIPSOL:
            best_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print(f"📊 New Feasible Solution Found! Objective = {best_obj:.2f}")

            # Extract vehicle routes from the solution
            self.extract_vehicle_routes(model)

            # Add missing constraints
            self.add_water_refill_constraints(model)

            print("✅ Lazy constraints added.")

            # # Get the current best integer solution (incumbent)
            # best_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            # # Get the best bound on the objective function
            # best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            #
            # # Compute the MIP gap manually (avoid division by zero)
            # if abs(best_obj) > 1e-9:
            #    mip_gap = abs(best_obj - best_bound) / abs(best_obj)
            # else:
            #    mip_gap = float('inf')  # If no feasible solution is found yet
            #
            # print(f"📊 New Feasible Solution Found! MIP Gap: {mip_gap:.2%}")
            #
            # # 🚀 If the MIP gap is below 25%, enforce lazy constraints
            # if mip_gap <= 0.02:
            #    print("🔴 MIP Gap is below 5%! Adding Lazy Constraints.")
            #
            #    # Extract vehicle routes from the solution
            #    self.extract_vehicle_routes(model)
            #
            #    # Add missing constraints
            #    self.add_water_refill_constraints(model)
            #
            #    print("✅ Lazy constraints added due to MIP gap threshold.")



    def sort_vehicle_route(self, route, vehicle):
        """
        Sorts the route for a given vehicle to ensure it follows a correct order.

        Args:
        - route (list of tuples): List of (from_node, to_node) pairs.
        - vehicle: The vehicle ID.

        Returns:
        - List of ordered nodes forming the complete route.
        """
        if not route:
            return []

        # Find the starting node (assume it's the depot/base node)
        start_node = self.mip_inputs.base_node_id  # Replace with the actual base node variable

        ordered_route = []
        current_node = start_node

        while route:
            for (i, j) in route:
                if i == current_node:
                    ordered_route.append((i, j))
                    current_node = j
                    route.remove((i, j))
                    break  # Move to the next arc

        return ordered_route

    def extract_vehicle_routes(self, model):
        """Extracts vehicle routes from the solution to determine which constraints to add."""
        vals_xijk = model.cbGetSolution(model._vars_x_ijk)
        nodes = self.mip_inputs.fire_ready_node_ids_and_base  # Get unique nodes
        vehicles = self.mip_inputs.vehicle_list  # Get unique vehicles

        # Dictionary to store the route of each vehicle
        self.vehicle_routes = {k: [] for k in vehicles}

        for k in vehicles:
            # Extract active arcs for vehicle k
            route = [(i, j) for i in nodes for j in nodes if i != j and vals_xijk.get((i, j, k), 0) > 0.5]

            # Store sorted route
            self.vehicle_routes[k] = self.sort_vehicle_route(route, k)

        print(self.vehicle_routes)



    def add_water_refill_constraints(self, model):
        """
        Dynamically enforces water refill constraints based on the current solution.

        This function retrieves the current solution values and applies necessary constraints
        to ensure that vehicles refuel appropriately, following problem-specific rules.
        """
        # # Constraint 8 - Add the water resource selection constraint dynamically
        # if self.best_obj >=0:
        #     model.cbLazy(model._vars_x_ijk[1, 6, 1] >= 1)


        # Iterate over each vehicle and its corresponding route
        for k, arc in self.vehicle_routes.items():
            for (i, j) in arc:
                # Skip arcs that start or end at node 1 (base node)
                if i == self.mip_inputs.base_node_id  or j == self.mip_inputs.base_node_id:
                    continue

                # 🚀 Skip if constraint is already added
                if (i, j, k) in self.added_constraints:
                    print(f"⏩ Skipping ({i}, {j}, {k}): Constraint previously added")
                    continue

                # Constraint 8 - Add the water resource selection constraint dynamically
                model.cbLazy(
                    model._vars_x_ijk[i, j, k] == model._vars_w_ijlk.sum(i, j, k, '*')
                )

                print(f"🔴 Lazy Constraint Added: x_ijk({i}, {j}, {k}) == w_ijlk.sum({i}, {j}, {k}, '*')")

                # Constraint 9 - Add the water resource connections for refilling
                # Find all water resource nodes l that exist in s_ijkw_links for (i, j, k)
                water_resources = self.mip_inputs.water_node_id
                # Add constraints for each water resource l
                for l in water_resources:
                    # Add lazy constraint
                    model.cbLazy(
                        2 * model._vars_w_ijlk[i, j, k, l] ==
                        model._vars_x_ijk.sum(i, l, k) + model._vars_x_ijk.sum(l, j, k)
                    )

                    print(f"🔴 Lazy Constraint Added: 2 * w_ijlk({i}, {j}, {k}, {l}) = x_ijk.sum({i}, {l}, {k}) + x_ijk.sum({l}, {j}, {k})")




                # Add constraints to determines arrival times to the nodes
                # Add lazy constraint
                i_to_water_coef = {key: v for key, v in self.mip_inputs.links_durations.items() if
                                          key[0] == i and key[1] in self.mip_inputs.water_node_id and key[2] == k}
                water_to_j_coef = {key: v for key, v in self.mip_inputs.links_durations.items() if
                                           key[0] in self.mip_inputs.water_node_id and key[1] == j and key[2] == k}

                model.cbLazy(
                    model._vars_tv_j[j] >= model._vars_tv_j[i] + model._vars_lv_j[i] +
                    model._vars_x_ijk.prod(i_to_water_coef, i, self.mip_inputs.water_node_id, '*') +
                    model._vars_x_ijk.prod(water_to_j_coef, self.mip_inputs.water_node_id, j, '*') -
                    self.mip_inputs.M_16[(i, j)] * (1 - model._vars_x_ijk.sum(i, j, k))

                )


                model.cbLazy(
                    model._vars_tv_j[j] <= model._vars_tv_j[i] +  model._vars_lv_j[i] +
                    model._vars_x_ijk.prod(i_to_water_coef, i, self.mip_inputs.water_node_id, '*') +
                    model._vars_x_ijk.prod(water_to_j_coef, self.mip_inputs.water_node_id, j, '*') +
                    self.mip_inputs.M_16[(i, j)] * (1 - model._vars_x_ijk.sum(i, j, k))

                )

                print(
                    f"⏳ Lazy Time Constraint Added: tv_j({j}) >= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")
                print(
                    f"⏳ Lazy Time Constraint Added: tv_j({j}) <= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")
                # Constraint 10 - Add the water resource connections for refilling



                # 🚀 Store the added constraint to prevent duplicates
                # Constraints for arc-vehicle pair i, j, k is added previously
                self.added_constraints.add((i, j, k))


def mathematical_model_solve_strengthen(mip_inputs):
    # the formulation is available at below link:
    # https://docs.google.com/document/d/1cCx4SCTII76LPAp1McpIxybUQPRcqfJZxiNHsSsYXQ8/


    model = gp.Model("firefighting")  #


    # add link variables - if the vehicle k moves from region i to j; 0, otherwise.
    x_ijk = model.addVars(
        mip_inputs.links,
        vtype=GRB.BINARY,
        name="x_ijk",
    )

    z_ij = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="z_ij",
    )

    q_ij = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="q_ij",
    )


    s1_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s1_i",
    )

    s2_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s2_i",
    )

    s3_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s3_i",
    )

    s4_i = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="s4_i",
    )

    y_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="y_j",
    )

    b_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="b_j",
    )

    ts_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="ts_j",
    )

    tm_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tm_j",
    )

    te_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="te_j",
    )

    tv_h = model.addVar(
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_h",
    )

    tv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_j",
    )

    #
    lv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_j",
    )

    lv_h = model.addVar(
        lb=0,
        ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_h",
    )


    p_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        # ub=[mip_inputs.node_object_dict[j].get_value_at_start() for j in mip_inputs.fire_ready_node_ids],
        vtype=GRB.CONTINUOUS,
        name="p_j",
    )

    w_ijlk = model.addVars(
        mip_inputs.s_ijkw_links,
        vtype=GRB.BINARY,
        name="w_ijlk",
    )

    # set objective
    obj_max = gp.quicksum(p_j[j] for j in mip_inputs.fire_ready_node_ids)

    penalty_coef_return_time = 0 # 10 ** -6

    obj_penalize_operation_time = penalty_coef_return_time * tv_h
    model.setObjective(obj_max - obj_penalize_operation_time)


    # # forced solution
    # model.addConstr(x_ijk.sum(1, 12, 1) == 1)
    # model.addConstr(z_ij[(4, 5)] == 0)


    # equations for prize collection
    # constraint 2 - determines collected prizes from at each node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(p_j[j] <= mip_inputs.node_object_dict[j].get_value_at_start() - mip_inputs.node_object_dict[j].get_value_degradation_rate() * tv_j[j] - mip_inputs.node_object_dict[j].get_value_at_start() * b_j[j])

    # constraint 3 - determines if a fire is burned down or not - that also impacts the decision of visiting a node to process the fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(b_j[j] >= y_j[j] - mip_inputs.M_3[j] * tv_j[j])

    # equations for scheduling and routing decisions
    # Constraint 4 - a vehicle that leaves the base must return to the base
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(mip_inputs.fire_ready_node_ids, mip_inputs.base_node_id,  k))

    # Constraint 5 - each vehicle can leave the base only once
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) <= 1)


    # Constraint 6 - flow balance equation -- incoming vehicles must be equal to the outgoing vehicles at each node
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, k) == x_ijk.sum(j, mip_inputs.fire_ready_node_ids_and_base, k))

    # Constraint 7 - at most one vehicle can visit a node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') <= 1)


    # # Constraint 8 - water resource selection for refilling
    # for i in mip_inputs.s_ijkw_links:
    #     model.addConstr(x_ijk.sum(i[0], i[1], i[2]) == w_ijlk.sum(i[0], i[1], i[2], '*'))

    # # Constraint 9 - water resource connections for refilling
    # for i in mip_inputs.s_ijkw_links:
    #     model.addConstr(2 * w_ijlk[i] <= x_ijk.sum(i[0], i[3], i[2]) + x_ijk.sum(i[3], i[1], i[2]) )

    # Constraint 10 - water resource connections for refilling
    for i in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(i, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(i, mip_inputs.water_node_id, k))

    # Constraint 11 - water resource connections for refilling
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids, j, k) == x_ijk.sum(mip_inputs.water_node_id, j, k))

    # Constraint 12 - time limitation
    model.addConstr(tv_h <= mip_inputs.time_limit)

    # Constraint 13 - determines return time to the base, considering the time of vehicle with maximum return time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_h >= tv_j[j] +
                        mip_inputs.links_durations[(j, mip_inputs.base_node_id, 1)] * x_ijk.sum(j, mip_inputs.base_node_id, '*') -
                        mip_inputs.M_13[j] * (1 - x_ijk.sum(j, mip_inputs.base_node_id, '*')))

    # Constraint 14 - determines arrival times to the nodes from home
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] <= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + mip_inputs.M_13[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 15 - determines arrival times to the nodes from home
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] >= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_13[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))
        # model.addConstr(tv_j[j] >= x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_13[j] * (
        #         1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 16 - determines arrival times to the nodes
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] == i and k[1] == j}

            # model.addConstr(
            #     tv_j[j] <= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_j_coef, i, j,'*') + mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))

            model.addConstr(
                tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_j_coef, i, j,'*') - mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))
                # tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_j_coef, i, j, '*') - mip_inputs.M_16[(i, j)] * (
                #             1 - x_ijk.sum(i, j, '*')))

    # # Constraint 16 - determines arrival times to the nodes
    # for i in mip_inputs.fire_ready_node_ids:
    #     to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
    #     for j in to_j_list:
    #         i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
    #                            k[0] == i and k[1] in mip_inputs.water_node_id}
    #         water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
    #                            k[0] in mip_inputs.water_node_id and k[1] == j}
    #         model.addConstr(
    #             tv_j[j] <= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
    #                 water_to_j_coef, mip_inputs.water_node_id, j, '*') + mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))
    #
    # # Constraint 17 - determines arrival times to the nodes
    # for i in mip_inputs.fire_ready_node_ids:
    #     to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
    #     for j in to_j_list:
    #         i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
    #                            k[0] == i and k[1] in mip_inputs.water_node_id}
    #         water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
    #                            k[0] in mip_inputs.water_node_id and k[1] == j}
    #         model.addConstr(
    #             tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
    #                 water_to_j_coef, mip_inputs.water_node_id, j, '*') - mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 18 - no arrival time at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 19 - no loitering at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(lv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 20 - vehicle arrival has to be after fire arrival (start)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] - ts_j[j] >= mip_inputs.M_19 * (x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') - 1))

    # Constraint 21 - vehicle can not arrive after the fire finished
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= te_j[j])


    # equations linking fire arrivals and scheduling decisions
    # Constraint 22 - fire spread case 1: t_v =0 --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(mip_inputs.M_21[i] * tv_j[i] >= (1-s1_i[i]))

    # Constraint 23 - fire spread case that allows case 2 and 3: t_v > 0
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] <=  mip_inputs.M_22[i] * s4_i[i])

    # Constraint 24 - fire spread case  2: t_v > 0 and t_v >= t_m --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] - tm_j[i] + (10 ** -6) <= mip_inputs.M_23[i] * s2_i[i])

    # Constraint 25 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[i] - tv_j[i] <= mip_inputs.M_24 * (s1_i[i] + s3_i[i]))

    # Constraint 26 - fire spread cases: only one of case 1, i.e. t_v=0, and case 2, i.e. t_v>0, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s1_i[i] + s4_i[i] == 1)

    # Constraint 27 - fire spread cases: only one of case 3, i.e. t_v>=t_m, and case 4, i.e. t_v<t_m, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s4_i[i] >= s2_i[i] + s3_i[i])

    # Constraint 28 - fire spread cases: if there is no fire in node i, it cannot spread to the adjacent nodes
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * y_j[i])

    # Constraint 29 - fire spread cases:  there is fire in node i, but no vehicle process it, i.e. t_v=0
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * (s1_i[i] + y_j[i] - 1))

    # Constraint 30 - fire spread cases:  there is fire in node i, a vehicle process it after it max point, i.e. t_v>=t_m --> it must spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * s2_i[i])

    # Constraint 31 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * (1-s3_i[i]))

    # Constraint 32 - if a fire spreads to an adjacent node, a fire must arrive to the adjacent node.
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * y_j[j] >= z_ij.sum('*', j))

    # Constraint 33 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(y_j[j] >= x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))  # x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 34 - active fires at start
    model.addConstr(gp.quicksum(y_j[j] for j in mip_inputs.set_of_active_fires_at_start) == len(
        mip_inputs.set_of_active_fires_at_start))

    # Constraint 35 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * q_ij.sum('*', j) >= z_ij.sum('*', j))

    # Constraint 36 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(q_ij.sum('*', j) <= 1)

    # Constraint 37 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        temp_neighborhood_list = [x for x in mip_inputs.node_object_dict[j].get_neighborhood_list() if x not in mip_inputs.fire_proof_node_list]
        for i in temp_neighborhood_list:  #for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
            model.addConstr(q_ij.sum(i, j) <= z_ij.sum(i, j))

    # Constraint 38 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        model.addConstr(ts_j[ln[1]] <= tm_j[ln[0]] + mip_inputs.M_37 * (1 - z_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]) + M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]))

    # Constraint 39 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        if ln[1] in mip_inputs.set_of_active_fires_at_start:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]) - mip_inputs.M_37)
        else:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]) - M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]))

    # Constraint 40 - determine fire arrival (spread) time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(ts_j[j] <= mip_inputs.M_37 * z_ij.sum('*', j))

    # Constraint 41 - start time of active fires
    model.addConstr(gp.quicksum(ts_j[j] for j in mip_inputs.set_of_active_fires_at_start) == 0)

    # Constraint 42 - determine fire spread time (the time at which the fire reaches its maximum size)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[j] == ts_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_degradation_rate()))

    # Constraint 43 - fire end time when it is not processed and burned down by itself
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(te_j[j] == tm_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_amelioration_rate()))


    # set starting solutions
    water_and_base = set(mip_inputs.water_node_id) | {mip_inputs.base_node_id}

    for key, value in mip_inputs.start_sol["x_ijk"].items():

        x_ijk[key].start = value

        i=key[0]
        j=key[1]
        k=key[2]

        if i not in water_and_base and j not in water_and_base:
            # Constraint 8 - Add the water resource selection constraint dynamically
            model.addConstr(
                x_ijk[key] == w_ijlk.sum(i, j, k, '*')
            )

            print(f"🔴 Constraint Added: x_ijk({i}, {j}, {k}) == w_ijlk.sum({i}, {j}, {k}, '*')")


            # Constraint 9 - Add the water resource connections for refilling
            # Find all water resource nodes l that exist in s_ijkw_links for (i, j, k)
            # Add constraints for each water resource l

            for l in  mip_inputs.water_node_id:
                # Add lazy constraint
                model.addConstr(
                    2 * w_ijlk[i, j, k, l] ==
                    x_ijk.sum(i, l, k) + x_ijk.sum(l, j, k)
                )
                print(
                    f"🔴 Constraint Added: 2 * w_ijlk({i}, {j}, {k}, {l}) = x_ijk.sum({i}, {l}, {k}) + x_ijk.sum({l}, {j}, {k})")

            # Add constraints to determines arrival times to the nodes
            # Add lazy constraint
            i_to_water_coef = {key: v for key, v in mip_inputs.links_durations.items() if
                               key[0] == i and key[1] in mip_inputs.water_node_id and key[2] == k}
            water_to_j_coef = {key: v for key, v in mip_inputs.links_durations.items() if
                               key[0] in mip_inputs.water_node_id and key[1] == j and key[2] == k}

            model.addConstr(
                tv_j[j] >= tv_j[i] + lv_j[i] +
                x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') +
                x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') -
                mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, k))

            )

            model.addConstr(
                tv_j[j] <= tv_j[i] + lv_j[i] +
                x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') +
                x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') +
                mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, k))

            )

            print(
                f"⏳ Time Constraint Added: tv_j({j}) >= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")
            print(
                f"⏳ Time Constraint Added: tv_j({j}) <= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")

            # cb_instance.added_constraints.add((i, j, k))




    for key, value in mip_inputs.start_sol["w_ijlk"].items():
        w_ijlk[key].start = value




    model.ModelSense = -1  # set objective to maximization
    # model.params.MIPFocus = 3
    # model.params.Presolve = 2

    model.setParam("LazyConstraints", 1)  # Enable lazy constraints

    model._vars_x_ijk = x_ijk
    model._vars_w_ijlk = w_ijlk
    model._vars_tv_j = tv_j
    model._vars_lv_j = lv_j
    model.update()


    # Create an instance of the callback using our CustomCallback class
    cb_instance = CustomCallback(mip_inputs)

    def my_callback(model, where):
        cb_instance.callback(model, where)

    # Optimize with the callback function
    start_time = time.time()
    model.optimize(my_callback)
    end_time = time.time()
    run_time_cpu = round(end_time - start_time, 2)



    if model.Status == GRB.Status.INFEASIBLE:
        max_dev_result = None
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("Go check infeasible_model.ilp file")
    else:

        x_ijk_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'value'])
        x_ijk_results_df = model_organize_results(x_ijk.values(), x_ijk_results_df)
        # x_ijk_results_df.loc[x_ijk_results_df["to_node_id"] == "24",]
       #  x_ijk_results_df.loc[x_ijk_results_df["from_node_id"] == "1",]

        w_ijlk_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id','value'])
        w_ijlk_results_df = model_organize_results(w_ijlk.values(), w_ijlk_results_df)

        y_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        y_j_results_df = model_organize_results(y_j.values(), y_j_results_df)

        z_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
        z_ij_results_df = model_organize_results(z_ij.values(), z_ij_results_df)
        #z_ij_results_df.loc[z_ij_results_df["to_node_id"]=="24",]

        q_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
        q_ij_results_df = model_organize_results(q_ij.values(), q_ij_results_df)

        b_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        b_j_results_df = model_organize_results(b_j.values(), b_j_results_df)

        ts_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        ts_j_results_df = model_organize_results(ts_j.values(), ts_j_results_df)

        tm_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        tm_j_results_df = model_organize_results(tm_j.values(), tm_j_results_df)

        te_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        te_j_results_df = model_organize_results(te_j.values(), te_j_results_df)

        tv_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        tv_j_results_df = model_organize_results(tv_j.values(), tv_j_results_df)
        tv_j_results_df.loc[len(tv_j_results_df.index)] = [tv_h.varName, mip_inputs.base_node_id, tv_h.X]

        tl_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        tl_j_results_df = model_organize_results(lv_j.values(), tl_j_results_df)
        tl_j_results_df.loc[len(tl_j_results_df.index)] = [lv_h.varName, mip_inputs.base_node_id, lv_h.X]

        p_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        p_j_results_df = model_organize_results(p_j.values(), p_j_results_df)

        s_ijkw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id', 'value'])
        s_ijkw_results_df = model_organize_results(w_ijlk.values(), s_ijkw_results_df)

        # model global results
        obj_result = model.objval + penalty_coef_return_time * tv_h.X

        global_results_df = pd.DataFrame(columns=['total_value', 'model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time', 'python_time'])
        global_results_df.loc[len(global_results_df.index)] = [obj_result, model.objval, model.objbound, model.mipgap,
                                                               model.runtime, run_time_cpu]



        if mip_inputs.experiment_mode in ["cluster_first", "cluster_first_combination_run"]:
            global_results_df["clustering_gurobi_time"] = mip_inputs.gurobi_time_clustering
            global_results_df["clustering_python_time"] = mip_inputs.run_time_cpu_clustering
            global_results_df["total_python_time"] = mip_inputs.run_time_cpu_clustering + run_time_cpu

        if mip_inputs.experiment_mode in ["single_run_hybrid_combination_run"]:
            global_results_df["run_time_original_mip"] = mip_inputs.run_time_original_mip
            global_results_df["run_time_callback_mip"] = run_time_cpu
            global_results_df["run_time_total"] = mip_inputs.run_time_original_mip + run_time_cpu

        global_results_df["operation_time"] = tv_h.X
        global_results_df["number_of_initial_fires"] = len(mip_inputs.set_of_active_fires_at_start)
        global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
        global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # subtract the base return time
        global_results_df["number_of_vehicles"] = len(mip_inputs.vehicle_list)  # subtract the base return time
        mip_inputs.base_node_id_string = str(mip_inputs.base_node_id)
        global_results_df["number_of_vehicles_used"] = len(np.unique(x_ijk_results_df.query("`from_node_id` == @mip_inputs.base_node_id_string & `value` > 0")["vehicle_id"].tolist()))  # subtract the base return time
        global_results_df["initial_fire_node_IDs"] = ','.join(map(str, mip_inputs.set_of_active_fires_at_start))


        if mip_inputs.experiment_mode in ["single_run", "single_run_lean", "single_run_strengthen", "single_run_hybrid", "cluster_first"]:
            writer_file_name = os.path.join('outputs', "{0}_results_{1}_nodes_{2}.xlsx".format(mip_inputs.experiment_mode,
                                                                                              mip_inputs.n_nodes,
                                                                                           str(datetime.now().strftime(
                                                                                               '%Y_%m_%d_%H_%M'))))

            writer = pd.ExcelWriter(writer_file_name)
            global_results_df.to_excel(writer, sheet_name='global_results')
            x_ijk_results_df.to_excel(writer, sheet_name='x_ijk_results')
            w_ijlk_results_df.to_excel(writer, sheet_name='w_ijlk_results')

            y_j_results_df.to_excel(writer, sheet_name='y_j_results')
            z_ij_results_df.to_excel(writer, sheet_name='z_ij_results')
            q_ij_results_df.to_excel(writer, sheet_name='q_ij_results')
            b_j_results_df.to_excel(writer, sheet_name='b_j_results')
            ts_j_results_df.to_excel(writer, sheet_name='ts_j_results')
            tm_j_results_df.to_excel(writer, sheet_name='tm_j_results')
            te_j_results_df.to_excel(writer, sheet_name='te_j_results')
            tv_j_results_df.to_excel(writer, sheet_name='tv_j_results')
            tl_j_results_df.to_excel(writer, sheet_name='tl_j_results')
            p_j_results_df.to_excel(writer, sheet_name='p_j_results')
            s_ijkw_results_df.to_excel(writer, sheet_name='s_ijkw_results')

            if mip_inputs.experiment_mode == "cluster_first":
                mip_inputs.u_jk_results_raw_df.to_excel(writer, sheet_name='u_jk_cluster_results')

            mip_inputs.problem_data_df.to_excel(writer, sheet_name='inputs_problem_data')
            mip_inputs.distance_df["flight_time"] = mip_inputs.distance_df["distance"] / mip_inputs.vehicle_flight_speed
            mip_inputs.distance_df.to_excel(writer, sheet_name='inputs_distances')
            mip_inputs.parameters_df.to_excel(writer, sheet_name='inputs_parameters')
            writer.close()

        elif mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file", "cluster_first_combination_run", "single_run_hybrid_combination_run"]:
            if mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file"]:
                writer_file_name = os.path.join('outputs', "combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes, mip_inputs.run_start_date))
            elif mip_inputs.experiment_mode == "single_run_hybrid_combination_run":
                writer_file_name = os.path.join('outputs',
                                                "hybrid_combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes,
                                                                                               mip_inputs.run_start_date))
            else:
                writer_file_name = os.path.join('outputs',
                                                "cluster_combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes,
                                                                                               mip_inputs.run_start_date))
            if os.path.isfile(writer_file_name):
                global_results_df.to_csv(writer_file_name, mode="a", index=False, header=False)
            else:
                global_results_df.to_csv(writer_file_name, mode="a", index=False, header=True)

        # global_results_df["operation_time"] = tv_h.X
        # global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
        # global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # substract the base return time

        return global_results_df



        # 24 - (24-mip_inputs.links_durations[1,7,1]) == mip_inputs.links_durations[1,7,1]






#
#
#
#
# def mathematical_model_solve_lean(mip_inputs):
#     # the formulation is available at below link:
#     # https://docs.google.com/document/d/1cCx4SCTII76LPAp1McpIxybUQPRcqfJZxiNHsSsYXQ8/
#
#     #
#     # update fire spread links
#     # if there is a fire in a node at start, there will be no spread to it ever
#     mip_inputs.neighborhood_links = gp.tuplelist(
#         (node_1, node_2) for node_1, node_2 in mip_inputs.neighborhood_links
#         if node_2 not in mip_inputs.set_of_active_fires_at_start
#     )
#
#     fire_free_node_ids = list(set(mip_inputs.fire_ready_node_ids) - set(mip_inputs.set_of_active_fires_at_start))
#
#
#
#     # Filter the active fires based on neighborhood conditions
#     filtered_active_fires = [
#         j for j in mip_inputs.fire_ready_node_ids
#         if any(neighbor in fire_free_node_ids for neighbor in mip_inputs.node_object_dict[j].get_neighborhood_list())
#     ]
#
#     # Update mip_inputs.set_of_active_fires_at_start with the filtered list
#     fire_free_node_ids = list(set(filtered_active_fires + fire_free_node_ids))
#
#
#     if mip_inputs.experiment_mode in ['cluster_first', 'cluster_first_combination_run', "cluster_first_lean",]:
#         # Build a dictionary mapping each vehicle to its cluster of nodes.
#         # For each vehicle, add home to its cluster list.
#
#
#         allowed_sets = {
#             int(vehicle.split('_')[1]): set(nodes + [mip_inputs.base_node_id] + mip_inputs.water_node_id )
#             for vehicle, nodes in mip_inputs.cluster_dict.items()
#         }
#
#         # Now filter the links.
#         # Each element in mip_inputs.links is a tuple (node_1, node_2, vehicle_id)
#         filtered_links = [
#             (node1, node2, veh)
#             for (node1, node2, veh) in mip_inputs.links
#             if node1 in allowed_sets[veh] and node2 in allowed_sets[veh]
#         ]
#
#         filtered_s_ijkw_links = [
#             (node1, node2, veh, node_water)
#             for (node1, node2, veh, node_water) in mip_inputs.s_ijkw_links
#             if node1 in allowed_sets[veh] and node2 in allowed_sets[veh]
#         ]
#
#
#         # Update the links tuple list
#         mip_inputs.links = gp.tuplelist(filtered_links)
#         mip_inputs.s_ijkw_links = gp.tuplelist(filtered_s_ijkw_links)
#
#
#
#
#     model = gp.Model("firefighting")  #
#
#
#     # add link variables - if the vehicle k moves from region i to j; 0, otherwise.
#     x_ijk = model.addVars(
#         mip_inputs.links,
#         vtype=GRB.BINARY,
#         name="x_ijk",
#     )
#
#     z_ij = model.addVars(
#         mip_inputs.neighborhood_links,
#         vtype=GRB.BINARY,
#         name="z_ij",
#     )
#
#     q_ij = model.addVars(
#         mip_inputs.neighborhood_links,
#         vtype=GRB.BINARY,
#         name="q_ij",
#     )
#
#
#     s1_i = model.addVars(
#         fire_free_node_ids,
#         vtype=GRB.BINARY,
#         name="s1_i",
#     )
#
#     s2_i = model.addVars(
#         fire_free_node_ids,
#         vtype=GRB.BINARY,
#         name="s2_i",
#     )
#
#     s3_i = model.addVars(
#         fire_free_node_ids,
#         vtype=GRB.BINARY,
#         name="s3_i",
#     )
#
#     s4_i = model.addVars(
#         fire_free_node_ids,
#         vtype=GRB.BINARY,
#         name="s4_i",
#     )
#
#     y_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         vtype=GRB.BINARY,
#         name="y_j",
#     )
#
#     b_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         vtype=GRB.BINARY,
#         name="b_j",
#     )
#
#     ts_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="ts_j",
#     )
#
#     tm_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="tm_j",
#     )
#
#     te_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="te_j",
#     )
#
#     tv_h = model.addVar(
#         lb=0,
#         ub=mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="tv_h",
#     )
#
#     tv_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="tv_j",
#     )
#
#     #
#     lv_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=0, # mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="lv_j",
#     )
#
#     lv_h = model.addVar(
#         lb=0,
#         ub=0, # mip_inputs.time_limit,
#         vtype=GRB.CONTINUOUS,
#         name="lv_h",
#     )
#
#
#     p_j = model.addVars(
#         mip_inputs.fire_ready_node_ids,
#         lb=0,
#         ub=[mip_inputs.node_object_dict[j].get_value_at_start() for j in mip_inputs.fire_ready_node_ids],
#         vtype=GRB.CONTINUOUS,
#         name="p_j",
#     )
#
#     w_ijlk = model.addVars(
#         mip_inputs.s_ijkw_links,
#         vtype=GRB.BINARY,
#         name="w_ijlk",
#     )
#
#     # set objective
#     obj_max = gp.quicksum(p_j[j] for j in mip_inputs.fire_ready_node_ids)
#
#     penalty_coef_return_time = 0 # 10 ** -6
#
#     # obj_penalize_fire_spread = gp.quicksum(z_ij[l] for l in mip_inputs.neighborhood_links)
#     obj_penalize_operation_time = penalty_coef_return_time * tv_h
#     model.setObjective(obj_max - obj_penalize_operation_time)
#
#
#     # # forced solution
#     # model.addConstr(x_ijk.sum(7, 8, 1) == 1)
#     # model.addConstr(x_ijk.sum(43, 7, 2) == 1)
#     # model.addConstr(x_ijk.sum(4, 3, 1) == 1)
#     # model.addConstr(x_ijk.sum(3, 5, 1) == 1)
#     # model.addConstr(x_ijk.sum(1, 8, 2) == 1)
#     # model.addConstr(x_ijk.sum(8, 9, 2) == 1)
#
#     # model.addConstr(z_ij[(4, 5)] == 0)
#     # model.addConstr(z_ij[(8, 5)] == 0)
#
#     # equations for prize collection
#     # constraint 2 - determines collected prizes from at each node
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(p_j[j] <= mip_inputs.node_object_dict[j].get_value_at_start() - mip_inputs.node_object_dict[j].get_value_degradation_rate() * tv_j[j] - mip_inputs.node_object_dict[j].get_value_at_start() * b_j[j])
#
#     # constraint 3 - determines if a fire is burned down or not - that also impacts the decision of visiting a node to process the fire
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(b_j[j] >= y_j[j] - mip_inputs.M_3[j] * tv_j[j])
#
#     # equations for scheduling and routing decisions
#     # Constraint 4 - a vehicle that leaves the base must return to the base
#     for k in mip_inputs.vehicle_list:
#         model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(mip_inputs.fire_ready_node_ids, mip_inputs.base_node_id,  k))
#
#     # Constraint 5 - each vehicle can leave the base only once
#     for k in mip_inputs.vehicle_list:
#         model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) <= 1)
#     # model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, '*') <= 1)
#     # model.addConstr(x_ijk.sum(mip_inputs.base_node_id, 7, '*') == 1)
#
#     # Constraint 6 - flow balance equation -- incoming vehicles must be equal to the outgoing vehicles at each node
#     for j in mip_inputs.fire_ready_node_ids:
#         for k in mip_inputs.vehicle_list:
#             model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, k) == x_ijk.sum(j, mip_inputs.fire_ready_node_ids_and_base, k))
#
#     # Constraint 7 - at most one vehicle can visit a node
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') <= 1)
#     #
#     # if mip_inputs.experiment_mode == 'cluster_first':
#     #     # Constraint 7 - at most one vehicle can visit a node
#     #     # vehicle_1_nodes = [24, 4, 10, 11,18, 19, 25]
#     #     # vehicle_1_nodes = [17, 2, 4, 9, 10, 15, 16, 22, 25]
#     #     # vehicle_2_nodes = [18, 6, 7, 8, 12, 14,19, 24]
#     #
#     #     mip_inputs.cluster_dict
#     #     mip_inputs.cluster_cant_go = {
#     #         f"{vehicle}_cant_go": list(set(mip_inputs.fire_ready_node_ids) - set(nodes))
#     #         for vehicle, nodes in mip_inputs.cluster_dict.items()
#     #     }
#     #
#     #     # vehicle_1_cant_go = list(set(mip_inputs.fire_ready_node_ids) - set(vehicle_1_nodes))
#     #     # vehicle_2_cant_go = list(set(mip_inputs.fire_ready_node_ids) - set(vehicle_2_nodes))
#     #     # vehicle_3_cant_go = list(set(mip_inputs.fire_ready_node_ids) - set(vehicle_3_nodes))
#     #     # model.update()
#     #     # vehicle = 'vehicle_1'
#     #     # nodes = [4, 7, 8, 9, 10, 12, 18, 19, 14]
#     #
#     #     # Constraints to force vehicles to leave from the base to their clusters
#     #     for vehicle, nodes in mip_inputs.cluster_dict.items():
#     #         # Extract the numeric part from the vehicle key (e.g., "vehicle_1" -> 1)
#     #         vehicle_num = int(vehicle.split('_')[-1])
#     #         model.addConstr(
#     #             x_ijk.sum(mip_inputs.base_node_id, nodes, vehicle_num) == 1,
#     #             name=f"constraint_vehicle_{vehicle_num}"
#     #         )
#     #
#     #     # Constraints to make sure that vehicles cant visit any nodes in other clusters
#     #     key = 'vehicle_1_cant_go'
#     #     cant_go_nodes = [2, 6, 15, 16, 17, 22, 24, 25]
#     #     for key, cant_go_nodes in mip_inputs.cluster_cant_go.items():
#     #         # Extract the vehicle number from the key.
#     #         # For "vehicle_1_cant_go", splitting by '_' gives ["vehicle", "1", "cant", "go"]
#     #         vehicle_num = int(key.split('_')[1])
#     #
#     #         # Add a constraint for each node j in the cant_go list for this vehicle.
#     #         for j in cant_go_nodes:
#     #             model.addConstr(
#     #                 x_ijk.sum('*', j, vehicle_num) == 0,
#     #                 name=f"cantgo_vehicle{vehicle_num}_node{j}"
#     #             )
#     #
#     #     # vehicle 1
#     #     model.addConstr(x_ijk.sum(mip_inputs.base_node_id, vehicle_1_nodes, 1) == 1)
#     #     for j in vehicle_1_cant_go:
#     #         model.addConstr(x_ijk.sum('*', j, 1) == 0)
#     #     # for j in vehicle_1_nodes:
#     #         # model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids+[1], j, 1) == 1)
#     #
#     #     # vehicle 2
#     #     model.addConstr(x_ijk.sum(mip_inputs.base_node_id, vehicle_2_nodes, 2) == 1)
#     #     for j in vehicle_2_cant_go:
#     #         model.addConstr(x_ijk.sum('*', j, 2) == 0)
#     #     # for j in vehicle_2_nodes:
#     #         # model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids+[1], j, 2) == 1)
#     #
#     #     # vehicle 3
#     #     model.addConstr(x_ijk.sum(mip_inputs.base_node_id, vehicle_3_nodes, 3) == 1)
#     #     for j in vehicle_3_cant_go:
#     #         model.addConstr(x_ijk.sum('*', j, 3) == 0)
#     #     # for j in vehicle_3_nodes:
#     #         # model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids+[1], j, 3) == 1)
#     #
#
#
#     # Constraint 8 - water resource selection for refilling
#     for i in mip_inputs.s_ijkw_links:
#         model.addConstr(x_ijk.sum(i[0], i[1], i[2]) == w_ijlk.sum(i[0], i[1], i[2], '*'))
#
#     # Constraint 9 - water resource connections for refilling
#     for i in mip_inputs.s_ijkw_links:
#         model.addConstr(2 * w_ijlk[i] <= x_ijk.sum(i[0], i[3], i[2]) + x_ijk.sum(i[3], i[1], i[2]) )
#
#     # Constraint 10 - water resource connections for refilling
#     for i in mip_inputs.fire_ready_node_ids:
#         for k in mip_inputs.vehicle_list:
#             model.addConstr(x_ijk.sum(i, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(i, mip_inputs.water_node_id, k))
#
#     # Constraint 11 - water resource connections for refilling
#     for j in mip_inputs.fire_ready_node_ids:
#         for k in mip_inputs.vehicle_list:
#             model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids, j, k) == x_ijk.sum(mip_inputs.water_node_id, j, k))
#
#     # Constraint 12 - time limitation
#     model.addConstr(tv_h <= mip_inputs.time_limit)
#     # constraint_final_time = model.addConstr(tv_h <= mip_inputs.time_limit)
#     # constraint_final_time.Lazy = 1
#
#     # Constraint 13 - determines return time to the base, considering the time of vehicle with maximum return time
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(tv_h >= tv_j[j] +
#                         mip_inputs.links_durations[(j, mip_inputs.base_node_id, 1)] * x_ijk.sum(j, mip_inputs.base_node_id, '*') -
#                         mip_inputs.M_13[j] * (1 - x_ijk.sum(j, mip_inputs.base_node_id, '*')))
#
#     # Constraint 14 - determines arrival times to the nodes
#     for j in mip_inputs.fire_ready_node_ids:
#         home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                           k[0] == mip_inputs.base_node_id and k[1] == j}
#         model.addConstr(tv_j[j] <= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + mip_inputs.M_13[j] * (
#                 1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))
#
#     # Constraint 15 - determines arrival times to the nodes
#     for j in mip_inputs.fire_ready_node_ids:
#         home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                           k[0] == mip_inputs.base_node_id and k[1] == j}
#         model.addConstr(tv_j[j] >= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_13[j] * (
#                 1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))
#
#     # Constraint 16 - determines arrival times to the nodes
#     for i in mip_inputs.fire_ready_node_ids:
#         to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
#         for j in to_j_list:
#             i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                                k[0] == i and k[1] in mip_inputs.water_node_id}
#             water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                                k[0] in mip_inputs.water_node_id and k[1] == j}
#             model.addConstr(
#                 tv_j[j] <= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
#                     water_to_j_coef, mip_inputs.water_node_id, j, '*') + mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))
#
#     # Constraint 17 - determines arrival times to the nodes
#     for i in mip_inputs.fire_ready_node_ids:
#         to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
#         for j in to_j_list:
#             i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                                k[0] == i and k[1] in mip_inputs.water_node_id}
#             water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
#                                k[0] in mip_inputs.water_node_id and k[1] == j}
#             model.addConstr(
#                 tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
#                     water_to_j_coef, mip_inputs.water_node_id, j, '*') - mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))
#
#     # Constraint 18 - no arrival time at unvisited nodes
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(tv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))
#
#     # Constraint 19 - no loitering at unvisited nodes
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(lv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))
#
#     # Constraint 20 - vehicle arrival has to be after fire arrival (start)
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(tv_j[j] - ts_j[j] >= mip_inputs.M_19 * (x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') - 1))
#
#     # Constraint 21 - vehicle can not arrive after the fire finished
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(tv_j[j] <= te_j[j])
#
#
#     # equations linking fire arrivals and scheduling decisions
#     # Constraint 22 - fire spread case 1: t_v =0 --> fire spreads
#     for i in fire_free_node_ids:
#         model.addConstr(mip_inputs.M_21[i] * tv_j[i] >= (1-s1_i[i]))
#
#     # Constraint 23 - fire spread case that allows case 2 and 3: t_v > 0
#     for i in fire_free_node_ids:
#         model.addConstr(tv_j[i] <=  mip_inputs.M_22[i] * s4_i[i])
#
#     # Constraint 24 - fire spread case  2: t_v > 0 and t_v >= t_m --> fire spreads
#     for i in fire_free_node_ids:
#         model.addConstr(tv_j[i] - tm_j[i] + (10 ** -6) <= mip_inputs.M_23[i] * s2_i[i])
#
#     # Constraint 25 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
#     for i in fire_free_node_ids:
#         model.addConstr(tm_j[i] - tv_j[i] <= mip_inputs.M_24 * (s1_i[i] + s3_i[i]))
#
#     # Constraint 26 - fire spread cases: only one of case 1, i.e. t_v=0, and case 2, i.e. t_v>0, can occur
#     for i in fire_free_node_ids:
#         model.addConstr(s1_i[i] + s4_i[i] == 1)
#
#     # Constraint 27 - fire spread cases: only one of case 3, i.e. t_v>=t_m, and case 4, i.e. t_v<t_m, can occur
#     for i in fire_free_node_ids:
#         model.addConstr(s4_i[i] >= s2_i[i] + s3_i[i])
#
#     # Constraint 28 - fire spread cases: if there is no fire in node i, it cannot spread to the adjacent nodes
#     for i in fire_free_node_ids:
#         i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
#         i_neighborhood_size = len(i_neighborhood)
#         model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * y_j[i])
#
#     # Constraint 29 - fire spread cases:  there is fire in node i, but no vehicle process it, i.e. t_v=0
#     for i in fire_free_node_ids:
#         i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
#         i_neighborhood_size = len(i_neighborhood)
#         model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * (s1_i[i] + y_j[i] - 1))
#
#     # Constraint 30 - fire spread cases:  there is fire in node i, a vehicle process it after it max point, i.e. t_v>=t_m --> it must spread to the adjacent cells
#     for i in fire_free_node_ids:
#         i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
#         i_neighborhood_size = len(i_neighborhood)
#         model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * s2_i[i])
#
#     # Constraint 31 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
#     for i in fire_free_node_ids:
#         i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
#         i_neighborhood_size = len(i_neighborhood)
#         model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * (1-s3_i[i]))
#
#     # Constraint 32 - if a fire spreads to an adjacent node, a fire must arrive to the adjacent node.
#     for j in fire_free_node_ids:
#         j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
#         model.addConstr(j_neighborhood_size * y_j[j] >= z_ij.sum('*', j))
#
#     # Constraint 33 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(y_j[j] >= x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))  # x_ijk.sum(mip_inputs.node_list, j, '*'))
#
#     # Constraint 34 - active fires at start
#     model.addConstr(gp.quicksum(y_j[j] for j in mip_inputs.set_of_active_fires_at_start) == len(
#         mip_inputs.set_of_active_fires_at_start))
#
#     # Constraint 35 - determine fire spread
#     for j in fire_free_node_ids:
#         j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
#         model.addConstr(j_neighborhood_size * q_ij.sum('*', j) >= z_ij.sum('*', j))
#
#     # Constraint 36 - determine fire spread
#     for j in fire_free_node_ids:
#         model.addConstr(q_ij.sum('*', j) <= 1)
#
#     # Constraint 37 - determine fire spread
#     for j in fire_free_node_ids:
#         temp_neighborhood_list = [x for x in mip_inputs.node_object_dict[j].get_neighborhood_list() if x not in mip_inputs.fire_proof_node_list]
#         for i in temp_neighborhood_list:  #for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
#             model.addConstr(q_ij.sum(i, j) <= z_ij.sum(i, j))
#
#     # Constraint 38 - determine fire arrival (spread) time
#     for ln in mip_inputs.neighborhood_links:
#         model.addConstr(ts_j[ln[1]] <= tm_j[ln[0]] + mip_inputs.M_37 * (1 - z_ij[ln]))
#         # if n[1] in mip_inputs.set_of_active_fires_at_start:
#         #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]) + M_30)
#         # else:
#         #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]))
#
#     # Constraint 39 - determine fire arrival (spread) time
#     for ln in mip_inputs.neighborhood_links:
#         if ln[1] in mip_inputs.set_of_active_fires_at_start:
#             model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]) - mip_inputs.M_37)
#         else:
#             model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]))
#         # if n[1] in mip_inputs.set_of_active_fires_at_start:
#         #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]) - M_30)
#         # else:
#         #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]))
#
#     # Constraint 40 - determine fire arrival (spread) time
#     for j in fire_free_node_ids:
#         model.addConstr(ts_j[j] <= mip_inputs.M_37 * z_ij.sum('*', j))
#
#     # Constraint 41 - start time of active fires
#     model.addConstr(gp.quicksum(ts_j[j] for j in mip_inputs.set_of_active_fires_at_start) == 0)
#
#     # Constraint 42 - determine fire spread time (the time at which the fire reaches its maximum size)
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(tm_j[j] == ts_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_degradation_rate()))
#
#     # Constraint 43 - fire end time when it is not processed and burned down by itself
#     for j in mip_inputs.fire_ready_node_ids:
#         model.addConstr(te_j[j] == tm_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_amelioration_rate()))
#
#
#     # Constraint xx - valid inequality cuts
#     #
#     # for i in mip_inputs.fire_ready_node_ids:
#     #     print(i)
#     #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
#     #     i_neighborhood_size = len(i_neighborhood)
#     #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * b_j[i])
#     #
#     # for i in mip_inputs.fire_ready_node_ids:
#     #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
#     #     i_neighborhood_size = len(i_neighborhood)
#     #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * s2_i[i])
#
#
#
#     model.ModelSense = -1  # set objective to maximization
#
#     # erdi parameters
#     model.params.TimeLimit = 3600
#     model.params.MIPGap = 0.03
#     model.params.NoRelHeurTime = 5
#     model.params.MIPFocus = 3
#     model.params.Cuts = 0
#     model.params.Presolve = 2
#     # model.params.VarBranch = 1
#     # model.params.BranchDir = 1
#    #  model.params.Aggregate = 0
#
#     # default parameters
#     # model.params.TimeLimit = 1200
#     # model.params.MIPGap = 0.03
#     #
#     # # heuristic parameters
#     # model.params.TimeLimit = max(120, mip_inputs.exact_run_time/10)
#     # model.params.MIPGap = 0.03
#     # model.params.NoRelHeurTime = 40
#     # model.params.Presolve = 2
#     # model.params.MIPFocus = 1
#     #
#
#     # model.params.TimeLimit = 1200
#     # model.params.MIPGap = 0.03
#     # model.params.MIPFocus = 2
#     # model.params.CliqueCuts = 2
#     # model.params.Cuts = 2
#     # model.params.Presolve = 2
#     # model.params.BranchDir = 1
#     # model.params.Heuristics = 0.1
#     # model.params.ImproveStartGap = 0.1
#     # model.params.NoRelHeurTime = 120
#
#     # model.params.LogFile = "gurobi_log"
#     # model.params.Heuristics = 0.2
#     # model.params.Threads = 8
#
#
#     # model.update()
#     # model.write("model_hand2.lp")
#     # (23.745 - 23.39) == (24.1-23.745)
#     # 23.745 - 23.390
#     # 0.455*0.355
#     model.update()
#     model.printStats()
#     start_time = time.time()
#     model.optimize()
#     end_time = time.time()
#     run_time_cpu = round(end_time - start_time, 2)
#
#     # for c in model.getConstrs():
#     #     if c.Slack < 1e-6:
#     #         print('Constraint %s is active at solution point' % (c.ConstrName))
#     #
#
#     if model.Status == GRB.Status.INFEASIBLE:
#         max_dev_result = None
#         model.computeIIS()
#         model.write("infeasible_model.ilp")
#         print("Go check infeasible_model.ilp file")
#     else:
#
#         x_ijk_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'value'])
#         x_ijk_results_df = model_organize_results(x_ijk.values(), x_ijk_results_df)
#         # x_ijk_results_df.loc[x_ijk_results_df["to_node_id"] == "24",]
#        #  x_ijk_results_df.loc[x_ijk_results_df["from_node_id"] == "1",]
#
#         y_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         y_j_results_df = model_organize_results(y_j.values(), y_j_results_df)
#
#         z_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
#         z_ij_results_df = model_organize_results(z_ij.values(), z_ij_results_df)
#         #z_ij_results_df.loc[z_ij_results_df["to_node_id"]=="24",]
#
#         q_ij_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'value'])
#         q_ij_results_df = model_organize_results(q_ij.values(), q_ij_results_df)
#
#         b_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         b_j_results_df = model_organize_results(b_j.values(), b_j_results_df)
#
#         ts_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         ts_j_results_df = model_organize_results(ts_j.values(), ts_j_results_df)
#
#         tm_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         tm_j_results_df = model_organize_results(tm_j.values(), tm_j_results_df)
#
#         te_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         te_j_results_df = model_organize_results(te_j.values(), te_j_results_df)
#
#         tv_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         tv_j_results_df = model_organize_results(tv_j.values(), tv_j_results_df)
#         tv_j_results_df.loc[len(tv_j_results_df.index)] = [tv_h.varName, mip_inputs.base_node_id, tv_h.X]
#
#         tl_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         tl_j_results_df = model_organize_results(lv_j.values(), tl_j_results_df)
#         tl_j_results_df.loc[len(tl_j_results_df.index)] = [lv_h.varName, mip_inputs.base_node_id, lv_h.X]
#
#         p_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
#         p_j_results_df = model_organize_results(p_j.values(), p_j_results_df)
#
#         s_ijkw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id', 'value'])
#         s_ijkw_results_df = model_organize_results(w_ijlk.values(), s_ijkw_results_df)
#
#
#         s1_i
#         s2_i
#         s3_i
#         s4_i
#
#         # model global results
#         obj_result = model.objval + penalty_coef_return_time * tv_h.X
#
#         global_results_df = pd.DataFrame(columns=['total_value', 'model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time', 'python_time'])
#         global_results_df.loc[len(global_results_df.index)] = [obj_result, model.objval, model.objbound, model.mipgap,
#                                                                model.runtime, run_time_cpu]
#
#
#
#         if mip_inputs.experiment_mode in ["cluster_first", "cluster_first_combination_run"]:
#             global_results_df["clustering_gurobi_time"] = mip_inputs.gurobi_time_clustering
#             global_results_df["clustering_python_time"] = mip_inputs.run_time_cpu_clustering
#             global_results_df["total_python_time"] = mip_inputs.run_time_cpu_clustering + run_time_cpu
#
#
#         global_results_df["operation_time"] = tv_h.X
#         global_results_df["number_of_initial_fires"] = len(mip_inputs.set_of_active_fires_at_start)
#         global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
#         global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # subtract the base return time
#         global_results_df["number_of_vehicles"] = len(mip_inputs.vehicle_list)  # subtract the base return time
#         mip_inputs.base_node_id_string = str(mip_inputs.base_node_id)
#         global_results_df["number_of_vehicles_used"] = len(np.unique(x_ijk_results_df.query("`from_node_id` == @mip_inputs.base_node_id_string & `value` > 0")["vehicle_id"].tolist()))  # subtract the base return time
#         global_results_df["initial_fire_node_IDs"] = ','.join(map(str, mip_inputs.set_of_active_fires_at_start))
#
#
#         if mip_inputs.experiment_mode in ["single_run", "single_run_lean", "cluster_first"]:
#             writer_file_name = os.path.join('outputs', "{0}_results_{1}_nodes_{2}.xlsx".format(mip_inputs.experiment_mode,
#                                                                                               mip_inputs.n_nodes,
#                                                                                            str(datetime.now().strftime(
#                                                                                                '%Y_%m_%d_%H_%M'))))
#
#             writer = pd.ExcelWriter(writer_file_name)
#             global_results_df.to_excel(writer, sheet_name='global_results')
#             x_ijk_results_df.to_excel(writer, sheet_name='x_ijk_results')
#             y_j_results_df.to_excel(writer, sheet_name='y_j_results')
#             z_ij_results_df.to_excel(writer, sheet_name='z_ij_results')
#             q_ij_results_df.to_excel(writer, sheet_name='q_ij_results')
#             b_j_results_df.to_excel(writer, sheet_name='b_j_results')
#             ts_j_results_df.to_excel(writer, sheet_name='ts_j_results')
#             tm_j_results_df.to_excel(writer, sheet_name='tm_j_results')
#             te_j_results_df.to_excel(writer, sheet_name='te_j_results')
#             tv_j_results_df.to_excel(writer, sheet_name='tv_j_results')
#             tl_j_results_df.to_excel(writer, sheet_name='tl_j_results')
#             p_j_results_df.to_excel(writer, sheet_name='p_j_results')
#             s_ijkw_results_df.to_excel(writer, sheet_name='s_ijkw_results')
#
#             if mip_inputs.experiment_mode == "cluster_first":
#                 mip_inputs.u_jk_results_raw_df.to_excel(writer, sheet_name='u_jk_cluster_results')
#
#             mip_inputs.problem_data_df.to_excel(writer, sheet_name='inputs_problem_data')
#             mip_inputs.distance_df["flight_time"] = mip_inputs.distance_df["distance"] / mip_inputs.vehicle_flight_speed
#             mip_inputs.distance_df.to_excel(writer, sheet_name='inputs_distances')
#             mip_inputs.parameters_df.to_excel(writer, sheet_name='inputs_parameters')
#             writer.close()
#
#         elif mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file", "cluster_first_combination_run"]:
#             if mip_inputs.experiment_mode in ["combination_run", "combination_run_from_file"]:
#                 writer_file_name = os.path.join('outputs', "combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes, mip_inputs.run_start_date))
#             else:
#                 writer_file_name = os.path.join('outputs',
#                                                 "cluster_combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes,
#                                                                                                mip_inputs.run_start_date))
#             if os.path.isfile(writer_file_name):
#                 global_results_df.to_csv(writer_file_name, mode="a", index=False, header=False)
#             else:
#                 global_results_df.to_csv(writer_file_name, mode="a", index=False, header=True)
#
#         # global_results_df["operation_time"] = tv_h.X
#         # global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
#         # global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # substract the base return time
#
#         return global_results_df
#
#
#
#         # 24 - (24-mip_inputs.links_durations[1,7,1]) == mip_inputs.links_durations[1,7,1]




    #
    # def mycallback(model, where):
    #     """Callback function to add subtour elimination constraints lazily."""
    #     if where == GRB.Callback.MIPSOL:
    #         # Get the current best solution's objective value
    #         current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    #         best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)  # Best bound so far
    #
    #         print(f"Current Feasible Solution: {current_obj}, Best Bound: {best_bound}")
    #
    #         # # Prevent stopping if the best objective is ≤ 0
    #         # if current_obj <= 0:
    #         #     print("Objective too low, forcing solver to continue...")
    #         #     return  # Do not allow termination yet
    #
    #         # Compute MIP Gap
    #         if abs(current_obj) > 1e-6:  # Avoid division by zero
    #             mip_gap = abs(best_bound - current_obj) / max(abs(current_obj), 1)
    #         else:
    #             mip_gap = float("inf")  # Set infinite gap if objective is zero
    #
    #         print(
    #             f"New Feasible Solution Found - Objective: {current_obj}, Best Bound: {best_bound}, MIP Gap: {mip_gap:.2%}")
    #
    #         # 🚫 Prevent stopping if the MIP gap is too high
    #         if mip_gap > 0.25:
    #             print("⚠️ MIP gap > 50%, forcing solver to continue...")
    #             return  # Do NOT allow stopping yet!
    #
    #
    #         # Get solution values
    #         vals_xijk = model.cbGetSolution(model._vars_x_ijk)
    #         vals_y_j = model.cbGetSolution(model._vars_y_j)
    #         vals_p_j = model.cbGetSolution(model._vars_p_j)
    #
    #         nodes = set(i for i, _, _ in model._vars_x_ijk.keys())  # Get unique nodes
    #         vehicles = set(k for _, _, k in model._vars_x_ijk.keys())  # Get unique vehicles
    #
    #         for k in vehicles:  # Process each vehicle separately
    #             edges = [(i, j) for i in nodes for j in nodes if vals_xijk.get((i, j, k), 0) > 0.5]
    #
    #             # Detect the smallest subtour for vehicle k
    #             tour = find_subtour(edges, nodes)
    #
    #             # If a subtour is found, add a lazy constraint
    #             if len(tour) < len(nodes):
    #                 expr = gp.quicksum(model._vars[i, j, k] for i in tour for j in tour if i < j)
    #                 model.cbLazy(expr <= len(tour) - 1)
    #
    #
    #         # Extract edges selected in the solution
    #         edges = [(i, j, k) for i in range(n) for j in range(i + 1, n) if vals[i, j] > 0.5]
    #
    #         # Identify the smallest subtour
    #         tour = subtour(edges, n)
    #
    #         # If a subtour is found, add a lazy constraint
    #         if len(tour) < n:
    #             expr = gp.quicksum(model._vars[i, j] for i in tour for j in tour if i < j)
    #             model.cbLazy(expr <= len(tour) - 1)
    #
    #     # Create the model
    #










      # if where == GRB.Callback.MIPSOL:
        #
        #     # Get the current best integer solution (incumbent)
        #     best_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        #     # Get the best bound on the objective function
        #     best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        #
        #     # Compute the MIP gap manually (avoid division by zero)
        #     if abs(best_obj) > 1e-9:
        #         mip_gap = abs(best_obj - best_bound) / abs(best_obj)
        #     else:
        #         mip_gap = float('inf')  # If no feasible solution is found yet
        #
        #     print(f"📊 New Feasible Solution Found! MIP Gap: {mip_gap:.2%}")
        #
        #     # 🚀 If the MIP gap is below 25%, enforce lazy constraints
        #     if mip_gap <= 0.05:
        #         print("🔴 MIP Gap is below 3%! Adding Lazy Constraints.")
        #
        #         # Extract vehicle routes from the solution
        #         self.extract_vehicle_routes(model)
        #
        #         # Add missing constraints
        #         self.add_water_refill_constraints(model)
        #
        #
        #         print("✅ Lazy constraints added due to MIP gap threshold.")
        #
        #     # Print all added lazy constraints
        #     print(f"📜 Current Lazy Constraints: {self.added_constraints}")
        #

            #
            #
            # # Get the current best feasible solution (incumbent)
            # current_best = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            #
            # # Check improvement compared to the previous best solution
            # if self.prev_best_obj is not None:
            #     improvement = abs(current_best - self.prev_best_obj) / max(abs(self.prev_best_obj), 1)
            #
            #     print(
            #         f"Best Feasible Solution: {current_best}, Previous: {self.prev_best_obj}, Improvement: {improvement:.2%}")
            #
            #     # 🚀 If improvement is below 5%, track consecutive stagnation
            #     if improvement < self.improvement_threshold:
            #         self.no_improve_count += 1
            #         print(f"⚠️ No significant improvement for {self.no_improve_count} iterations.")
            #     else:
            #         self.no_improve_count = 0  # Reset counter if improvement occurs
            #
            #     # 🚀 If improvement is under 5% for `max_no_improve` consecutive iterations, add water refill constraint
            #     if self.no_improve_count >= self.max_no_improve:
            #         print(
            #             f"🔴 No improvement for {self.max_no_improve} consecutive iterations! Adding Water Refill Constraint #{self.water_refill_constraints_added + 1}")
            #
            #         # Extract vehicle routes from the solution
            #         self.extract_vehicle_routes(model)
            #
            #         self.add_water_refill_constraints(model)
            #
            #         self.water_refill_constraints_added += 1
            #
            #         self.no_improve_count = 0  # Reset after adding constraint
            #
            # # Update previous best objective for the next iteration
            # self.prev_best_obj = current_best

        #
        # elif where == GRB.Callback.MIP:
        #     """🚀 Ensure that the final solution satisfies all lazy constraints before termination."""
        #     nodes_left = model.cbGet(GRB.Callback.MIP_NODLFT)  # Remaining nodes
        #     best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)  # Best known integer solution
        #     best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)  # Best bound on objective
        #
        #     # Compute MIP gap manually
        #     if abs(best_obj) > 1e-9:
        #         mip_gap = abs(best_obj - best_bound) / abs(best_obj)
        #     else:
        #         mip_gap = float('inf')
        #
        #     print(f"📊 Periodic Check in MIP: MIP Gap: {mip_gap:.2%}, Nodes Left: {nodes_left}")
        #
        #     # 🚀 If solver is about to terminate (no nodes left & MIP gap ≤ 3%), force a final lazy constraint check
        #     if nodes_left == 0 and mip_gap <= 0.03:
        #         print("🔴 Solver is about to terminate! Ensuring final lazy constraints are enforced.")
        #
        #         # Extract vehicle routes one last time
        #         self.extract_vehicle_routes(model)
        #
        #         # Add any missing constraints
        #         self.add_water_resource_constraints(model)
        #         self.add_water_resource_visit_constraints(model)
        #         self.add_time_variable_constraints(model)
        #
        #         print("✅ Final lazy constraints added before termination.")

        # elif where == GRB.Callback.MIP:
        #     """🚀 Ensuring we enforce lazy constraints if the gap drops below 3% without a new feasible solution."""
        #     # Get the best objective (current best feasible solution)
        #     best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        #     # Get the best bound
        #     best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        #
        #     # Compute the MIP gap manually
        #     if abs(best_obj) > 1e-9:
        #         mip_gap = abs(best_obj - best_bound) / abs(best_obj)
        #     else:
        #         mip_gap = float('inf')  # If no feasible solution exists
        #
        #     print(f"📊 Periodic Check in MIP: MIP Gap: {mip_gap:.2%}")
        #
        #     # 🚀 If the MIP gap is below 3% but no new feasible solution was found, enforce lazy constraints
        #     if mip_gap <= 0.03:
        #         print("🔴 MIP Gap is below 3% (even without new feasible solution)! Adding Lazy Constraints.")
        #
        #         # Extract vehicle routes from the solution
        #         self.extract_vehicle_routes(model)
        #
        #         # Add missing constraints
        #         self.add_water_refill_constraints(model)
        #
        #
        #         print("✅ Lazy constraints added due to MIP gap threshold (MIP check).")
        #



    #
    #
    #
    #
    # iteration=1
    # while True:
    #     print(f"\n🌀 Iteration {iteration}: Optimizing model...")
    #     model.optimize(my_callback)
    #
    #     missing_constraints = []
    #
    #     all_arcs = set(model._vars_x_ijk.keys()) | cb_instance.added_constraints
    #
    #     for (i, j, k) in all_arcs: # model._vars_x_ijk.items():
    #         var = model._vars_x_ijk[i, j, k]
    #
    #         if var.X <= 0.5:
    #             continue
    #         if i in [mip_inputs.base_node_id, *mip_inputs.water_node_id] or j in [mip_inputs.base_node_id, *mip_inputs.water_node_id]:
    #             continue
    #         print(f"⚠️ Missing constraint for arc ({i}, {j}, {k}) in final solution")
    #         missing_constraints.append((i, j, k))
    #
    #     # Step 2: Exit if no new constraints are needed
    #     if not missing_constraints:
    #         print(f"\n✅ No missing constraints. Final solution is valid.")
    #         break
    #
    #
    #     for (i, j, k) in missing_constraints:
    #     #     if i == mip_inputs.base_node_id or j == mip_inputs.base_node_id:
    #     #         continue  # skip depot-to-depot arcs
    #
    #         # Constraint 8 - Add the water resource selection constraint dynamically
    #         model.addConstr(
    #             model._vars_x_ijk[i, j, k] == model._vars_w_ijlk.sum(i, j, k, '*')
    #         )
    #
    #         print(f"🔴 Constraint Added: x_ijk({i}, {j}, {k}) == w_ijlk.sum({i}, {j}, {k}, '*')")
    #
    #         # Constraint 9 - Add the water resource connections for refilling
    #         # Find all water resource nodes l that exist in s_ijkw_links for (i, j, k)
    #         # Add constraints for each water resource l
    #
    #         for l in  mip_inputs.water_node_id:
    #             # Add lazy constraint
    #             model.addConstr(
    #                 2 * model._vars_w_ijlk[i, j, k, l] ==
    #                 model._vars_x_ijk.sum(i, l, k) + model._vars_x_ijk.sum(l, j, k)
    #             )
    #             print(
    #                 f"🔴 Constraint Added: 2 * w_ijlk({i}, {j}, {k}, {l}) = x_ijk.sum({i}, {l}, {k}) + x_ijk.sum({l}, {j}, {k})")
    #
    #         # Add constraints to determines arrival times to the nodes
    #         # Add lazy constraint
    #         i_to_water_coef = {key: v for key, v in mip_inputs.links_durations.items() if
    #                            key[0] == i and key[1] in mip_inputs.water_node_id and key[2] == k}
    #         water_to_j_coef = {key: v for key, v in mip_inputs.links_durations.items() if
    #                            key[0] in mip_inputs.water_node_id and key[1] == j and key[2] == k}
    #
    #         model.addConstr(
    #             model._vars_tv_j[j] >= model._vars_tv_j[i] + model._vars_lv_j[i] +
    #             model._vars_x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') +
    #             model._vars_x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') -
    #             mip_inputs.M_16[(i, j)] * (1 - model._vars_x_ijk.sum(i, j, k))
    #
    #         )
    #
    #         model.addConstr(
    #             model._vars_tv_j[j] <= model._vars_tv_j[i] + model._vars_lv_j[i] +
    #             model._vars_x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') +
    #             model._vars_x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') +
    #             mip_inputs.M_16[(i, j)] * (1 - model._vars_x_ijk.sum(i, j, k))
    #
    #         )
    #
    #         print(
    #             f"⏳ Time Constraint Added: tv_j({j}) >= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")
    #         print(
    #             f"⏳ Time Constraint Added: tv_j({j}) <= tv_i({i}) + travel via water node - M16({i}, {j}) * (1 - x_ijk({i}, {j}, {k}))")
    #
    #         # cb_instance.added_constraints.add((i, j, k))
    #
    #     print(f"🔁 Added {len(missing_constraints)} constraints. Re-optimizing...\n")
    #     model.update()
    #     iteration += 1


# v2
# model.params.TimeLimit = 3600
# model.params.MIPGap = 0.03
# model.params.NoRelHeurTime = 5
# model.params.MIPFocus = 3
# model.params.Cuts = 3
# model.params.Presolve = 2
# model.params.BranchDir = 1
#


# default parameters
# model.params.TimeLimit = 1200
# model.params.MIPGap = 0.03
#
# # heuristic parameters
# model.params.TimeLimit = max(120, mip_inputs.exact_run_time/10)
# model.params.MIPGap = 0.03
# model.params.NoRelHeurTime = 40
# model.params.Presolve = 2
# model.params.MIPFocus = 1
#

# model.params.TimeLimit = 1200
# model.params.MIPGap = 0.03
# model.params.MIPFocus = 2
# model.params.CliqueCuts = 2
# model.params.Cuts = 2
# model.params.Presolve = 2
# model.params.BranchDir = 1
# model.params.Heuristics = 0.1
# model.params.ImproveStartGap = 0.1
# model.params.NoRelHeurTime = 120

# model.params.LogFile = "gurobi_log"
# model.params.Heuristics = 0.2
# model.params.Threads = 8


# model.update()
# model.write("model_hand2.lp")
# (23.745 - 23.39) == (24.1-23.745)
# 23.745 - 23.390
# 0.455*0.355

    # Constraint xx - valid inequality cuts
    #
    # for i in mip_inputs.fire_ready_node_ids:
    #     print(i)
    #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
    #     i_neighborhood_size = len(i_neighborhood)
    #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * b_j[i])
    #
    # for i in mip_inputs.fire_ready_node_ids:
    #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
    #     i_neighborhood_size = len(i_neighborhood)
    #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * s2_i[i])

