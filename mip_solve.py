import gurobipy as gp
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
        current_var.append(round(v.X, 2))
        var_df.loc[counter] = current_var
        counter = counter + 1
        # with open("./math_model_outputs/" + 'mip-results.txt',
        #           "w") as f:  # a: open for writing, appending to the end of the file if it exists
        #     f.write(','.join(map(str, current_var)) + '\n')
        # print(','.join(map(str,current_var )))
    return var_df


def mathematical_model_solve(mip_inputs):
    # the formulation is available at below link:
    # https://docs.google.com/document/d/1cCx4SCTII76LPAp1McpIxybUQPRcqfJZxiNHsSsYXQ8/

    model = gp.Model("firefighting")  # Carvana Supply Chain Optimizer

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
        vtype=GRB.CONTINUOUS,
        name="ts_j",
    )

    tm_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="tm_j",
    )

    te_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="te_j",
    )

    tv_h = model.addVar(
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="tv_h",
    )

    tv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="tv_j",
    )

    p_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
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

    penalty_coef_spread = 9999
    penalty_coef_return_time = 10 ** -6

    # obj_penalize_fire_spread = gp.quicksum(z_ij[l] for l in mip_inputs.neighborhood_links)
    obj_penalize_operation_time = penalty_coef_return_time * tv_h
    # model.setObjective(obj_max - penalty_coef_spread * obj_penalize_fire_spread - penalty_coef_return_time * obj_penalize_operation_time) #
    model.setObjective(obj_max - obj_penalize_operation_time)


    # # forced solution
    # model.addConstr(x_ijk.sum(1, 4, 1) == 1)
    # model.addConstr(x_ijk.sum(4, 3, 1) == 1)
    # model.addConstr(x_ijk.sum(3, 5, 1) == 1)
    # model.addConstr(x_ijk.sum(1, 8, 2) == 1)
    # model.addConstr(x_ijk.sum(8, 9, 2) == 1)

    # model.addConstr(z_ij[(4, 5)] == 0)
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


    # Constraint 8 - water resource selection for refilling
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(x_ijk.sum(i[0], i[1], i[2]) == w_ijlk.sum(i[0], i[1], i[2], '*'))

    # Constraint 9 - water resource connections for refilling
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(2 * w_ijlk[i] <= x_ijk.sum(i[0], i[3], i[2]) + x_ijk.sum(i[3], i[1], i[2]) )

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
        # model.addConstr(tv_h >= tv_j[j] +
        #                 gp.quicksum(x_ijk[(j, mip_inputs.base_node_id, k)] * mip_inputs.links_durations[
        #                     (j, mip_inputs.base_node_id, 1)] for k in mip_inputs.vehicle_list) -
        #                 M_13[j] * (1 - x_ijk.sum(j, mip_inputs.base_node_id, '*')))

    # Constraint 14 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] <= x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + mip_inputs.M_13[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 15 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] >= x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_13[j] * (
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
                tv_j[j] <= tv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
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
                tv_j[j] >= tv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
                    water_to_j_coef, mip_inputs.water_node_id, j, '*') - mip_inputs.M_16[(i, j)] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 18 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= mip_inputs.M_13[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 19 - vehicle arrival has to be after fire arrival (start)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] - ts_j[j] >= mip_inputs.M_19 * (x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') - 1))

    # Constraint 20 - vehicle can not arrive after the fire finished
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= te_j[j])


    # equations linking fire arrivals and scheduling decisions
    # Constraint 21 - fire spread case 1: t_v =0 --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(mip_inputs.M_21[i] * tv_j[i] >= (1-s1_i[i]))

    # Constraint 22 - fire spread case that allows case 2 and 3: t_v > 0
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] <=  mip_inputs.M_22[i] * s4_i[i])

    # Constraint 23 - fire spread case  2: t_v > 0 and t_v >= t_m --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[i] - tm_j[i] + (10 ** -6) <= mip_inputs.M_23[i] * s2_i[i])

    # Constraint 24 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[i] - tv_j[i] <= mip_inputs.M_24 * (s1_i[i] + s3_i[i]))

    # Constraint 25 - fire spread cases: only one of case 1, i.e. t_v=0, and case 2, i.e. t_v>0, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s1_i[i] + s4_i[i] == 1)

    # Constraint 26 - fire spread cases: only one of case 3, i.e. t_v>=t_m, and case 4, i.e. t_v<t_m, can occur
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(s4_i[i] >= s2_i[i] + s3_i[i])

    # Constraint 27 - fire spread cases: if there is no fire in node i, it cannot spread to the adjacent nodes
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * y_j[i])

    # Constraint 28 - fire spread cases:  there is fire in node i, but no vehicle process it, i.e. t_v=0
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * (s1_i[i] + y_j[i] - 1))

    # Constraint 29 - fire spread cases:  there is fire in node i, a vehicle process it after it max point, i.e. t_v>=t_m --> it must spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') >= i_neighborhood_size * s2_i[i])

    # Constraint 30 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i]
        i_neighborhood_size = len(i_neighborhood)
        model.addConstr(z_ij.sum(i, '*') <= i_neighborhood_size * (1-s3_i[i]))

    # Constraint 31 - if a fire spreads to an adjacent node, a fire must arrive to the adjacent node.
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * y_j[j] >= z_ij.sum('*', j))

    # Constraint 32 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(y_j[j] >= x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))  # x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 33 - active fires at start
    model.addConstr(gp.quicksum(y_j[j] for j in mip_inputs.set_of_active_fires_at_start) == len(
        mip_inputs.set_of_active_fires_at_start))

    # Constraint 34 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * q_ij.sum('*', j) >= z_ij.sum('*', j))

    # Constraint 35 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(q_ij.sum('*', j) <= 1)

    # Constraint 36 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        temp_neighborhood_list = [x for x in mip_inputs.node_object_dict[j].get_neighborhood_list() if x not in mip_inputs.fire_proof_node_list]
        for i in temp_neighborhood_list:  #for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
            model.addConstr(q_ij.sum(i, j) <= z_ij.sum(i, j))

    # Constraint 37 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        model.addConstr(ts_j[ln[1]] <= tm_j[ln[0]] + mip_inputs.M_37 * (1 - z_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]) + M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_37 * (1 - z_ij[n]))

    # Constraint 38 - determine fire arrival (spread) time
    for ln in mip_inputs.neighborhood_links:
        if ln[1] in mip_inputs.set_of_active_fires_at_start:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]) - mip_inputs.M_37)
        else:
            model.addConstr(ts_j[ln[1]] >= tm_j[ln[0]] - mip_inputs.M_37 * (2 - z_ij[ln] - q_ij[ln]))
        # if n[1] in mip_inputs.set_of_active_fires_at_start:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]) - M_30)
        # else:
        #     model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_37 * (1 - z_ij[n]) - M_31 * (1 - q_ij[n]))

    # Constraint 39 - determine fire arrival (spread) time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(ts_j[j] <= mip_inputs.M_37 * z_ij.sum('*', j))

    # Constraint 40 - start time of active fires
    model.addConstr(gp.quicksum(ts_j[j] for j in mip_inputs.set_of_active_fires_at_start) == 0)

    # Constraint 41 - determine fire spread time (the time at which the fire reaches its maximum size)
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[j] == ts_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_degradation_rate()))

    # Constraint 42 - fire end time when it is not processed and burned down by itself
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(te_j[j] == tm_j[j] + (mip_inputs.node_area / mip_inputs.node_object_dict[j].get_fire_amelioration_rate()))

    # model.params.DualReductions = 0

    model.ModelSense = -1  # set objective to maximization
    start_time = time.time()
    model.params.MIPFocus = 3
    # model.params.MIPGap = 0.01
    model.params.Presolve = 2

    # model.update()
    # model.write("model_hand.lp")


    model.optimize()
    end_time = time.time()
    run_time_cpu = round(end_time - start_time, 2)

    # for c in model.getConstrs():
    #     if c.Slack < 1e-6:
    #         print('Constraint %s is active at solution point' % (c.ConstrName))
    #

    if model.Status == GRB.Status.INFEASIBLE:
        max_dev_result = None
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("Go check infeasible_model.ilp file")
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

        p_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        p_j_results_df = model_organize_results(p_j.values(), p_j_results_df)

        s_ijkw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id', 'value'])
        s_ijkw_results_df = model_organize_results(w_ijlk.values(), s_ijkw_results_df)

        # model global results
        # obj_result = model.objval + (penalty_coef_spread * sum(z_ij_results_df.loc[:, 'value'])) + penalty_coef_return_time * tv_h.X
        obj_result = model.objval + penalty_coef_return_time * tv_h.X




        global_results_df = pd.DataFrame(columns=['total_value', 'model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time', 'python_time'])
        global_results_df.loc[len(global_results_df.index)] = [obj_result, model.objval, model.objbound, model.mipgap, model.runtime, run_time_cpu]




        writer_file_name = os.path.join('outputs', "results_{0}_nodes_{1}.xlsx".format(mip_inputs.n_nodes, str(datetime.now().strftime('%Y_%m_%d_%H_%M'))))


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
        p_j_results_df.to_excel(writer, sheet_name='p_j_results')
        s_ijkw_results_df.to_excel(writer, sheet_name='s_ijkw_results')

        writer.close()
