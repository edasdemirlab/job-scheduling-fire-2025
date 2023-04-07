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

    # Big M values
    M_1 = 999
    M_2 = 999
    M_3 = 999
    M_4 = 999
    M_5 = 999
    M_6 = 9999
    M_7 = 999
    M_8 = 999
    M_9 = 999
    M_10 = 999
    M_11 = 999


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

    y_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="y_j",
    )

    w_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        vtype=GRB.BINARY,
        name="w_j",
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

    # set objective
    model.setObjective(
        gp.quicksum(p_j[j] for j in mip_inputs.fire_ready_node_ids), sense=GRB.MAXIMIZE)

    # Add constraints
    # Constraint 2 - helper to the objective function, i.e. determines collected prizes from each region
    model.addConstrs(p_j[j] <= mip_inputs.node_object_dict[j].get_value_at_start() - mip_inputs.node_object_dict[
        j].get_value_degradation_rate() * tv_j[j] - mip_inputs.node_object_dict[j].get_value_at_start() * w_j[j] for j
                     in mip_inputs.fire_ready_node_ids)

    # Constraint 3 - time limit
    model.addConstr(tv_h <= mip_inputs.time_limit)

    # Constraint 4 - determines return time to the base, considering the time of vehicle with maximum return time
    # i=3
    for i in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_h >= tv_j[i] +
                        gp.quicksum(x_ijk[(i, mip_inputs.base_node_id, k)] * mip_inputs.links_durations[
                            (i, mip_inputs.base_node_id, 1)] for k in mip_inputs.vehicle_list) -
                        M_1 * (1 - x_ijk.sum(i, mip_inputs.base_node_id, '*')))

    # # # Constraint 5 - k vehicle leave the bas
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.node_list, k) == 1)

    for k in mip_inputs.vehicle_list:  # k vehicle returns the bas
        model.addConstr(x_ijk.sum(mip_inputs.node_list, mip_inputs.base_node_id, k) == 1)

    # Constraint 6 - flow balance equation -- incoming vehicles must be equal to the outgoing vehicles at each node
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.node_list, j, k) == x_ijk.sum(j, mip_inputs.node_list, k))

    # Constraint 7 - at most one vehicle can visit a node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(x_ijk.sum(mip_inputs.node_list, j, '*') <= 1)

    # Constraint 8 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(y_j[j] >= x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 9
    for i in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(i, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(i, mip_inputs.water_node_id, k))

    # Constraint 10
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids, j, k) == x_ijk.sum(mip_inputs.water_node_id, j, k))

    # job degradation and creation constraints
    # Constraint 11 - determine fire spread
    for i in mip_inputs.neighborhood_links:
        model.addConstr(tv_j[i[0]] - tm_j[i[0]] <= M_2 * z_ij[
            i] - 10 ** -3)  # we have 10^-3 as strict inequality constraints are not allowed in gurobi

    # Constraint 12
    for i in mip_inputs.neighborhood_links:
        model.addConstr(y_j[i[0]] - M_3 * tv_j[i[0]] <= z_ij[i])

    # Constraint 13
    for n in mip_inputs.neighborhood_links:
        if n[1] in mip_inputs.set_of_active_fires_at_start:
            model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_4 * (1 - z_ij[n]) + M_5)
        else:
            model.addConstr(ts_j[n[1]] <= tm_j[n[0]] + M_4 * (1 - z_ij[n]))

    # Constraint 14
    for n in mip_inputs.neighborhood_links:
        if n[1] in mip_inputs.set_of_active_fires_at_start:
            model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_4 * (1 - z_ij[n]) - M_6 * (1 - q_ij[n]) - M_5)
        else:
            model.addConstr(ts_j[n[1]] >= tm_j[n[0]] - M_4 * (1 - z_ij[n]) - M_6 * (1 - q_ij[n]))

    # Constraint 15
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(M_7 * q_ij.sum('*', j) >= z_ij.sum('*', j))

    # Constraint 16
    for j in mip_inputs.fire_ready_node_ids:
        for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
            model.addConstr(q_ij.sum(i, j) <= z_ij.sum(i, j))

    # Constraint 17
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(q_ij.sum('*', j) <= 1)

    # Constraint 18
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(ts_j[j] <= M_8 * z_ij.sum('*', j))

    # Constraint 19
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tm_j[j] == ts_j[j] + 1 / mip_inputs.node_object_dict[j].get_fire_degradation_rate())

    # Constraint 20
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(te_j[j] == tm_j[j] + 1 / mip_inputs.node_object_dict[j].get_fire_amelioration_rate())

    # Constraint 21
    for j in mip_inputs.fire_ready_node_ids:
        j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        model.addConstr(j_neighborhood_size * y_j[j] >= z_ij.sum('*', j))

    # Constraint 22
    model.addConstr(gp.quicksum(y_j[j] for j in mip_inputs.set_of_active_fires_at_start) == len(
        mip_inputs.set_of_active_fires_at_start))

    # Constraint 23
    model.addConstr(gp.quicksum(ts_j[j] for j in mip_inputs.set_of_active_fires_at_start) == 0)

    # Constraint 24
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == i and k[1] == mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == mip_inputs.water_node_id and k[1] == j}
            model.addConstr(tv_j[j] <= tv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') + M_9 * (1 - x_ijk.sum(i, j, '*')))

    # # Constraint 24
    # for i in mip_inputs.fire_ready_node_ids:
    #     to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
    #     for j in to_j_list:
    #         j_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == i and k[1] == j}
    #         model.addConstr(tv_j[j] <= tv_j[i] + x_ijk.prod(j_coef, i, j, '*') + M_9 * (1 - x_ijk.sum(i, j, '*')))
    #

    # Constraint 25
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == i and k[1] == mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == mip_inputs.water_node_id and k[1] == j}
            model.addConstr(tv_j[j] >= tv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(water_to_j_coef, mip_inputs.water_node_id, j, '*') - M_9 * (1 - x_ijk.sum(i, j, '*')))

    # # Constraint 25
    # for i in mip_inputs.fire_ready_node_ids:
    #     to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
    #     for j in to_j_list:
    #         j_coef = {k: v for k, v in mip_inputs.links_durations.items() if k[0] == i and k[1] == j}
    #         model.addConstr(tv_j[j] >= tv_j[i] + x_ijk.prod(j_coef, i, j, '*') - M_9 * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 26
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] <= x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + M_9 * (
                    1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 27
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] >= x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - M_9 * (
                    1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 28
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= M_9 * x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 29
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] - ts_j[j] >= M_10 * (x_ijk.sum(mip_inputs.node_list, j, '*') - 1))

    # Constraint 30
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= te_j[j])

    # Constraint 31
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(w_j[j] >= y_j[j] - M_11 * tv_j[j])




    # model.update()
    # model.write("model_hand.lp")

    # model.params.DualReductions = 0

    model.ModelSense = -1  # set objective to maximization
    start_time = time.time()
    model.params.MIPFocus = 2
    model.params.MIPGap = 0.01
    model.params.Presolve = 2
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

        w_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        w_j_results_df = model_organize_results(w_j.values(), w_j_results_df)

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

        # model global results
        global_results_df = pd.DataFrame(columns=['obj_value', 'obj_bound', 'gap', 'gurobi_time', 'python_time'])
        global_results_df.loc[len(global_results_df.index)] = [model.objval, model.objbound, model.mipgap, model.runtime, end_time]




        writer_file_name = os.path.join('outputs', "results_{0}_nodes_{1}.xlsx".format(mip_inputs.n_nodes, str(datetime.now().strftime('%Y_%m_%d_%H_%M'))))


        writer = pd.ExcelWriter(writer_file_name)
        global_results_df.to_excel(writer, sheet_name='global_results')
        x_ijk_results_df.to_excel(writer, sheet_name='x_ijk_results')
        y_j_results_df.to_excel(writer, sheet_name='y_j_results')
        z_ij_results_df.to_excel(writer, sheet_name='z_ij_results')
        q_ij_results_df.to_excel(writer, sheet_name='q_ij_results')
        w_j_results_df.to_excel(writer, sheet_name='w_j_results')

        ts_j_results_df.to_excel(writer, sheet_name='ts_j_results')
        tm_j_results_df.to_excel(writer, sheet_name='tm_j_results')
        te_j_results_df.to_excel(writer, sheet_name='te_j_results')
        tv_j_results_df.to_excel(writer, sheet_name='tv_j_results')
        p_j_results_df.to_excel(writer, sheet_name='p_j_results')
        writer.close()
