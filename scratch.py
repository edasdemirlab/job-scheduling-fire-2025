
def refill_lazy_constrained_model_solve(mip_inputs):

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


        if mip_inputs.experiment_mode in ["single_run", "single_run_strengthen", "single_run_hybrid", "cluster_first"]:
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





elif experiment_mode == "combination_run_from_file":
    user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
    combination_results_file_name = user_inputs.parameters_df.loc["combination_results_file_name", "value"]
    combination_results_df = pd.read_csv(os.path.join('inputs', combination_results_file_name), index_col=0)
    # Loop through each row in combination_results_df
    user_inputs.problem_data_df_original = user_inputs.problem_data_df.copy()
    user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))

    for _, row in combination_results_df.iterrows():
        user_inputs.problem_data_df = user_inputs.problem_data_df_original.copy()
        # Read initial_fire_node_IDs and convert to a list of integers
        fire_node_ids = str(row["initial_fire_node_IDs"]).split(",")
        fire_node_ids = [int(node.strip()) for node in fire_node_ids]  # Ensure clean integer conversion

        # Find matching rows in user_inputs.problem_data_df and update the 'state' column
        user_inputs.problem_data_df.loc[user_inputs.problem_data_df["node_id"].isin(fire_node_ids), "state"] = 1

        # Run the necessary functions
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_solve.exact_model_solve(mip_inputs)