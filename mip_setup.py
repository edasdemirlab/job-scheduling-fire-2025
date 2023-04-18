import gurobipy as gp
import pandas as pd
import os
from ast import literal_eval


class NodeSetup:
    def __init__(self, attributes_given):  # attributes=values
        self.attributes = attributes_given

    def get_node_id(self):
        return self.attributes["node_id"]

    def get_x_coordinate(self):
        return self.attributes["x_coordinate"]

    def get_y_coordinate(self):
        return self.attributes["y_coordinate"]

    def get_value_at_start(self):
        return float(self.attributes["value_at_start"])

    def get_value_degradation_rate(self):
        return float(self.attributes["value_degradation_rate"])

    def get_fire_degradation_rate(self):
        return float(self.attributes["fire_degradation_rate"])

    def get_fire_amelioration_rate(self):
        return float(self.attributes["fire_amelioration_rate"])

    def get_state(self):
        return self.attributes["state"]

    def get_neighborhood_list(self):
        return self.attributes["neighborhood_list"]

# define setup classes
class InputsSetup:
    def __init__(self):

        # read problem input
        self.directory = os.path.join('inputs', 'inputs_to_load.xlsx')  # os.getcwd(),
        self.parameters_df = pd.read_excel(self.directory, sheet_name="parameters", index_col=0)
        self.problem_data_df = pd.read_excel(self.directory, sheet_name="inputs_df")
        fix_df = self.problem_data_df.copy()
        self.problem_data_df["neighborhood_list"] = fix_df["neighborhood_list"].apply(literal_eval)  # make string lists of csv to python lists
        self.distance_df = pd.read_excel(self.directory, sheet_name="distance_df")

        # problem parameters
        self.region_side_length = self.parameters_df.loc["region_side_length", "value"]
        self.node_area = self.parameters_df.loc["node_area", "value"]
        self.n_vehicles = self.parameters_df.loc["n_vehicles", "value"]
        self.vehicle_list = list(range(1, self.n_vehicles + 1))
        self.vehicle_flight_speed = self.parameters_df.loc["vehicle_flight_speed", "value"]
        self.time_limit = self.parameters_df.loc["time_limit", "value"]

        self.n_nodes = self.parameters_df.loc["n_nodes", "value"]
        self.node_list = list(range(1, self.n_nodes + 1))
        self.base_node_id = self.problem_data_df.query("state == 6")["node_id"].item()
        # self.water_node_id = self.problem_data_df.query("state == 5")["node_id"].item()
        self.water_node_id = self.problem_data_df.query("state == 5")["node_id"].tolist()
        # self.problem_data_df.query("state not in 5")["node_id"].tolist()

        # define empty lists for classes, i.e. each element of the list will be an element of the corresponding class
        self.node_object_dict = {}  # dictionary of elements of IC attribute class
        self.links_multidict_input = {}  # multi-dictionary input for the transportation cost information for each available arc in the network

        # set neighborhood df
        # node states --> 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fire proof, 5: water, 6:home/base
        fire_proof_states = list(range(2, 7))
        fire_proof_nodes = self.problem_data_df.query("state in @fire_proof_states").loc[:,"node_id"].tolist()
        self.neighborhood_links_df = self.problem_data_df[['node_id', 'neighborhood_list']].copy()
        self.neighborhood_links_df = self.neighborhood_links_df.explode("neighborhood_list")
        self.neighborhood_links_df = self.neighborhood_links_df.rename(columns={"node_id": "from", "neighborhood_list": "to"})
        self.neighborhood_links_df = self.neighborhood_links_df.query("`from` not in @fire_proof_nodes & `to` not in @fire_proof_nodes")
        self.neighborhood_links_df = self.neighborhood_links_df.reset_index(drop=True)
        self.neighborhood_links = gp.tuplelist(list(self.neighborhood_links_df.itertuples(index=False, name=None)))

        self.set_of_active_fires_at_start = self.problem_data_df.query("`state`==1")["node_id"].tolist()
        flow_active_states = [0, 1, 5, 6]

        flow_active_nodes = self.problem_data_df.query("`state` in @flow_active_states")["node_id"]

        self.flow_active_distance_df = self.distance_df.query("`from` in @flow_active_nodes & `to`  in @flow_active_nodes").copy().reset_index(drop=True)

        # setup links and their distances
        for i in range(len(self.flow_active_distance_df)):
            values = self.flow_active_distance_df .loc[i, :]
            for k in range(1, self.n_vehicles + 1):
                self.links_multidict_input[int(values["from"]), int(values["to"]), k] = (values["distance"] / self.vehicle_flight_speed)

        # setup multi dictionaries --> if you are unfamiliar with multidict and want to learn about, go to below link
        # https://www.gurobi.com/documentation/8.1/refman/py_multidict.html
        self.links, self.links_durations = gp.multidict(self.links_multidict_input)

        self.fire_ready_nodes_data_df = self.problem_data_df.query("`node_id` not in @fire_proof_nodes").copy().reset_index(drop=True)

        # setup nodes and their attributes
        for nd in self.fire_ready_nodes_data_df["node_id"]:
            self.node_object_dict[nd] = NodeSetup(self.fire_ready_nodes_data_df.query('node_id == @nd').iloc[0, :])

        self.fire_ready_node_ids = list(self.fire_ready_nodes_data_df["node_id"])
        self.fire_ready_node_ids_and_base =  [self.base_node_id]  + self.fire_ready_node_ids


        self.s_ijkw_links = gp.tuplelist()
        for i in self.fire_ready_node_ids:
            to_j_list = [x for x in self.fire_ready_node_ids if x != i]
            for j in to_j_list:
                for k in self.vehicle_list:
                    for w in self.water_node_id:
                        self.s_ijkw_links.append((i, j, k, w))





        # self.node_vehicle_pairs = gp.tuplelist([(j, k) for j in self.fire_ready_nodes_data_df["node_id"].tolist() for k in self.vehicle_list])
        #


