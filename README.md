# Scheduling and Routing with Degradation-Triggered Job Arrivals: An Application to Aerial Forest Firefighting with a UAV Fleet

Authors:
- Erdi Dasdemir (1)
- Esther Jose (2)
- Rajan Batta (2)

(1) Department of Industrial Engineering, Hacettepe University, 06800 Ankara, Turkey

(2) Department of Industrial and Systems Engineering, University at Buffalo (SUNY), Buffalo, NY 14260

Reach out to Erdi Dasdemir (edasdemir@hacettepe.edu.tr) or Esther Jose (estherjo@buffalo.edu) for your further questions.

## Installation

The codebase requires Python and the Gurobi solver installed on your computer. The required Python packages are listed in `requirements.txt`.

To deploy the code, copy the `requirements.txt` file to your desired location, then activate your virtual environment and install the packages with the following commands in your terminal:

```bash
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```

## Codebase Structure

The codebase has the following files:
- user_application.py
- case_generator_setup.py
- mip_setup.py
- mip_solve.py

Users do not need to change any of these codes to run the model. It is enough to modify the input file located in the inputs folder.

The `inputs` folder contains a spreadsheet file (`inputs_to_load.xlsx`) that serves as a graphical interface. To use, simply modify this file as needed; no changes to the code are necessary. Prepare your input file and run `python user_application.py`.

You can create new problem instances or find optimal solutions for your specific instances by preparing the necessary inputs in the spreadsheet.

## Run Modes
The `inputs_to_load.xlsx` spreadsheet in the `inputs` folder has two sheets:
- `inputs_df`
- `parameters`.

### inputs_df

This sheet is for entering the inputs of your problem instance. The columns are as follows:

- `node_id`: A unique ID you define for the nodes (job locations).
- `x_coordinate`, `y_coordinate`: Coordinates in a two-dimensional space.
- `value_at_start`: The initial value of the node when it is job-free.
- `value_degradation_rate`: The rate at which the node's value decreases after a job arrives.
- `fire_degradation_rate`: The rate at which job demand degrades (e.g., fire size increases) after the fire starts.
- `fire_amelioration_rate`: The rate at which job demand improves (e.g., fire size decreases as there's nothing left to burn).
- `state`: The initial state of the node. For details, refer to the node states described in the paper. Briefly, 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fireproof, 5: water, 6: home/base.
- `neighborhood_list`: A list of adjacent nodes. For example, `[6, 2]` indicates that nodes 6 and 2 are adjacent. See the example input file for more details.

### parameters

This sheet requires defining problem and experimentation parameters.

There are three modes: `instance_generation`, `single_run`, `combination_run`.

**instance_generation**:
This mode is used to create new instances based on the California wildfires case study described in the paper. The following parameters are required:
- `number_of_initial_fires`: How many nodes have fire at the start.
- `number_of_grids_at_a_side`: Defines the region size (e.g., 7 for a 7x7 region).
- `number_of_areas_of_different_types`, `number_of_water_bodies`, `number_of_blocks`, `default_housing_density`, `default_vegetation_density`, `fire_degradation_rate_min`, `fire_degradation_rate_max`, `region_value_min`, `region_value_max`: Refer to the paper for detailed descriptions and values.

**single_run**:
This mode runs the model for the inputs defined in `inputs_df`. Additional parameters required in the `parameters` sheet include `n_nodes`, `region_side_length`, `node_area`, `n_vehicles`, `vehicle_flight_speed`, `time_limit`.

**combination_run**:
This mode replicates the combination runs conducted in the paper. It uses the inputs in `inputs_df` but requires the sheet to have no nodes with states indicating initial fire at the start. Then, the combination mode will systematically create combinations of different initial fires for the experimental study as described in the paper.
