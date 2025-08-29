library(readxl)
library(scales)
library(ggplot2)
library(colorspace)

# functions
# Source the external script to load functions
source("decision_space_functions.R")

# read output file
input_name <- "e-rlm_single_run_results_49_nodes_2025_08_17_14_36"
input_file <- paste0(input_name,".xlsx")

# Step 2: Create a folder in the current directory using the input file's name
output_directory <- file.path(getwd(), input_name)  # Create the full path
if (!dir.exists(output_directory)) {
  dir.create(output_directory)  # Create the directory if it doesn't exist
}

# Get the names of all sheets in the Excel file
sheet_names <- excel_sheets(input_file)

# Initialize an empty list to store the data frames
df_list <- list()

# Loop through the sheet names and read each sheet
for (sheet in sheet_names) {
  # Read the current sheet and store it as a data frame in the list
  df_list[[sheet]] <- read_excel(input_file, sheet = sheet)
}

# Read inputs
inputs_problem_data_df <- df_list[["inputs_problem_data"]]
node_coordinates <- inputs_problem_data_df[, c("node_id", "x_coordinate", "y_coordinate")]



# problem parameters
base_id <- 1
water_id <- inputs_problem_data_df$node_id[inputs_problem_data_df$state == 5]
node_at_a_side <- 7
pX <- inputs_problem_data_df$x_coordinate 
pY <- inputs_problem_data_df$y_coordinate
pMat <- c()


# vehicle routes
x_ijk_results <- df_list[["x_ijk_results"]]
# Filter rows where 'value' == 1
x_ijk_results <- x_ijk_results[x_ijk_results$value == 1, ]

# Extract the value of 'n_vehicles'
n_vehicles <-  as.numeric(df_list[["inputs_parameters"]]$value[ df_list[["inputs_parameters"]]$parameter == "n_vehicles"])

# Initialize a list to store the routes for each vehicle
vehicle_routes <- list()

# Loop through vehicle_ids from 1 to n_vehicles
for (vehicle_id in 1:n_vehicles) {
  # Extract the route for the current vehicle_id
  route <- as.numeric(extract_route(vehicle_id, x_ijk_results, water_id))
  
  # Store the route in the list using the vehicle_id as the key
  vehicle_routes[[as.character(vehicle_id)]] <- route
}


# node arrival times
tv_j_results <- round_df(df_list[["tv_j_results"]],2)
t_v_list <- tv_j_results$value

# Read the second sheet into a data frame
y_j_results <- df_list[["y_j_results"]]
ts_j_results <- round_df(df_list[["ts_j_results"]],2)
tm_j_results <- round_df(df_list[["tm_j_results"]],2)
te_j_results <- round_df(df_list[["te_j_results"]],2)
r_j_results <- round_df(df_list[["p_j_results"]],2)

# Step 2: Loop through each scenario and construct a data frame for each
# Extract the relevant rows for the current scenario from each results data frame
y_j <- y_j_results
ts_j <- ts_j_results
tm_j <- tm_j_results
te_j <- te_j_results
r_j <- r_j_results
  
# each result data frame has 'node_id' and 'value' columns, combine them
scenario_data <- data.frame(
  node_id = y_j$node_id,          # Assuming all frames have a common node_id
  y_j = y_j$value,               # y_j attribute
  ts_j = ts_j$value,             # ts_j attribute
  tm_j = tm_j$value,             # tm_j attribute
  te_j = te_j$value,             # te_j attribute
  r_j = r_j$value                # r_j attribute
)
  


plot_scenarios(inputs_problem_data_df, vehicle_routes, scenario_data, output_directory)

plot_scenarios(inputs_problem_data_df, vehicle_routes, scenario_data, output_directory, include_routes = FALSE)

plot_initial_fires(inputs_problem_data_df, output_directory)

