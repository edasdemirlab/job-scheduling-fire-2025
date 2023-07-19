library(tidyverse )
library(readxl)

# read inputted user parameters 
problem_parameters <- as.data.frame(read_excel('generator_parameters.xlsx', sheet = "parameters"))
region_side_length <- problem_parameters %>% filter(parameter == "region_side_length") %>% pull(value)
node_side_length <- problem_parameters %>% filter(parameter == "node_side_length") %>% pull(value)
base_node_id <- problem_parameters %>% filter(parameter == "base_node_id") %>% pull(value)
n_of_fire_proof_nodes <- problem_parameters %>% filter(parameter == "number_of_fire_proof_nodes") %>% pull(value)
n_of_water_resources <- problem_parameters %>% filter(parameter == "number_of_water_resources") %>% pull(value)
number_of_fires_at_start <- problem_parameters %>% filter(parameter == "number_of_fires_at_start") %>% pull(value)
fire_degradation_level <- problem_parameters %>% filter(parameter == "fire_degradation_level") %>% pull(value)
vehicle_speed <- problem_parameters %>% filter(parameter == "vehicle_speed") %>% pull(value)


# fixed parameters
region_area <- region_side_length^2  # overall region area in km^2
node_area <- node_side_length^2  # node area in km^2
number_of_nodes_in_region <- region_area/node_area

# # map parameters
# if (number_of_nodes_in_region <= 12) {
#   number_of_spread_rate_class = 3
#   number_of_value_class = 3
# } else if (number_of_nodes_in_region <= 25) {
#   number_of_spread_rate_class = 5
#   number_of_value_class = 5
# } else {
#   number_of_spread_rate_class = 7
#   number_of_value_class = 7
# }
number_of_spread_rate_class <- 4
number_of_value_class <- 4
water_node_id <- c(5, 13, 21) 

node_available <- setdiff(1:number_of_nodes_in_region, c(base_node_id,water_node_id)) 
# fire proof node list
fire_proof_node_list <- sample(node_available, n_of_fire_proof_nodes, replace = FALSE)
node_available <- setdiff(node_available, fire_proof_node_list)

# create fires
# active_fire_list <- sample(node_available, number_of_fires_at_start, replace = FALSE)
# node_available <- setdiff(node_available, active_fire_list)
# create fires
# active_fire_list <- c(3, 11, 15, 21)
active_fire_list <- sample(node_available, number_of_fires_at_start, replace = FALSE)



# node_available_for_spread <- 1:number_of_nodes_in_region
# spread_sample_size <- (number_of_nodes_in_region/number_of_spread_rate_class)
# 
# node_spread_list <- list()
# while (length(node_available_for_spread) > 0) {
#   sampled_set = sample(node_available_for_spread, spread_sample_size, replace = FALSE)
#   node_spread_list = append(node_spread_list, list(sampled_set))
#   node_available_for_spread = setdiff(node_available_for_spread, sampled_set)
# }

# create fire spread clusters
grid_spread_low <- c(1:5, 10, 15, 20)
grid_spread_moderate <- c(21:25)
grid_spread_high <- c(6, 7, 11, 16, 17, 5, 10, 15, 20, 25)
grid_spread_extreme <- c(8, 9, 12, 14, 18, 19)
node_spread_list <- list(grid_spread_low, grid_spread_moderate, grid_spread_high, grid_spread_extreme)

# create valuable regions
grid_value_low <- c(1:5, 21:25)
grid_value_moderate <- c(11, 16, 21, 22, 23)
grid_value_high <- c(6:10)
grid_value_extreme <- c(12, 14, 17, 18, 19)
node_value_list <- list(grid_value_low, grid_value_moderate, grid_value_high, grid_value_extreme)


# node attributes
value_at_start_list <- c(0.4, 0.6, 0.8, 1)
fire_degradation_rate_list <- c(0.4, 0.6, 0.8, 1) # rep(c(0, 0.25, 0, 0.5, 0, 0.75, 0, 1, 0), each = (region_side_length/grid_side_length)_in_each_cluster) # km^2/h for linear case
fire_amelioration_rate_list <- c(0.4, 0.3, 0.2, 0.1) # rep(c(0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0), each = (region_side_length/grid_side_length)_in_each_cluster) # km^2/h for linear case


# Following calculations are done automatically according to the above user inputs
# create problem layout
region_locations <- seq(node_side_length/2, region_side_length - node_side_length/2, by = node_side_length) 
x_coordinate <- rep(region_locations, each = (region_side_length/node_side_length)) # x coordinates of targets
y_coordinate <- rep(region_locations, (region_side_length/node_side_length))
coordinates <- cbind(x_coordinate, y_coordinate)


# prepare fire map data frame
# node states --> 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fire proof, 5: water
coordinates <- as.data.frame(cbind(node_id = 1:nrow(coordinates), coordinates, node_value_at_start = value_at_start_list[1], node_degradation_rate = 0,  node_amelioration_rate = 0, node_state = 0))
coordinates[water_node_id,"node_state"] <- 5

# active fires
coordinates <- coordinates %>% mutate(node_state = replace(node_state, node_id %in% active_fire_list, 1))

# fireproof regions
coordinates <- coordinates %>% mutate(node_state = replace(node_state, node_id %in% fire_proof_node_list, 4))


# spread attributes
for (i in (1:number_of_spread_rate_class)) {
  coordinates <- coordinates %>% mutate(node_degradation_rate = replace(node_degradation_rate, node_id %in% node_spread_list[[i]], fire_degradation_rate_list[i]))
  coordinates <- coordinates %>% mutate(node_amelioration_rate = replace(node_amelioration_rate, node_id %in% node_spread_list[[i]], fire_amelioration_rate_list[i]))
}


# value attributes
for (i in (1:number_of_value_class)) {
  coordinates <- coordinates %>% mutate(node_value_at_start = replace(node_value_at_start, node_id %in% node_value_list[[i]], value_at_start_list[i]))
}

coordinates <- coordinates %>% mutate(node_value_at_start  = replace(node_value_at_start, node_id %in% fire_proof_node_list, 0))
coordinates <- coordinates %>% mutate(node_degradation_rate  = replace(node_degradation_rate, node_id %in% fire_proof_node_list, 0))


coordinates[water_node_id, "node_amelioration_rate"] <- NA
coordinates[water_node_id,"node_degradation_rate"] <- NA
coordinates[water_node_id,"node_value_at_start"] <- NA

neighborhood_list <- lapply(1:number_of_nodes_in_region, function(x){
  increment <- (region_side_length/node_side_length)
  if(x >= increment){
    if(x%%increment == 0){
      c(x - increment, x + increment, x - 1)
    }else if(x%%increment == 1){
      c(x - increment, x + increment, x + 1)
      
    }else{
      c(x - increment, x + increment, x - 1, x + 1)
    }
  } else{
    return(c(x + increment, x - 1, x + 1))
  }})
  
neighborhood_list <- lapply(neighborhood_list, function(x){
  setdiff(x, setdiff(0:100, 1:number_of_nodes_in_region))
})
 
