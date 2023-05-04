library(tidyverse )
library(readxl)

# read inputted user parameters 
problem_parameters <- as.data.frame(read_excel('generator_parameters.xlsx', sheet = "parameters"))
region_side_length <- problem_parameters %>% filter(parameter == "region_side_length") %>% pull(value)
node_side_length <- problem_parameters %>% filter(parameter == "node_side_length") %>% pull(value)
base_node_id <- problem_parameters %>% filter(parameter == "base_node_id") %>% pull(value)
n_of_fire_proof_nodes <- problem_parameters %>% filter(parameter == "n_of_fire_proof_nodes") %>% pull(value)
n_of_water_resources <- problem_parameters %>% filter(parameter == "n_of_water_resources") %>% pull(value)
n_of_fires_at_start <- problem_parameters %>% filter(parameter == "n_of_fires_at_start") %>% pull(value)
fire_degradation_level <- problem_parameters %>% filter(parameter == "fire_degradation_level") %>% pull(value)
vehicle_speed <- problem_parameters %>% filter(parameter == "vehicle_speed") %>% pull(value)


# fixed parameters
region_area <- region_side_length^2  # overall region area in km^2
node_area <- node_side_length^2  # grid area in km^2
n_nodes_in_region <- region_area/node_area

# map parameters
if (n_nodes_in_region <= 12) {
  n_spread_rate_class = 3
  n_value_class = 3
} else if (n_nodes_in_region <= 25) {
  n_spread_rate_class = 5
  n_value_class = 5
} else {
  n_spread_rate_class = 7
  n_value_class = 7
}



# create fires
active_fire_list <- c(3, 4, 9)

# fire proof grid list
fire_proof_grid_list <- c(2)

# create fire spread clusters
grid_spread_low <- c(1:3)
grid_spread_moderate <- c(4:6)
grid_spread_high <- c(7)
grid_spread_extreme <- c(8, 9)
grid_spread_list <- list(grid_spread_low, grid_spread_moderate, grid_spread_high, grid_spread_extreme)

# create valuable regions
grid_value_low <- c(8:9)
grid_value_moderate <- c(5:7)
grid_value_high <- c(4)
grid_value_extreme <- c(1:3)
grid_value_list <- list(grid_value_low, grid_value_moderate, grid_value_high, grid_value_extreme)


# grid attributes
value_at_start_list <- c(0.4, 0.6, 0.8, 1)
fire_degradation_rate_list <- c(0.4, 0.6, 0.8, 1) # rep(c(0, 0.25, 0, 0.5, 0, 0.75, 0, 1, 0), each = (region_side_length/node_side_length)_in_each_cluster) # km^2/h for linear case
fire_amelioration_rate_list <- c(0.4, 0.3, 0.2, 0.1) # rep(c(0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0), each = (region_side_length/node_side_length)_in_each_cluster) # km^2/h for linear case


# Following calculations are done automatically according to the above user inputs
# create problem layout
region_locations <- seq(node_side_length/2, region_side_length - node_side_length/2, by = node_side_length) 
x_coordinate <- rep(region_locations, each = (region_side_length/node_side_length)) # x coordinates of targets
y_coordinate <- rep(region_locations, (region_side_length/node_side_length))
coordinates <- cbind(x_coordinate, y_coordinate)


# prepare fire map data frame
# grid states --> 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fire proof, 5: water
coordinates <- as.data.frame(cbind(grid_id = 1:nrow(coordinates), coordinates, grid_value_at_start = value_at_start_list[1], grid_degradation_rate = 0,  grid_amelioration_rate = 0, grid_state = 0))
water_grid_id <- base_node_id # coordinates %>% filter(between(x_coordinate, 2, 3) & between(y_coordinate, 2, 3)) %>% pull(grid_id)
coordinates[water_grid_id,"grid_state"] <- 5

# active fires
coordinates <- coordinates %>% mutate(grid_state = replace(grid_state, grid_id %in% active_fire_list, 1))

# fireproof regions
coordinates <- coordinates %>% mutate(grid_state = replace(grid_state, grid_id %in% fire_proof_grid_list, 4))


# spread attributes
for (i in (1:n_spread_rate_class)) {
  coordinates <- coordinates %>% mutate(grid_degradation_rate = replace(grid_degradation_rate, grid_id %in% grid_spread_list[[i]], fire_degradation_rate_list[i]))
  coordinates <- coordinates %>% mutate(grid_amelioration_rate = replace(grid_amelioration_rate, grid_id %in% grid_spread_list[[i]], fire_amelioration_rate_list[i]))
}


# value attributes
for (i in (1:n_value_class)) {
  coordinates <- coordinates %>% mutate(grid_value_at_start = replace(grid_value_at_start, grid_id %in% grid_value_list[[i]], value_at_start_list[i]))
}

coordinates <- coordinates %>% mutate(grid_value_at_start  = replace(grid_value_at_start, grid_id %in% fire_proof_grid_list, 0))


coordinates[water_grid_id, "grid_amelioration_rate"] <- NA
coordinates[water_grid_id,"grid_degradation_rate"] <- NA
coordinates[water_grid_id,"grid_value_at_start"] <- NA

neighborhood_list <- lapply(1:n_nodes_in_region, function(x){
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
  setdiff(x, setdiff(0:100, 1:n_nodes_in_region))
})
 
