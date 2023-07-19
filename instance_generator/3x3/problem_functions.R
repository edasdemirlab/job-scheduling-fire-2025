
generate_dist_matrix <- function(param){
  dist_matrix <- as.matrix(dist(param$coordinates[,c('x_coordinate','y_coordinate')], method = "euclidean", diag = TRUE, upper = TRUE, p = 2))
  dist_matrix <- dist_matrix/param$vehicle_speed # in terms of travel time
  return(dist_matrix)
}


# generate_region_map_df <- function(param) {
#   # extract parameters to be used
#   # create fire map data frame
#   region_map_df <- as.data.frame(param$coordinates)
#   region_map_df["node_value_at_start"] <- param$node_value_at_start
#   region_map_df["node_degradation_rate"] <- param$node_degradation_rate
#   region_map_df["node_amelioration_rate"] <- param$node_amelioration_rate
#   region_map_df[param$water_node_id,"node_amelioration_rate"] = NA
#   region_map_df[param$water_node_id,"node_degradation_rate"] = NA
#   region_map_df[param$water_node_id,"node_value_at_start"] = NA
#   return(region_map_df)
# }


  
generate_fire_df <- function(param, region_map_df) {
  # check active fires
  fire_size_start <- 0
  fire_size_max <- param$node_area
  fire_map_df <- region_map_df
  fire_map_df["fire_size_now"] <- 0
  fire_map_df["fire_time_start"] <- 0
  fire_map_df <- fire_map_df %>% mutate(fire_time_max = (fire_size_max - fire_size_start) / node_degradation_rate )
  fire_map_df <- fire_map_df %>% mutate(fire_time_end = fire_time_max + (fire_size_max / node_amelioration_rate))
  fire_map_df <- fire_map_df %>% mutate(value_now = node_value_at_start)
  fire_map_df <- fire_map_df %>% mutate(value_degradation_rate = node_value_at_start / fire_time_end)
  return(fire_map_df)
}


generate_all_vehicle_df <- function(param){
  # set initial attributes of the vehicles 
  # vehicle_current_location_index <- rep(1, param$n_vehicles)
  # vehicle_previous_location <- rep(param$base_node_id, param$n_vehicles)
  # vehicle_current_location <- rep(param$base_node_id, param$n_vehicles)
  # vehicle_current_time <- rep(0, param$n_vehicles)
  # vehicle_current_status <- rep(0, param$n_vehicles) # 0 is returned to base, 1 is active in the field
  # vehicle_df <- data.frame(vehicle_id = 1:param$n_vehicles,vehicle_current_location_index, vehicle_previous_location, vehicle_current_location, vehicle_current_time, vehicle_current_status, vehicle_next_status)
  
  departure_location <- rep(param$base_node_id, param$n_vehicles)
  arrival_location <- rep(param$base_node_id, param$n_vehicles)
  departure_time <- rep(0, param$n_vehicles)
  arrival_time <- rep(0, param$n_vehicles)
  current_status <- rep(0, param$n_vehicles) # 0 is returning to base, 1 is flying to the assigned job
  next_status <- rep(1, param$n_vehicles) # 0 is returning to base, 1 is flying to the assigned job
  vehicle_df <- data.frame(vehicle_id = 1:param$n_vehicles,departure_location, arrival_location, departure_time, arrival_time, current_status, next_status)
  return(vehicle_df)
}