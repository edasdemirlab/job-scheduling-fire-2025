# libraries
library(plotly)
library(gganimate)
library(ggplot2)
library(nodeExtra)
library(tidyverse)
library(ggthemes)
library(ggpubr)
library(reshape)


source("problem_functions.R", local = func <- new.env())
source("problem_parameters.R", local = param <- new.env())
source("plot_functions.R", local = ploting <- new.env())



# generate fire region data frame
#region_map_df <- func$generate_region_map_df(param)
region_map_df <- param$coordinates

# distance matrix
dist_matrix <- func$generate_dist_matrix(param)
dist_matrix_mip_model <- dist_matrix

# dist_matrix_mip_model <- rbind(dist_matrix[param$base_node_id,], dist_matrix)
# dist_matrix_mip_model <- cbind(dist_matrix_mip_model[,param$base_node_id], dist_matrix_mip_model)
# rownames(dist_matrix_mip_model) <- 0:nrow(dist_matrix)  
# colnames(dist_matrix_mip_model) <- 0:ncol(dist_matrix)  

dist_pairwise <- data.frame(from = colnames(dist_matrix_mip_model)[col(dist_matrix_mip_model)], to=rownames(dist_matrix_mip_model)[row(dist_matrix_mip_model)], dist = c(dist_matrix_mip_model)) %>% filter(if_any(from:to, ~ .x != to))
dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] == param$base_node_id & (x[2] %in% param$water_node_id)) | ((x[1] %in% param$water_node_id) & x[2] == param$base_node_id))) {return(as.numeric(x))}}))
#dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] %in% param$water_node_id & (x[2] %in% param$water_node_id)) | ((x[1] %in% param$water_node_id) & x[2] %in% param$water_node_id))) {return(as.numeric(x))}}))
dist_pairwise <- t(apply(dist_pairwise,1, function(x){if (!((x[1] %in% param$water_node_id & (x[2] %in% param$water_node_id)) | ((x[1] %in% param$water_node_id) & x[2] %in% param$water_node_id))) {return(as.numeric(x))}}))
dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] %in% param$fire_proof_node_list | (x[2] %in% param$fire_proof_node_list)))) {return(as.numeric(x))}}))


write.table(dist_pairwise, "dist_pairwise.txt")
# 
# dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] == 0 & x[2] == 1) | (x[2] == 0 & x[1] == 1))) {return(as.numeric(x))}}))

# 
# 
# neighborhood_list <- apply(dist_pairwise, 1, function(x){
#   if(x[1] != x[2]){
#     return(c(x + increment, x - 1, x + 1))
#   }})


# plot fire intensity and value maps
ploting$plot_region_maps(region_map_df)

# generate active fire data frame
fire_df <- func$generate_fire_df(param, region_map_df)
write.table(fire_df, "fire_df.txt")
param$neighborhood_list

# # plot active fire degradation and amelioration plots
# ploting$plot_active_fire_plots(active_fire_df)

# active_fire_map_dynamic_df will be dynamically updated during firefighting
fire_dynamic_df <- fire_df
# fire_dynamic_df["clock_time"] <- 0

#------------------ start dynamic job scheduling (fire fighting)------------------
# create vehicle data frame
all_vehicle_df <- func$generate_all_vehicle_df(param)


# store all vehicle and fire progress in decision_space_df which will be updated during firefighting
decision_space_df <- all_vehicle_df

there_are_jobs_to_process <- 'TRUE'
plot_counter <- 0
ploting$plot_live_map(fire_dynamic_df, plot_counter)
plot_counter <- plot_counter + 1
debug_c<-0
while (there_are_jobs_to_process) {
  debug_c = debug_c +1
  print(debug_c)
  vehicle_next_df <- t(apply(all_vehicle_df, 1, function(x){
    # vehicle_df<-all_vehicle_df[2,]
    vehicle_df <- (as.data.frame(t(x)))
    
    if (nrow(fire_dynamic_df %>% filter(node_state == 1)) > 0){
      
    
    if (vehicle_df$next_status == 1) {  # vehicle is ready to go a job
      
      next_arrival_location <- fire_dynamic_df %>% filter(node_state == 1) %>% arrange(desc(value_now)) %>% slice(1) %>% .$node_id
      next_arrival_time  <- vehicle_df$arrival_time + dist_matrix[vehicle_df$arrival_location, next_arrival_location]
      next_status <- 0 # will return to base
      

      }else{ # returning to base
      
      next_arrival_location <- param$base_node_id
      next_arrival_time  <- vehicle_df$arrival_time + dist_matrix[vehicle_df$arrival_location, next_arrival_location]
      next_status <- 1 # will return to base
      
      }

    # update the conditions of nodes with fire
    
    # update fire_size_now 
    
    # if next_arrival_time < fire_time_max --> use degradation,
    temp_before <- fire_dynamic_df %>% filter(node_state == 1 & next_arrival_time < fire_time_max) %>% mutate(fire_size_now = replace(fire_size_now, values =   0 + (node_degradation_rate * (next_arrival_time - fire_time_start))))
    # if next_arrival_time > fire_time_max --> use amelioration
    temp_after <- fire_dynamic_df %>% filter(node_state == 1 & next_arrival_time > fire_time_max) %>% mutate(fire_size_now = replace(fire_size_now, values =   1 - (node_amelioration_rate * (next_arrival_time - fire_time_max))))
    # update fire_size_now
    fire_dynamic_df[fire_dynamic_df$node_state == 1 & (next_arrival_time < fire_dynamic_df$fire_time_max),] <- temp_before
    fire_dynamic_df[fire_dynamic_df$node_state == 1 & (next_arrival_time > fire_dynamic_df$fire_time_max),] <- temp_after
    
    # update value_now
    temp_value <- fire_dynamic_df %>% filter(node_state == 1) %>% mutate(value_now  = replace(value_now, values =   node_value_at_start  - (value_degradation_rate * next_arrival_time)))
    fire_dynamic_df[fire_dynamic_df$node_state == 1, ] <- temp_value
    # update active fires that are burned down
    # a node becomes burned down if fire_size <=0 
    
    fire_dynamic_df[fire_dynamic_df$node_state == 1 & fire_dynamic_df$fire_size_now <= 0, "node_state"] <- 3
    
    
    # check if active fires caused a spread / new fire
                                                # compare arrival time with fire_time_max, if arrival is later, spread fire to the neighborhoods
    
    fire_spreaded <- fire_dynamic_df[fire_dynamic_df$node_state == 1 & next_arrival_time >= fire_dynamic_df$fire_time_max,] 
    for (i in 1:nrow(fire_spreaded)){
      
      spreading_fire <- fire_spreaded[i,]
      new_fire_id_list <- unique(unlist(lapply(spreading_fire$node_id, function(x){param$neighborhood_list[[x]]})))
      new_fire_df <- fire_dynamic_df %>% filter(node_state == 0 & node_id %in% new_fire_id_list)
      
      if(nrow(new_fire_df) > 0){
        new_fire_df$fire_time_start <- new_fire_df$fire_time_start + spreading_fire$fire_time_max
        new_fire_df$fire_time_max <- new_fire_df$fire_time_max  + spreading_fire$fire_time_max
        new_fire_df$fire_time_end <- new_fire_df$fire_time_end   + spreading_fire$fire_time_max
        
        new_fire_df[next_arrival_time <= new_fire_df$fire_time_max, "fire_size_now"] <- 0 + (new_fire_df$node_degradation_rate * (next_arrival_time - new_fire_df$fire_time_start))
        new_fire_df[next_arrival_time > new_fire_df$fire_time_max, "fire_size_now"] <- 0 + (new_fire_df$node_degradation_rate * (next_arrival_time - new_fire_df$fire_time_start))
        new_fire_df$node_state <- 1
        
        
        # if (next_arrival_time <= new_fire_df$fire_time_max){
        #   new_fire_df$fire_size_now  <- 0 + (new_fire_df$node_degradation_rate * (next_arrival_time - new_fire_df$fire_time_start))
        # } else {
        #   new_fire_df$fire_size_now  <- 1 - (new_fire_df$node_amelioration_rate * (next_arrival_time - new_fire_df$fire_time_max))
        # }
      }
      
      fire_dynamic_df[fire_dynamic_df$node_state == 0 & fire_dynamic_df$node_id %in% new_fire_id_list,] <- new_fire_df 
      
      
    }
    
    
    
    # if (nrow(fire_spreaded) > 0) {
    #   new_fire_id_list <- unique(unlist(lapply(fire_spreaded$node_id, function(x){param$neighborhood_list[[x]]})))
    #   
    #   new_fire_df <- fire_dynamic_df %>% filter(node_state == 0 & node_id %in% new_fire_id_list)
    #   
    #   if(nrow(new_fire_df) > 0){
    #     new_fire_df$fire_time_start <- new_fire_df$fire_time_start + next_arrival_time
    #     new_fire_df$fire_time_max <- new_fire_df$fire_time_max  + next_arrival_time
    #     new_fire_df$fire_time_end <- new_fire_df$fire_time_end   + next_arrival_time
    #     new_fire_df$node_state <- 1
    #   }
    #   
    #   fire_dynamic_df[fire_dynamic_df$node_state == 0 & fire_dynamic_df$node_id %in% new_fire_id_list,] <- new_fire_df 
    #   
    #   
    #   }
   
    # process the job and change node state of next_arrival_location to rescued (state 2)
    if (next_arrival_location != param$base_node_id){
      fire_dynamic_df[fire_dynamic_df$node_id == next_arrival_location, 'node_state'] = 2
    }

    
    # update fire_dynamic_df globally
    fire_dynamic_df <<- fire_dynamic_df
    
    plot_counter <<- plot_counter+1
    ploting$plot_live_map(fire_dynamic_df, plot_counter)
    
    
    return(c(vehicle_df$vehicle_id, vehicle_df$arrival_location, next_arrival_location, vehicle_df$arrival_time,
           next_arrival_time, vehicle_df$next_status, next_status))
      
    } else{
      
      
      return(c(vehicle_df$vehicle_id, vehicle_df$arrival_location, vehicle_df$arrival_location, vehicle_df$arrival_time,
               vehicle_df$arrival_time, vehicle_df$next_status, 0))
    }  
    
    
    
    }))
  colnames(vehicle_next_df) = c('vehicle_id', 'departure_location', 'arrival_location', 'departure_time', 'arrival_time', 'current_status', 'next_status')
  decision_space_df <- rbind(decision_space_df, vehicle_next_df )
  all_vehicle_df <- as.data.frame(vehicle_next_df)
  
  if (sum(fire_dynamic_df$node_state == 1) == 0) {
    there_are_jobs_to_process = FALSE
  }
}

  
  
obj_total_value <- fire_dynamic_df %>% filter(node_state %in% c(0, 2) & value_now > 0)  %>% select (value_now) %>% sum()
  
obj_total_time <- decision_space_df %>% select(arrival_time) %>% max()
  
vehicle_1_route <- decision_space_df %>% filter(vehicle_id == 1) %>% select(arrival_location)
vehicle_2_route <- decision_space_df %>% filter(vehicle_id == 2) %>% select(arrival_location)

  
  
  
  
  
  
  
  for (i in 1:n_free_vehicles){
    
    
    
    
  }
  
  # find next jobs and their attributes
  new_jobs <- jobs_in_order_df %>% slice(1:n_free_vehicles) 
  
  new_jobs$arrival_time <-  new_jobs$clock_time + dist_matrix[param$base_node_id, new_jobs$node_id]
  new_jobs$return_time <-   new_jobs$arrival_time + dist_matrix[new_jobs$node_id, param$base_node_id]
  
  
  # update fire status when there is an arrival
  new_jobs <- new_jobs %>% arrange(arrival_time)
  
  upcoming_events <- rbind(new_jobs$arrival_time, new_jobs$return_time)
  
  a <- new_jobs %>% select(node_id, arrival_time) %>% mutate(evet_type='arrival', time)
  b <- new_jobs %>% select(node_id, return_time)  %>% mutate(evet_type='return')
  
  rbind(a,b)
  
  
  apply(new_jobs, 1, function(x){
    # x<-new_jobs[1,]
    x <- (as.data.frame(t(x)))
    fire_dynamic_df$clock_time <- fire_dynamic_df$clock_time + x$arrival_time
    
    fire_dynamic_df 
    x$node_id
    
    
  })
  new_jobs %>% arrange(arrival_time)
  fire_dynamic_df
  
  
  
  
  
  
  vehicle_next_df <- t(apply(vehicle_df, 1, function(x){
    # df<-vehicle_df[1,]
    df <- (as.data.frame(t(x)))
    # if vehicle is in travel
    if (df$vehicle_status == 1) { # vehicle is active
      

      
      
      
      
      %>% slice_sample(n = 1)
      next_job <- next_task$node_id
      next_job_arrival_time <- df$vehicle_return_time + dist_matrix[param$base_node_id, next_job]
      next_return_time <- next_job_arrival_time + dist_matrix[next_job, param$base_node_id]
 
      
      fire_dynamic_df
      
      
      
      # update map
      next_task$fire_time_end <- vehicle_next_arrival_time
      next_task$fire_size_now  <- next_task$fire_size_now
      
      
      
      
      
      
      vehicle_next_location_index <- (df$vehicle_current_location_index  + 1)
      vehicle_next_loc <- param$vehicle_routes[df$vehicle_id, vehicle_next_location_index]
      vehicle_next_loc_time_allocated <- param$vehicle_time_allocations[df$vehicle_id, vehicle_next_location_index]
      vehicle_current_location_time_spent <- 0 # restart time counter
      
      # if vehicle is completing its operation and will return to the base
      if (vehicle_next_loc == param$base_node_id) {
        vehicle_next_arrival_time <- df$vehicle_current_time + dist_matrix[df$vehicle_previous_location, param$base_node_id]
        
        vehicle_next_status <- 2  # vehicle returning to the base. status = 2 --> vehicle is completing its route
        fire_next_size_before_water_drop <- 0
        fire_next_size_after_water_drop <- 0
        fire_next_updated_end_time <- 0
        return(c(df$vehicle_id,vehicle_next_location_index,df$vehicle_current_location, vehicle_next_loc, vehicle_next_arrival_time, df$vehicle_next_status, vehicle_next_status, fire_next_size_before_water_drop, fire_next_size_after_water_drop, fire_next_updated_end_time, vehicle_current_location_time_spent)) 
        
      } else if (filter(active_fire_dynamic_df, region_id == vehicle_next_loc)$fire_size_max <= 0) {
        
        df$vehicle_current_location_index <- vehicle_next_location_index
        df$current_location_time_spent <- 0
        skip_finished_fire <- as.vector(t(df[1,]))
        return(skip_finished_fire)
        
      }else{ # vehicle is arrived at a job and will start processing
        
        # calculate arrival time to the next job
        # the vehicle must first go to the water resource and then to the next job
        vehicle_next_arrival_time <- df$vehicle_current_time + dist_matrix[df$vehicle_previous_location, param$base_node_id] + dist_matrix[param$base_node_id, vehicle_next_loc]
        
        vehicle_next_status <- 1 # change status to "process"
        processed_job_df <- filter(active_fire_dynamic_df, region_id == vehicle_next_loc)
        if (vehicle_next_arrival_time <= processed_job_df$fire_time_max) { # if arrival is before break even point
          fire_next_size_before_water_drop <- processed_job_df$fire_size_max - (processed_job_df$fire_time_max - vehicle_next_arrival_time)*processed_job_df$fire_degradation_rate  
          fire_next_size_after_water_drop <- fire_next_size_before_water_drop
          
        }else{ # if arrival is after break even point
          fire_next_size_before_water_drop <- processed_job_df$fire_size_max - (vehicle_next_arrival_time - processed_job_df$fire_time_max)*processed_job_df$fire_amelioration_rate
          fire_next_size_after_water_drop <- fire_next_size_before_water_drop
        }
        
        fire_next_updated_end_time <- processed_job_df$fire_time_end
        
        
        return(c(df$vehicle_id,vehicle_next_location_index,df$vehicle_current_location, vehicle_next_loc, vehicle_next_arrival_time, df$vehicle_next_status, vehicle_next_status, fire_next_size_before_water_drop, fire_next_size_after_water_drop, fire_next_updated_end_time, vehicle_current_location_time_spent)) 
      }
      
      
    }else if (df$vehicle_next_status == 1) { # process
      
      # we need to calculate the process time
      processed_job_df <- filter(active_fire_dynamic_df, region_id == df$vehicle_current_location)
      
      # if the recent status is 0 (travel), we already include the travel time to and from the water resource during travel time calculation
      if (df$vehicle_current_status  == 0) {
        water_drop_time <- df$vehicle_current_time # the vehicle immediately drops the water
        time_spent_for_water_drop <- 0
      }else{ #otherwise, if the recent status is 1, means that the job is in process, and the vehicle is just dropped a water, now it has to first go a water resource 
        water_drop_time <- df$vehicle_current_time + dist_matrix[df$vehicle_current_location, param$base_node_id] + dist_matrix[param$base_node_id, df$vehicle_current_location]
        time_spent_for_water_drop <- dist_matrix[df$vehicle_current_location, param$base_node_id] + dist_matrix[param$base_node_id, df$vehicle_current_location]
      }
      
      df$vehicle_current_location_time_spent = df$vehicle_current_location_time_spent + time_spent_for_water_drop
      
      
      if (water_drop_time <= processed_job_df$fire_time_max) { # if water drop is before break even point
        fire_size_at_water_drop_time <- processed_job_df$fire_size_max - (processed_job_df$fire_time_max - water_drop_time)*processed_job_df$fire_degradation_rate
        
      }else{ # if arrival is after break even point
        fire_size_at_water_drop_time <- processed_job_df$fire_size_max - (water_drop_time - processed_job_df$fire_time_max)*processed_job_df$fire_amelioration_rate
      }
      
      # if vehicle capacity is enough to cover all remaining fire, then the fire is done. 
      if (fire_size_at_water_drop_time <= (param$vehicle_capacity/param$water_req_per_km_sq) ) {
        update_fire_size_after_water_drop <- 0
        updated_fire_size_max <- 0
        update_fire_time_end <-  water_drop_time
        
        
      }else{ # else update fire size max and end
        update_fire_size_after_water_drop <- fire_size_at_water_drop_time - (param$vehicle_capacity/param$water_req_per_km_sq)
        updated_fire_size_max <- processed_job_df$fire_size_max - (param$vehicle_capacity/param$water_req_per_km_sq)
        update_fire_time_end <-  processed_job_df$fire_time_max  + (updated_fire_size_max / processed_job_df$fire_amelioration_rate)
      }
      
      processed_job_df$fire_size_max <- updated_fire_size_max
      processed_job_df$fire_time_end <- update_fire_time_end
      
      # update dynamic fire map
      # active_fire_dynamic_df[active_fire_dynamic_df["region_id"] == processed_job_df$region_id,] <<- processed_job_df
      active_fire_dynamic_df[active_fire_dynamic_df["region_id"] == processed_job_df$region_id,] <- processed_job_df
      active_fire_dynamic_df <<- active_fire_dynamic_df
      
      if (processed_job_df$fire_size_max == 0 | df$vehicle_current_location_time_spent >= (param$vehicle_time_allocations[df$vehicle_id, df$vehicle_current_location_index])) { # fire is finished
        vehicle_next_status_updated <- 0 # will start travel
      } else { #still burning
        vehicle_next_status_updated <- 1 # will continue processing
        
      }
      
      vehicle_current_time_updated <- water_drop_time
      
      return(c(df$vehicle_id,df$vehicle_current_location_index,df$vehicle_current_location, df$vehicle_current_location, vehicle_current_time_updated, df$vehicle_next_status, vehicle_next_status_updated, fire_size_at_water_drop_time, update_fire_size_after_water_drop, update_fire_time_end, df$vehicle_current_location_time_spent)) 
      
      
    } else{ # if returned at the base
      df$vehicle_current_location_index <- vehicle_next_location_index
      # df$vehicle_current_location  <- vehicle_next_loc
      
      skip_finished_fire <- as.vector(t(df[1,]))
      
      return(skip_finished_fire)
      
    }
  }))
  
  colnames(vehicle_next_df) = c('vehicle_id', 'vehicle_current_location_index', 'vehicle_previous_location', 'vehicle_current_location', 'vehicle_current_time', 'vehicle_current_status', 'vehicle_next_status', 'fire_size_before_water_drop', 'fire_size_after_water_drop', 'fire_end_time_without_more_process', 'vehicle_current_location_time_spent')
  decision_space_df <- rbind(decision_space_df,vehicle_next_df)
  vehicle_df <- as.data.frame(vehicle_next_df)
  
  if (all(vehicle_df$vehicle_next_status == 2)) {
    all_vehicles_are_not_in_base <- 'FALSE'    
    
  }
  
}



plot_solution_animated(decision_space_df)



# calculate objectives
fires_finished <- decision_space_df %>% filter(fire_size_after_water_drop <= 0 & vehicle_current_location != base_node_id)
fires_finished <- fires_finished[!duplicated(fires_finished[,c('vehicle_current_location')]),]



objective_space_df <- as.data.frame(t(apply(fires_finished, 1, function(x){
  
  df=(as.data.frame(t(x)))
  
  
  certain_fire <- active_fire_df[active_fire_df$region_id == df$vehicle_current_location,] 
  value_collected <- certain_fire$region_value_start - (df$vehicle_current_time * certain_fire$value_degradation_rate)
  
  return(c(df$vehicle_current_location, df$vehicle_current_time, value_collected))
  
})))

colnames(objective_space_df) <- c("region_id", "fire_finish_time","value_collected")

total_time_objective <- max(decision_space_df$vehicle_current_time)
total_value_objective <- sum(objective_space_df$value_collected)



# https://docs.google.com/spreadsheets/d/1NdHlEww9vG1lI2b4UcQkyP9Q7H9RYmCvZpdic0ow2gY/edit#gid=0



