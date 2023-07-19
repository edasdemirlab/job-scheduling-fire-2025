plot_live_map <- function(fire_dynamic_df, plot_counter){
 
  
    p1 <- fire_dynamic_df %>%   ggplot(aes(x_coordinate, y_coordinate, label = node_id, fill = node_degradation_rate)) +
    scale_fill_gradient2(low = "white", high = "red", na.value = "#009cdc") +
    geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE)
    geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape = 4, stroke = 10) +
    geom_point(data = ~filter(.x, node_state == 2), color = "aquamarine2",shape = 3, stroke = 10) +
    geom_point(data = ~filter(.x, node_state == 3), color = "black",shape = 4, stroke = 10, alpha=0.3) +

    geom_text(alpha = 0.7) +
    theme(panel.background = element_rect(fill = "transparent")) +
    # coord_equal() +
    theme(legend.position = "bottom") +
    theme(axis.title = element_blank()) +
    theme()
    
    p1 <- fire_dynamic_df %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
      scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
      geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE) + 
      geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
      geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 10) + 
      geom_point(data = ~filter(.x, node_state == 2), color = "aquamarine2",shape = 3, stroke = 10) + 
      geom_point(data = ~filter(.x, node_state == 3), color = "black",shape = 4, stroke = 10, alpha=0.3) + 
      geom_text(alpha = 0.7, size = 8) +
      theme(panel.background = element_rect(fill = "transparent")) +
      # coord_equal() +
      theme(legend.position = "none") +
      theme(axis.title = element_blank()) +
      theme(axis.text = element_text(size = 14)) + 
      theme()
    
    
  
    
    
    # p2 <- fire_dynamic_df %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
    #   scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc") +
    #   geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE)
    #   geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape = 4, stroke = 10) + 
    #   geom_point(data = ~filter(.x, node_state == 2), color = "aquamarine2",shape = 3, stroke = 10) + 
    #   geom_point(data = ~filter(.x, node_state == 3), color = "black",shape = 4, stroke = 10, alpha=0.3) + 
    #   geom_text(alpha = 0.7) +
    #   theme(panel.background = element_rect(fill = "transparent")) +
    #   # coord_equal() +
    #   theme(legend.position = "bottom") + 
    #   theme(axis.title = element_blank()) +
    #   theme() 
    
    # plot_to_print <- p2 + p1 + plot_layout(guides = "collect") & theme(legend.position = "bottom", legend.title = element_text(size=8) )
    
    # ggsave(plot_to_print, filename=paste("plots/fire_map_",plot_counter,".png",sep=""))
    ggsave(p1, filename=paste("plots/fire_map_",plot_counter,".png",sep=""))
    
  
}

plot_region_maps <- function(region_map_df) {
  
  p1 <- region_map_df %>%   ggplot(aes(x_coordinate, y_coordinate, label = node_id, fill = node_degradation_rate)) +
    scale_fill_gradient2(low = "white", high = "red", na.value = "#009cdc", name="spread rate") +
    geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE)
    geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
    geom_text(alpha = 0.7, size = 6) +
    # geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 1.5) + 
    theme(panel.background = element_rect(fill = "transparent")) + 
    # coord_equal() +
    theme(legend.position = "right") +
    theme(axis.title = element_blank()) +
    theme(axis.text = element_text(size = 14)) + 
    theme()


  
  p2 <- region_map_df %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
    scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
    geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE) + 
    geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
    geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 10) + 
    geom_text(alpha = 0.7, size = 8) +
    theme(panel.background = element_rect(fill = "transparent")) +
    # coord_equal() +
    theme(legend.position = "right") +
    theme(axis.title = element_blank()) +
    theme(axis.text = element_text(size = 14)) + 
    theme()

  
  p3 <- region_map_df %>%   ggplot(aes(x_coordinate, y_coordinate, label = node_id, fill = node_degradation_rate)) +
    scale_fill_gradient2(low = "white", high = "red", na.value = "#009cdc", name="spread rate") +
    geom_raster(aes(fill = node_degradation_rate), interpolate = TRUE) +
    geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.7) + 
    geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 2) + 
    theme(panel.background = element_rect(fill = "transparent")) +
    theme(axis.title = element_blank()) +
    theme(axis.text = element_text(size = 14)) + 
    theme() 
  
  p4 <- region_map_df %>%   ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start)) +
    scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
    geom_raster(aes(fill = node_value_at_start), interpolate = TRUE) + 
    geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.7) + 
    # geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 2) + 
    theme(panel.background = element_rect(fill = "transparent")) +
    theme(axis.title = element_blank()) +
    theme(axis.text = element_text(size = 14)) + 
    theme() 
  
  p4 + p3 + p2 + p1 + plot_layout(guides = "collect") & theme(legend.position = "bottom", legend.title = element_text(size=8) )
  
  # ggarrange(p4, p3, p2, p1, common.legend = TRUE, legend="bottom")
  # node.arrange(p4, p3, p2, p1, nrow = 2)
}
library(patchwork)


# 
# plot_active_fire_plots <- function(active_fire_map_df) {
# 
# active_fire_graphs <- as.data.frame(do.call("rbind", lapply(active_fire_map_df$region_id, function(x){
#   temp_df <- active_fire_map_df[active_fire_map_df$region_id==x,] 
#   temp1 <- temp_df %>% select(fire_time_start:fire_time_end)
#   temp2 <- temp_df %>% select(fire_size_start:fire_size_end)
#   temp3 <- temp_df %>% select(c(region_value_start, region_value_max, region_value_end))
#   temp_df <- cbind(paste0("region_",x),t(temp1), t(temp2), t(temp3))
#   rownames(temp_df) <- NULL                 # Reset row names
#   colnames(temp_df) <- c("region_id", "time", "fire_size", "region_value")
#   return(temp_df)
# })))
# 
# active_fire_graphs$time <- as.numeric(active_fire_graphs$time)
# active_fire_graphs$fire_size <- as.numeric(active_fire_graphs$fire_size)
# active_fire_graphs$region_value <- as.numeric(active_fire_graphs$region_value)
# 
# a1 <- active_fire_graphs %>% ggplot(aes(time, fire_size)) + 
#   geom_line(aes(colour = region_id, group = region_id ), lwd = 1) +
#   scale_x_continuous(expand = c(0,0), breaks = round(seq(min(active_fire_graphs$time), max(active_fire_graphs$time), by = 2),1)) +
#   facet_node(region_id~., ) +  scale_y_continuous(expand = c(0,0)) + 
#   theme_bw()  +  theme(legend.position = "none") 
# 
# 
# a2 <- active_fire_graphs %>% ggplot(aes(time, region_value)) + 
#   geom_line(aes(colour = region_id, group = region_id ), lwd = 1) +
#   scale_x_continuous(expand = c(0,0), breaks = round(seq(min(active_fire_graphs$time), max(active_fire_graphs$time), by = 2),1)) +
#   facet_node(region_id~., ) +  scale_y_continuous(expand = c(0,0)) + 
#   theme_bw() +   theme(legend.position = "bottom") 
# 
# ggarrange(a1, a2, ncol = 2, common.legend = TRUE, legend = "bottom")
# }


plot_solution_animated <- function(param, decision_space_df){
  get_coordinates <- apply(decision_space_df, 1, function(x){
    df <- (as.data.frame(t(x)))
    current_loc <- df$vehicle_current_location
    x_y_coordinates <- region_map_df[region_map_df['region_id' ] == current_loc,c('x_coordinate', 'y_coordinate', "region_id")]
    x_y_coordinates
  })
  get_coordinates <- do.call('rbind', get_coordinates)
  rownames(get_coordinates) <- 1:nrow(get_coordinates)    # Assign sequence to row names
  
  decision_space_animation_df <- cbind(decision_space_df, get_coordinates )
  
  
  decision_space_animation_df <- apply(decision_space_animation_df, 1, function(x){
    df <- (as.data.frame(t(x)))
    if (df$vehicle_current_status == 1) {
      time_to_go_water <- dist_matrix[df$vehicle_current_location, param$base_node_id]
      df_2 <- df
      df_2$vehicle_current_time <- df_2$vehicle_current_time + time_to_go_water
      df_2$x_coordinate <- 7.5
      df_2$y_coordinate <- 7.5
      return(rbind(df,df_2))
      
    }else{
      return(df)
    }
  })
  decision_space_animation_df <- do.call("rbind", decision_space_animation_df)
  
  
  dene2 <- decision_space_animation_df %>% ggplot(aes(x_coordinate, y_coordinate , label = region_id)) + #, fill = fire_size_after_water_drop
    geom_raster(aes(fill = 'red')) + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE)
    geom_text() +
    geom_point(aes(frame = vehicle_current_time, ids = vehicle_id, colour = factor(vehicle_id), shape = factor(vehicle_id)), size = 4) + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE)
    scale_color_manual(values = c("1" = "darkgreen", "2" = "blue")) +
    theme(axis.title = element_blank()) +
    theme(legend.position = "none")
  
  fig <- ggplotly(dene2)
  
  fig <- fig %>%
    animation_slider(
      currentvalue = list(prefix = "time ", font = list(color = "black"))
    )
  
  fig
  
}









unlist(lapply(region_map_df$node_value_at_start, function(x){bquote(p[j] == .(x))}))

(bquote(p[j] == .(1)))


# figures for paper

# value grid map with initial fires
p1 <- region_map_df %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
  scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
  geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE) + 
  geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
  geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 10) + 
  geom_text(alpha = 0.7, size = 8) +
  theme(panel.background = element_rect(fill = "transparent")) +
  # coord_equal() +
  theme(legend.position = "none") +
  theme(axis.title = element_blank()) +
  theme(axis.text = element_text(size = 14)) + 
  theme()

p1 +   
  geom_text(label=paste0("p=",region_map_df$node_value_at_start), nudge_y=-0.2,
                 check_overlap=T, size =5, col="white")  +





# plot scenario 1 fire spread
# node states --> 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fire proof, 5: water, 6:home/base
# state          1   2  3  4  5  6  7  8  9   
node_state_1 = c(6 , 4, 1, 5, 0, 0, 1, 0, 1)
node_state_2 = c(6 , 4, 1, 5, 0, 1, 1, 1, 1)
node_state_3 = c(6 , 4, 2, 5, 1, 1, 2, 1, 1)
node_state_4 = c(6 , 4, 2, 5, 2, 1, 2, 1, 1)
node_state_5 = c(6 , 4, 2, 5, 2, 3, 2, 1, 1)
node_state_6 = c(6 , 4, 2, 5, 2, 3, 2, 2, 1)
node_state_7 = c(6 , 4, 2, 5, 2, 3, 2, 2, 2)

node_state_list = list(node_state_1, node_state_2, node_state_3, node_state_4, node_state_5, node_state_6, node_state_7)

fire_dynamic_df_updated <- fire_dynamic_df
plot_counter <- 1
for (i_state in node_state_list) {
  fire_dynamic_df_updated$node_state <- i_state
  
  p1 <- fire_dynamic_df_updated %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
    scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
    geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE) + 
    geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
    geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 10) + 
    geom_point(data = ~filter(.x, node_state == 2), color = "aquamarine2",shape = 3, stroke = 10) + 
    geom_point(data = ~filter(.x, node_state == 3), color = "black",shape = 4, stroke = 10, alpha=0.3) + 
    geom_text(alpha = 0.7, size = 8) +
    theme(panel.background = element_rect(fill = "transparent")) +
    # coord_equal() +
    theme(legend.position = "none") +
    theme(axis.title = element_blank()) +
    theme(axis.text = element_text(size = 14)) + 
    theme()
  
  ggsave(p1, filename=paste("plots/fire_map_",plot_counter,".png",sep=""))
  plot_counter <- plot_counter + 1
}


fire_dynamic_df[6, "node_state"] = 1
fire_dynamic_df[8, "node_state"] = 1


# 2
p1 <- fire_dynamic_df %>%  ggplot(aes(x_coordinate, y_coordinate , label = node_id, fill = node_value_at_start )) +
  scale_fill_gradient2(low = "white", high = "darkgreen", na.value = "#009cdc", name="initial value") +
  geom_tile(colour = 'black') + #  geom_raster(aes(fill = fire_size_start), interpolate = TRUE) + 
  geom_point(data = ~filter(.x, node_state == 6), color = "black",shape = 2, stroke = 10, alpha=0.5) + 
  geom_point(data = ~filter(.x, node_state == 1), color = "yellow2",shape=4, stroke = 10) + 
  geom_point(data = ~filter(.x, node_state == 2), color = "aquamarine2",shape = 3, stroke = 10) + 
  geom_point(data = ~filter(.x, node_state == 3), color = "black",shape = 4, stroke = 10, alpha=0.3) + 
  geom_text(alpha = 0.7, size = 8) +
  theme(panel.background = element_rect(fill = "transparent")) +
  # coord_equal() +
  theme(legend.position = "none") +
  theme(axis.title = element_blank()) +
  theme(axis.text = element_text(size = 14)) + 
  theme()

ggsave(p1, filename=paste("plots/fire_map_",plot_counter,".png",sep=""))

