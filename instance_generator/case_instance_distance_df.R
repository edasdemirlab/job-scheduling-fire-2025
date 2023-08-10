
base_node_id <- 43
water_node_id <- c(13, 20)
fire_proof_node_list <- c()
# vehicle_speed <- 60
coordinates <- read.csv('case_coordinates.csv')
dist_matrix <- as.matrix(dist(coordinates[,c('x_coordinate','y_coordinate')], method = "euclidean", diag = TRUE, upper = TRUE, p = 2))
# dist_matrix <- dist_matrix/vehicle_speed # in terms of travel time
 
dist_matrix_mip_model <- dist_matrix

dist_pairwise <- data.frame(from = colnames(dist_matrix_mip_model)[col(dist_matrix_mip_model)], to=rownames(dist_matrix_mip_model)[row(dist_matrix_mip_model)], dist = c(dist_matrix_mip_model)) %>% filter(if_any(from:to, ~ .x != to))
dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] == base_node_id & (x[2] %in% water_node_id)) | ((x[1] %in% water_node_id) & x[2] == base_node_id))) {return(as.numeric(x))}}))


dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] %in% water_node_id & (x[2] %in% water_node_id)) | ((x[1] %in% water_node_id) & x[2] %in% water_node_id))) {return(as.numeric(x))}}))

# dist_pairwise <- do.call("rbind", apply(dist_pairwise,1, function(x){if (!((x[1] %in% fire_proof_node_list | (x[2] %in% fire_proof_node_list)))) {return(as.numeric(x))}}))


write.table(dist_pairwise, "dist_pairwise_case.txt")