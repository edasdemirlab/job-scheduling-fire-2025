
# problem parameters
base_id <- 1
water_id <- 4
node_at_a_side <- 3
pX <- c(0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5)
pY <- c(0.5,1.5,2.5,0.5,1.5,2.5,0.5,1.5,2.5)
pMat <- c()


# functions
# plot vehicle route
plot_vehicle_route <- function(visiting_order, vehicle_col, vehicle_line_type) {
  for (i in 1:(length(visiting_order)-1)) {
    print(i)
    arrows(pX[visiting_order[i]],pY[visiting_order[i]], pX[visiting_order[i+1]],pY[visiting_order[i+1]], col=vehicle_col,lwd=3, lty=vehicle_line_type, length = 0.25, angle = 20)
    i=i+1
  }
}

# # plot nodes
# plot_node_add <- function(node_x, node_y, node_id, t_s, t_v, col_def){
#   points(x=node_x,y=node_y,col = "black", bg="white", pch=21,cex=3.5)
#   text(x=node_x,y=node_y,col = "black", labels=node_id ,cex=1.3)
#   text(node_x,node_y+0.32,labels =  bquote(t[s] == .(t_s)), col = col_def)
#   text(node_x,node_y+0.2,labels =  bquote(t[v] == .(t_v)), col = col_def)
# }


# plot nodes
plot_node_add <- function(status, node_x, node_y, node_id, t_s, t_m, t_e, t_v, p_j, col_def){
  if (status == "visited") {
      col_to_use = "forestgreen"
    }
  else{
    col_to_use = "red"
  }
  points(x=node_x, y=node_y, col = "black", bg=alpha(col_to_use, 0.2),  pch = 21, cex = 5.5)
  text(x=node_x, y=node_y, col = "black", labels = node_id, cex = 1.5)
  text(node_x, node_y + 0.38,labels =  bquote(tw[.(i)] == "[" ~.(t_s) ~"," ~.(t_m) ~","~ .(t_e) ~"]"), col = col_to_use, cex = 0.8)
  text(node_x, node_y + 0.25,labels =  bquote(t[v] == .(t_v)), col = col_to_use, cex = 0.8)
  text(node_x, node_y - 0.28,labels =  bquote(p[.(i)] == .(p_j)), col = col_to_use, cex = 0.8)
  
}


# node arrival times
t_s_list <- c(0, 0, 0, 0, 2, 1, 0, 1, 0)
t_m_list <- c(0, 0, 2.5, 0, 3.67, 2.67, 1.25, 2, 1)
t_e_list <- c(0, 0, 5, 0, 7, 6, 6.25, 12, 11)


t_v_list <- c(0, 0, 2, 0, 4, "NA", 2, 6.41, 6.47)
p_j_list <- c(0, 0, 0.6, 0, 0.12, 0, 0.41, 0.17, 0.16)


# vehicle routes
vehicle1_route <- c(1,7,4,5,4,8,1)
vehicle2_route <- c(1,3,4,9,1,3)
vehicle_routes <- list(vehicle1_route, vehicle2_route )

plot(c(), xlim=c(0,ceiling(max(pX))), ylim=c(0,ceiling(max(pY))),asp=1,xaxs="i", yaxs="i",xaxt='n', yaxt="n", ylab="", xlab="")
# plot(c(), xlim=c(0,ceiling(max(pX))), ylim=c(0,ceiling(max(pY))),asp=1,xaxs="i", yaxs="i",xaxt='n', yaxt="n", ylab="", xlab="", bty="n") # see ?plot.function
x <- seq(0, ceiling(max(pX)), 1)
axis(side = 1, at = x, labels = T)
axis(side = 2, at = x, labels = T)
grid(nx = node_at_a_side, ny = node_at_a_side, col = "gray25", lty = "dotted",lwd = par("lwd"), equilogs = TRUE)


# add workplace
points(x=0.5,y=0.5,col = "black", pch=17,cex=3)
# text(0.5,0.5-0.2,labels = bquote(t[s] == 0), col=1)
text(0.5,0.5-0.2,labels = bquote(t[h] == 9.30), col=1)

# add routes
plot_vehicle_route(vehicle1_route, 4, 1)
plot_vehicle_route(vehicle2_route,"darkorange1", 5)

# add node results for the routes
for (v in vehicle_routes) {
  for (i in setdiff(unique(v), c(base_id, water_id))) {
    plot_node_add("visited", pX[i], pY[i], i, t_s_list[i], t_m_list[i], t_e_list[i], t_v_list[i], p_j_list[i], "gray20") # node 3
  }
}

# add nodes that are not visited
not_visited_nodes = c(6)

for (i in not_visited_nodes){
  plot_node_add("not_visited", pX[i],  pY[i], i, t_s_list[i], t_m_list[i], t_e_list[i], t_v_list[i], p_j_list[i], "gray20") # node 3
}

# add water resource
water_resouce_nodes = c(4)
for (i in water_resouce_nodes) {
  points(x=pX[water_id],y=pY[water_id],col = "black", bg=alpha("deepskyblue", 0.2), pch=21,cex=5.5)
  text(x=pX[water_id],y=pY[water_id],col = "black", labels=water_id ,cex=1.5)
}


legend_text = c()
for (v in (1:length(vehicle_routes))){
  legend_text <- append(legend_text, paste0("UAV ", v, ": ", paste(vehicle_routes[[v]], collapse = "-")))}
# c("UAV 1: 1-7-4-5-4-8-1", "UAV 2: 1-3-4-9-1-3")

legend('top', legend_text, pch = c(NA, NA), lty = c(1, 5), col = c(4,"darkorange2"), 
       lwd=c(2,2),text.col = c(4,"darkorange2"), cex = 1.2, 
       xpd=TRUE, bty = "n", inset=c(0,-0.15),
         ncol = 1)



t(combn(c(3,5,6,7,8,9), 3))
