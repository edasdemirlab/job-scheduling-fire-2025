library(dplyr)
library(ggplot2)
library(gridExtra)
results <- read.csv("combination_results_3x3.csv")


label_names <- c(
  `1` = "number of initial fires: 1",
  `2` = "number of initial fires: 2",
  `3` = "number of initial fires: 3",
  `4` = "number of initial fires: 4",
  `5` = "number of initial fires: 5",
  `6` = "number of initial fires: 6"
)

# box plot initial number of lines vs total value
p1 <- results %>% ggplot(aes(x = factor(number_of_jobs_arrived) , y = total_value, fill= factor(number_of_initial_fires))) +
    geom_boxplot() +
    theme_bw() + 
    xlab("number of job arrivals") + 
     ylab("value maintained (objective value)") +
    facet_wrap(.~factor(number_of_initial_fires), labeller = as_labeller(label_names)) +
    theme(legend.position = "none") 

p1


# box plot initial number of lines vs solution time
p2 <- results %>% ggplot(aes(x = factor(number_of_jobs_arrived) , y = python_time, fill= factor(number_of_initial_fires))) +
  geom_boxplot() +
  theme_bw() + 
  xlab("number of job arrivals") + 
  ylab("solution time (in seconds)") +
  facet_wrap(.~factor(number_of_initial_fires), labeller = as_labeller(label_names)) +
  theme(legend.position = "none") 

p2



# box plot initial number of lines vs operation time
p3 <- results %>% ggplot(aes(x = factor(number_of_jobs_arrived) , y = operation_time, fill= factor(number_of_initial_fires))) +
  geom_boxplot() +
  theme_bw() + 
  xlab("number of job arrivals") + 
  ylab("operation time (in seconds)") +
  facet_wrap(.~factor(number_of_initial_fires), labeller = as_labeller(label_names)) +
  theme(legend.position = "none") 

p3




