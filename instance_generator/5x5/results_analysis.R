library(tidyverse )
library(readxl)
library(ggplot2)
library(gridExtra)


results <- read.csv("combination_results_5x5.csv")


label_names <- c(
  `1` = "number of initial fires: 1",
  `2` = "number of initial fires: 2",
  `3` = "number of initial fires: 3",
  `4` = "number of initial fires: 4",
  `5` = "number of initial fires: 5",
  `6` = "number of initial fires: 6",
  `7` = "number of initial fires: 7",
  `8` = "number of initial fires: 8",
  `9` = "number of initial fires: 9",
  `10` = "number of initial fires: 10",
  `11` = "number of initial fires: 11",
  `12` = "number of initial fires: 12",
  `13` = "number of initial fires: 13",
  `14` = "number of initial fires: 14",
  `15` = "number of initial fires: 15",
  `16` = "number of initial fires: 16",
  `17` = "number of initial fires: 17",
  `18` = "number of initial fires: 18"
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
  scale_y_continuous(name="solution time (in seconds)", limits=c(0, 3700), breaks=c(0, 1000, 2000, 3000, 3600)) +
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




p4

my_vec = c(1:17)

my_combi1 <- unlist(lapply(1:length(my_vec),    # Get all combinations
                           combinat::combn, 
                           x = my_vec,
                           simplify = FALSE), 
                    recursive = FALSE)
length(my_combi1)