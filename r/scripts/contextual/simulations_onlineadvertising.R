# Load and attach the contextual package.
library(contextual)
library(here)
set.seed(1)
filePath <- paste0(here(), "/scripts/contextual/")
# Define for how long the simulation will run.
horizon <- 400
# Define how many times to repeat the simulation.
simulations <- 10000
# Define the probability that each ad will be clicked.
click_probabilities <- c(0.8, 0.4, 0.2)
# Initialize a ContextualBernoulliBandit
bandit <- ContextualBernoulliBandit$new(weights = click_probabilities)
# Initialize an EpsilonGreedyPolicy with a 40% exploiration rate.
eg_policy <- EpsilonGreedyPolicy$new(epsilon = 0.4)
# Initialize an EpsilonFirstPolicy with a 100 step exploration period.
ef_policy <- EpsilonFirstPolicy$new(time_steps = 100)
# Initialize two Agents, binding each policy to a bandit.
ef_agent <- Agent$new(ef_policy, bandit)
eg_agent <- Agent$new(eg_policy, bandit)
# Assign both agents to a list.
agents <- list(ef_agent, eg_agent)
# Initialize Simulator with agent list, horizon, and nr of simulations.
simulator <- Simulator$new(agents, horizon, simulations, do_parallel = TRUE)
# Now run the simulator.
history <- simulator$run()
saveRDS(history, paste0(filePath,"history_simulations_onlineadvertising.rds"))
history <- readRDS(paste0(filePath,"history_simulations_onlineadvertising.rds"))
# Finally, plot the average reward per time step t
plot(history, type = "average", regret = FALSE, lwd = 2, 
     legend_position = "lowerleft")

# And the cumulative reward rate, which equals the Click Through Rate)
plot(history, type = "cumulative", regret = FALSE, rate = TRUE, lwd = 2)

### COntextual bandits
#                                  +-----+----+-----------> ads: k = 3
#                                  |     |    |
click_probs         <- matrix(c(  0.2,  0.3, 0.1,     # --> d1: old   (p=.5)
                                  0.6,  0.1, 0.1   ), # --> d2: young (p=.5)
                              #     features: d = 2
                              
                              nrow = 2, ncol = 3, byrow = TRUE)

# Initialize a ContextualBernoulliBandit with contextual weights
context_bandit      <- ContextualBernoulliBandit$new(weights = click_probs)
# Initialize LinUCBDisjointPolicy
lucb_policy         <- LinUCBDisjointPolicy$new(0.6)
# Initialize three Agents, binding each policy to a bandit.
ef_agent            <- Agent$new(ef_policy, context_bandit)
eg_agent            <- Agent$new(eg_policy, context_bandit)
lucb_agent          <- Agent$new(lucb_policy, context_bandit)
# Assign all agents to a list.
agents              <- list(ef_agent, eg_agent, lucb_agent)
# Initialize Simulator with agent list, horizon, and nr of simulations.
simulator           <- Simulator$new(agents, horizon, simulations, do_parallel = FALSE)
# Now run the simulator.
history_lucb             <- simulator$run()
saveRDS(history_lucb, paste0(filePath,"history_lucb_simulations_onlineadvertising.rds"))
history_lucb <-readRDS(paste0(filePath,"history_lucb_simulations_onlineadvertising.rds"))
# And plot the cumulative reward rate again.
plot(history_lucb, type = "cumulative", regret = FALSE, rate = TRUE, legend_position ="bottomright")

plot(history_lucb, type = "average", regret = FALSE, rate = TRUE, legend_position ="bottomright")

