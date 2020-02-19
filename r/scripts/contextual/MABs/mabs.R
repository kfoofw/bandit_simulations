# https://nth-iteration-labs.github.io/contextual/articles/mabs.html
library(contextual)
library(here)

set.seed(1)
prob_per_arm       <- c(0.9, 0.1, 0.1)
horizon            <- 100
simulations        <- 1000

bandit             <- BasicBernoulliBandit$new(prob_per_arm)

agents             <- list(Agent$new(OraclePolicy$new(), bandit),
                           Agent$new(EpsilonGreedyPolicy$new(0.1), bandit),
                           Agent$new(ThompsonSamplingPolicy$new(1.0, 1.0), bandit),
                           Agent$new(Exp3Policy$new(0.1), bandit),
                           Agent$new(GittinsBrezziLaiPolicy$new(), bandit),
                           Agent$new(UCB1Policy$new(), bandit),
                           Agent$new(UCB2Policy$new(0.1), bandit))

simulation         <- Simulator$new(agents, horizon, simulations)
history            <- simulation$run()

saveRDS(history, paste0(filePath,"history_mabs.rds"))
history <- readRDS(paste0(filePath,"history_mabs.rds"))

plot(history, type = "cumulative")
