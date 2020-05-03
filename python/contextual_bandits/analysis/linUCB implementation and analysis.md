# Contextual Bandit: Linear Upper Confidence Bound Disjoint (LinUCB Disjoint) Algorithm

Contextual bandits (CB) are more granular in terms of the way they use information. Compared to their Multi-armed Bandits (MAB), we utilise contextual information about the observed instance in order to recommend the most effective variant.

Let's use an example of A/B testing. Assume a website has 3 webpage variants (A, B, C), and the objective is to optimise the click through rate (CTR):
- With a MAB approach, the variants are treated similarly, and the objective is to find the webpage that has the highest CTR for all user traffic
- With a CB approach, the variants are treated differently in terms of their effectiveness for specific users. Some users may differ in terms of their age or demographics which creates some form of profile. Such contextual information may be useful for us to recommend the appropriate variant as different variants may be more effective for certain types of users.

In short, the difference between MABs and CBs can be illustrated by the philosophies: 
- MABs are created with the "one size fits all" in mind.
- CBs are created with the concept of having bespoke solutions for different circumstances.

# LinUCB

As shown in my other analysis, UCB is based on 



The Upper Confidence Bound (UCB) algorithm is often phrased as "optimism in the face of uncertainty". To understand why, consider at a given round that each arm's reward function can be perceived as a point estimate based on the average rate of reward as observed. Drawing intuition from confidence intervals, for each point estimate, we can also incorporate some form of uncertainty boundary around the point estimate. In that sense, we have both lower boundary and upper boundary for each arm. 

The UCB algorithm is aptly named because we are only concerned with the upper bound, given that we are trying to find the arm with the highest reward rate.

There are different variants of the UCB algorithms but in this article, we will take a look at the UCB1 algorithm. At each given round of `n` trials, the reward UCB of all arms are represented by the following:

<p align="center">
    <img src="../img/ucb_eqn.png" />
</p>