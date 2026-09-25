post training(source : Natasha Jaques lecture on Youtube, https://www.youtube.com/watch?v=_9FltLLOOVE)
  -problem with naive rl fine tuning
    -catastrophic forgetting
    -RL will traivially exploit the reward
    -limited reward data, or imperfect reward function.
      -constrain RL updates by KL divergence penalty.

KL control from pre trained data prior p(a|s) : 
  -L(q) = E[reward(gamma)/c - D_KL[q(gamma)||p(gamma)] : gamma is policy
  q : RL policy , p : pre trained prior 
