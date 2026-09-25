post training(source : Natasha Jaques lecture on Youtube, https://www.youtube.com/watch?v=_9FltLLOOVE)
  -problem with naive rl fine tuning
    -catastrophic forgetting
    -RL will traivially exploit the reward
    -limited reward data, or imperfect reward function.
      -constrain RL updates by KL divergence penalty.

KL control from pre trained data prior p(a|s) : 
  -L(q) = E[reward(gamma)/c - D_KL[q(gamma)||p(gamma)] : gamma is policy
  q : RL policy , p : pre trained prior 


follow up working : Online RL, "self play"(synthetic data).


How to learn the reward function:

  -p_hat[p1>p2] = exp(sum_r_hat_(output_1_t,answer_1_t))/ (exp_sum_r_hat_(o_1_t,a_1_t) + exp_sum_r_hat_(o_2_t,a_2_t))
    - learn r_hat by minimizing ce between predictions and human labels.

  
