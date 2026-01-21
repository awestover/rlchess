Blue and green lines:
do RL on centipawn loss.

data: we have a function that generates random interesting board states. 
reward signal: the thing that computes centipawn loss. 

let's do GRPO.

You do a bunch of rollouts, grade them, backprop, and continue!
