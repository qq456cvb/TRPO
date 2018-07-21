# TRPO in TensorFlow
A Tensorflow implementation of "Trust Region Proximal Optimization" method. 

See the paper http://arxiv.org/abs/1502.05477

Currently working with discrete actions, continous(gaussian) variables support is straight forward.

# Features
Purely build on Tensorflow graphs and encapsulated as a seperate optimizer

You only need to pass the policy function and the cost function to the optimizer and create the cache variables.
