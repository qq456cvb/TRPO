# TRPO in TensorFlow
A Tensorflow implementation of "Trust Region Proximal Optimization" method. 

See the paper http://arxiv.org/abs/1502.05477

Currently working with discrete actions, continous(gaussian) variables support is straight forward.

Taking tf.gradients twice seems extremely slow.

# TODO
Reject gradients when loss drops to improve stability
