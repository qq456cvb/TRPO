import tensorflow as tf
import numpy as np
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import get_current_tower_context


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


class ConjGradient(optimizer.Optimizer):

    def __init__(self, actions, learning_rate=0.001, use_locking=False, name="ConjGradient"):
        super(ConjGradient, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._g = None
        self._actions = actions

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        pass
        # self._g = tf.gradients(tf.lo)
        # Create slots for the first and second moments.
        # for v in var_list:
        #     self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        if 'b' in var.name:
            loga = tf.log(self._actions)
            g = tf.zeros([0, var.shape[0], var.shape[0]], dtype=tf.float32)

            i = tf.constant(0)
            c = lambda i, g: i < tf.shape(self._actions)[0]
            b = lambda i, g: [i + 1, tf.concat([g, tf.expand_dims(tf.tensordot(tf.gradients(loga[i], var)[0], tf.gradients(loga[i], var)[0], 0) * self._actions[i], 0)], axis=0)]
            _, g = tf.while_loop(c, b, loop_vars=[i, g], shape_invariants=[i.get_shape(), tf.TensorShape([None, var.shape[0], var.shape[0]])])

            A = tf.reduce_sum(g, 0)
            # A = tf.Print(A, [tf.tensordot(tf.gradients(loga[0], var)[0], tf.gradients(loga[0], var)[0], 0)], first_n=1000)

            with tf.control_dependencies([tf.assert_equal(tf.transpose(A), A)]):
                # conjugate gradient method
                grad = tf.expand_dims(grad, -1)
                r = grad
                p = r
                x = tf.zeros_like(grad)
                c = lambda i, x, r, p: i < tf.shape(self._actions)[0]

                def cg(i, x, r, p):
                    Ap = tf.matmul(A, p)
                    rr = tf.reduce_sum(r * r)
                    alpha = rr / tf.reduce_sum(p * Ap)
                    x = x + alpha * p
                    r_old = r
                    r = r_old - alpha * Ap
                    # r = tf.Print(r, [tf.reduce_sum(r * r_old)])
                    # due to numerical error, the following check does not pass
                    # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(r * r_old), 0.)]):
                    beta = tf.reduce_sum(r * r) / rr
                    p_old = p
                    p = r + beta * p_old
                    # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(p * p_old), 0.)]):
                    #     p = tf.identity(p)
                    return i + 1, x, r, p

                b = cg
                _, x, _, _ = tf.while_loop(c, b, loop_vars=[i, x, r, p])
                x = tf.squeeze(x, -1)
                # x = tf.Print(x, [x], first_n=100)
                grad = x
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        var_update = state_ops.assign_add(var, lr_t * grad)
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(var_update)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder_with_default(tf.zeros([1, 5]), [1, 5], name='x')]

    def build_graph(self, x):
        action = FullyConnected('fc', x, 5, activation=tf.nn.softmax)
        self.action = tf.squeeze(action, 0, name='action')
        loss = tf.reduce_sum(self.action * tf.Variable(np.expand_dims(np.arange(5), 0), trainable=False, dtype=tf.float32))
        add_moving_summary(self.action[-1])
        return loss

    def optimizer(self):
        return ConjGradient(self.action, 0.1)


if __name__ == '__main__':
    trainer = SimpleTrainer()
    config = TrainConfig(
        model=Model(),
        dataflow=FakeData(shapes=[[1, 5]]),
        steps_per_epoch=50,
        max_epoch=1
    )
    launch_train_with_config(config, trainer)

