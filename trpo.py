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


class ConjugateGradientOptimizer(optimizer.Optimizer):
    # cg_iter: conjugate gradient iteration
    # ls_max_iter: line search max iteration
    # back_trace_ratio: backward line search ratio per iteration
    def __init__(self, actions, delta=0.01, cg_iter=10, ls_max_iter=15, back_trace_ratio=0.8, use_locking=False, name="ConjGradient"):
        super(ConjugateGradientOptimizer, self).__init__(use_locking, name)
        self._cg_iter = cg_iter
        self._delta = delta
        self._ls_max_iter = ls_max_iter
        self._back_trace_ratio = back_trace_ratio
        self._actions = actions

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._cg_iter_t = ops.convert_to_tensor(self._cg_iter, name='trpo_cg_iter')
        self._ls_max_iter_t = ops.convert_to_tensor(self._ls_max_iter, name='trpo_max_iter')
        self._delta_t = ops.convert_to_tensor(self._delta, name="trpo_delta")
        self._back_trace_ratio_t = ops.convert_to_tensor(self._back_trace_ratio, name='trpo_back_trace_ratio')

    def _create_slots(self, var_list):
        pass

    # calculate mean KL
    def _apply_dense(self, grad, var):
        action_shape = tf.shape(self._actions)
        stacked_actions = tf.reshape(self._actions, [-1])

        # var_1d_shape = tf.reduce_prod(tf.shape(var))
        grad_1d = tf.reshape(grad, [-1])
        var_shape = tf.shape(var)
        J = tf.zeros(tf.stack([0, tf.reduce_prod(tf.shape(var))]), dtype=tf.float32)

        i = tf.constant(0)
        c = lambda i, g: i < tf.shape(stacked_actions)[-1]
        b = lambda i, g: [i + 1, tf.concat([g, tf.expand_dims(tf.reshape(tf.gradients(stacked_actions[i], var)[0], [-1]), 0)], axis=0)]
        _, J = tf.while_loop(c, b, loop_vars=[i, J], shape_invariants=[i.get_shape(), tf.TensorShape([None, None])], back_prop=False)

        # J: NA * V
        J = tf.reshape(J, tf.stack([-1, action_shape[-1], tf.reduce_prod(tf.shape(var))]))
        # A : N*V*V
        # use the trick in the paper: A = JtMJ, where M is the fisher information matrix of multinomial distribution
        A = tf.map_fn(lambda x: tf.matmul(tf.transpose(x[0]), x[0] / tf.expand_dims(1. / x[1], -1)), (J, self._actions), dtype=tf.float32, back_prop=False)
        # with tf.control_dependencies([tf.assert_less(tf.transpose(A[0])-A[0], 1e-5)]):
        #     A = tf.identity(A)
        A = tf.reduce_mean(A, [0]) + 1e-5 * tf.eye(tf.reduce_prod(tf.shape(var)))

        with tf.control_dependencies(None):
            # conjugate gradient method
            gard = tf.expand_dims(grad_1d, -1)
            r = gard
            p = r
            x = tf.zeros_like(gard)
            c = lambda i, x, r, p: i < self._cg_iter_t

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
            # x = tf.Print(x, [x], first_n=100)
            xAx = tf.reshape(tf.matmul(tf.transpose(x), tf.matmul(A, x)), [])
            beta = tf.sqrt(2 * self._delta_t / xAx)
            x = tf.reshape(x, var_shape)
            grad = x

        i = tf.constant(0)
        c = lambda i, beta: tf.logical_and(i < self._ls_max_iter_t, 0.5 * beta * beta * xAx > self._delta_t)
        b = lambda i, beta: [i + 1, self._back_trace_ratio_t * beta]
        _, beta = tf.while_loop(c, b, loop_vars=[i, beta], back_prop=False)

        # check again
        kl = 0.5 * beta * beta * xAx
        # beta_t = math_ops.cast(beta, var.dtype.base_dtype)

        var_update = tf.cond(tf.logical_and(kl < self._delta_t, tf.logical_not(tf.reduce_any(tf.is_nan(grad)))), lambda: state_ops.assign_add(var, beta * grad), lambda: tf.identity(var))
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(var_update)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class TestModel(ModelDesc):
    def inputs(self):
        return [tf.placeholder_with_default(tf.ones([3, 5]), [3, 5], name='x')]

    def build_graph(self, x):
        action = FullyConnected('fc', x, 5, activation=tf.nn.softmax)
        self.action = tf.identity(action, name='action')
        loss = tf.reduce_sum(self.action * tf.Variable(np.expand_dims(np.arange(5), 0), trainable=False, dtype=tf.float32))
        add_moving_summary(self.action[-1, -1])
        return loss

    def optimizer(self):
        return ConjugateGradientOptimizer(self.action, 0.1)


if __name__ == '__main__':
    trainer = SimpleTrainer()
    config = TrainConfig(
        model=TestModel(),
        dataflow=FakeData(shapes=[[3, 5]]),
        steps_per_epoch=500,
        max_epoch=1
    )
    launch_train_with_config(config, trainer)

