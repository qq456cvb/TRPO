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
    def __init__(self, policy, cost, delta=0.01, cg_iter=10, ls_max_iter=15, back_trace_ratio=0.8, use_locking=False, name="ConjGradient"):
        super(ConjugateGradientOptimizer, self).__init__(use_locking, name)
        self._cg_iter = cg_iter
        self._delta = delta
        self._ls_max_iter = ls_max_iter
        self._back_trace_ratio = back_trace_ratio
        # self._actions = actions
        self._policy = policy
        self._cost_before = cost
        self._mean_KL = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(self._policy) * tf.log(tf.stop_gradient(self._policy) / (self._policy + 1e-8) + 1e-8), 1))
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._cg_iter_t = ops.convert_to_tensor(self._cg_iter, name='trpo_cg_iter')
        self._ls_max_iter_t = ops.convert_to_tensor(self._ls_max_iter, name='trpo_max_iter')
        self._delta_t = ops.convert_to_tensor(self._delta, name="trpo_delta")
        self._back_trace_ratio_t = ops.convert_to_tensor(self._back_trace_ratio, name='trpo_back_trace_ratio')

    def _create_slots(self, var_list):
        pass

    def cg(self, Hx_fn, g):
        # i_init = tf.constant(0)
        # # conjugate gradient method
        # r_init = g + 0
        # # r = tf.Print(r, [tf.norm(r)])
        # p_init = r_init + 0
        # x_init = tf.zeros_like(g)
        # c = lambda i, x, r, p: tf.logical_and(i < self._cg_iter_t, tf.norm(r) > 1e-5)
        #
        # def cb(i, x, r, p):
        #     # Ap = tf.matmul(A, p)
        #     Ap = Hx_fn(p)
        #     rr = tf.reduce_sum(r * r)
        #     alpha = rr / tf.reduce_sum(p * Ap)
        #     x = x + alpha * p
        #     r_old = r
        #     r = r_old - alpha * Ap
        #     # r = tf.Print(r, [tf.reduce_sum(r * r_old)])
        #     # due to numerical error, the following check does not pass
        #     # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(r * r_old), 0.)]):
        #     beta = tf.reduce_sum(r * r) / rr
        #     p_old = p
        #     p = r + beta * p_old
        #     # p = tf.Print(p, [tf.norm(r)])
        #     # p = tf.Print(p, [tf.reduce_sum(p * tf.matmul(hess, p))])
        #     # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(p * p_old), 0.)]):
        #     #     p = tf.identity(p)
        #     return i + 1, x, r, p
        #
        # return tf.while_loop(c, cb, loop_vars=[i_init, x_init, r_init, p_init], back_prop=False)[1]
        r = tf.stop_gradient(g)
        p = tf.stop_gradient(r)
        x = tf.zeros_like(g)
        for i in range(self._cg_iter):
            Ap = Hx_fn(p)
            rr = tf.reduce_sum(r * r)
            alpha = rr / tf.reduce_sum(p * Ap)
            x = tf.cond(tf.norm(r) > 1e-5, lambda: x + alpha * p, lambda: tf.identity(x))
            r_old = r
            r = r_old - alpha * Ap
            # r = tf.Print(r, [tf.reduce_sum(r * r_old)])
            # due to numerical error, the following check does not pass
            # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(r * r_old), 0.)]):
            beta = tf.reduce_sum(r * r) / rr
            p_old = p
            p = r + beta * p_old
        return x
        # p = tf.Print(p, [tf.norm(r)])
        # p = tf.Print(p, [tf.reduce_sum(p * tf.matmul(hess, p))])
        # return x

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                        "Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            # p = _get_processor(v)
            converted_grads_and_vars.append((g, v))

        # converted_grads_and_vars = tuple(converted_grads_and_vars)
        converted_grads_and_vars = tuple([(g, v) for g, v in converted_grads_and_vars if g is not None])
        var_list = [v for g, v in converted_grads_and_vars]
        cache_var_list = []
        for v in var_list:
            for c in self.cache_vars:
                if c.op.name == v.op.name + 'cache':
                    cache_var_list.append(c)
                    break
        assert len(var_list) == len(cache_var_list)
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([v.name for _, v in converted_grads_and_vars],))
        with ops.init_scope():
            self._create_slots(var_list)
        var_shapes = [v.shape for _, v in converted_grads_and_vars]
        slice_idx = np.concatenate([[0], np.cumsum([np.prod(vs) for vs in var_shapes])], 0)
        # print(var_shapes)
        # print(slice_idx)
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            grad_flatten = tf.concat([tf.reshape(grad, [-1]) for grad, _ in converted_grads_and_vars], 0)
            KL_grad  = tf.gradients(self._mean_KL, var_list)
            KL_grad_flatten = tf.concat([tf.reshape(g, [-1]) for g in KL_grad], 0)

            # calculate Hessian * x
            def Hx_fn(m):
                grads = tf.gradients(tf.reduce_sum(KL_grad_flatten * tf.stop_gradient(m)), var_list)
                return tf.concat([tf.reshape(g, [-1]) for g in grads], 0) + 1e-5
            x = self.cg(Hx_fn, grad_flatten)
            xHx = tf.reduce_sum(tf.transpose(x) * Hx_fn(x))
            beta = tf.sqrt(2 * self._delta_t / (xHx + 1e-8))

            def get_KL(policy):
                return tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(self._policy) * tf.log(
                    tf.stop_gradient(self._policy) / (policy + 1e-8) + 1e-8), 1))

            i = tf.constant(0)

            def c(i, beta):
                with tf.control_dependencies([control_flow_ops.group(
                        [state_ops.assign(var, cache_var_list[i] - beta * tf.reshape(x[slice_idx[i]:slice_idx[i + 1]], var_shapes[i])) for
                         i,  (_, var) in enumerate(grads_and_vars)])]):
                    kl = get_KL(self.policy_fn())
                    cost = self.cost_fn()
                    return tf.logical_and(i < self._ls_max_iter_t, tf.logical_or(kl > self._delta_t,
                               cost > self._cost_before))
            b = lambda i, beta: [i + 1, self._back_trace_ratio_t * beta]
            i, _ = tf.while_loop(c, b, loop_vars=[i, beta], back_prop=False)

            var_update = tf.cond(tf.logical_or(tf.equal(i, self._ls_max_iter_t), tf.logical_not(tf.reduce_any(tf.is_nan(x)))),
                                 lambda: self.cache2var,
                                 lambda: self.var2cache)

            if not context.executing_eagerly():
                if isinstance(var_update, ops.Tensor):
                    var_update = var_update.op
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if var_update not in train_op:
                    train_op.append(var_update)
            # with tf.control_dependencies([self._cost]):
            #     with tf.control_dependencies([var_update]):
            #         cost = self._cost + 0
            #         with tf.control_dependencies([cost]):
            #             cost = cost + 0
            return var_update

    # calculate mean KL
    # def _apply_dense(self, grad, var):
    #     # action_shape = tf.shape(self._actions)
    #     # stacked_actions = tf.reshape(self._actions, [-1])
    #     # grad = tf.Print(grad, [tf.norm(grad)])
    #
    #     # var_1d_shape = tf.reduce_prod(tf.shape(var))
    #     var_shape = tf.shape(var)
    #     var_1d_shape = tf.reduce_prod(tf.shape(var))
    #     # J = tf.zeros(tf.stack([0, tf.reduce_prod(tf.shape(var))]), dtype=tf.float32)
    #     #
    #     i = tf.constant(0)
    #     # grad = tf.Print(grad, [self._cost])
    #     # c = lambda i, g: i < tf.shape(stacked_actions)[-1]
    #     # b = lambda i, g: [i + 1, tf.concat([g, tf.expand_dims(tf.reshape(tf.gradients(stacked_actions[i], var)[0], [-1]), 0)], axis=0)]
    #     # _, J = tf.while_loop(c, b, loop_vars=[i, J], shape_invariants=[i.get_shape(), tf.TensorShape([None, None])], back_prop=False)
    #     #
    #     # # J: NA * V
    #     # with tf.control_dependencies([tf.assert_equal(tf.reduce_any(tf.is_nan(J)), False)]):
    #     #  J = tf.reshape(J, tf.stack([-1, action_shape[-1], tf.reduce_prod(tf.shape(var))]))
    #     # # A : N*V*V
    #     # # use the trick in the paper: A = JtMJ, where M is the fisher information matrix of multinomial distribution
    #     # A = tf.map_fn(lambda x: tf.matmul(tf.transpose(x[0]), tf.matmul(tf.diag(1. / (x[1] + 1e-5)), x[0])), (J, self._actions), dtype=tf.float32, back_prop=False)
    #     # with tf.control_dependencies([tf.assert_less(tf.reduce_sum(tf.transpose(A[0])-A[0]), 1e-3)]):
    #     #     A = tf.identity(A)
    #     # A = tf.reduce_mean(A, [0]) + 1e-5 * tf.eye(tf.reduce_prod(tf.shape(var)))
    #
    #     # Hx = lambda x: tf.reshape(tf.gradients(tf.reduce_sum(tf.reshape(tf.gradients(self._mean_KL, var), [-1]) * x), var), [-1])
    #     hess = tf.hessians(self._mean_KL, var)[0]
    #     hess = tf.reshape(hess, [var_1d_shape, var_1d_shape])
    #     with tf.control_dependencies([tf.assert_less(tf.reduce_sum(tf.transpose(hess[0]) - hess[0]), 1e-3)]):
    #         hess = tf.identity(hess)
    #
    #     x = self.cg(hess, tf.expand_dims(tf.reshape(grad, [-1]), -1))
    #     # x = tf.Print(x, [x], first_n=100)
    #     xAx = tf.reshape(tf.matmul(tf.transpose(x), tf.matmul(hess, x)), [])
    #     # with tf.control_dependencies([tf.assert_equal(tf.reduce_any(tf.is_nan(x)), False)]):
    #     with tf.control_dependencies(None):
    #         beta = tf.sqrt(2 * self._delta_t / (xAx + 1e-8))
    #     x = tf.reshape(x, var_shape)
    #     grad = x + 0
    #
    #     i = tf.constant(0)
    #     c = lambda i, beta: tf.logical_and(i < self._ls_max_iter_t, 0.5 * beta * beta * xAx > self._delta_t)
    #     b = lambda i, beta: [i + 1, self._back_trace_ratio_t * beta]
    #     _, beta = tf.while_loop(c, b, loop_vars=[i, beta], back_prop=False)
    #
    #     # check again
    #     kl = 0.5 * beta * beta * xAx
    #     # beta_t = math_ops.cast(beta, var.dtype.base_dtype)
    #
    #     var_update = tf.cond(tf.logical_and(kl < self._delta_t, tf.logical_not(tf.reduce_any(tf.is_nan(grad)))), lambda: state_ops.assign_add(var, beta * grad), lambda: state_ops.assign_add(var, tf.zeros_like(var)))
    #     # Create an op that groups multiple operations.
    #     # When this op finishes, all ops in input have finished
    #     return control_flow_ops.group(var_update)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    # def _finish(self, update_ops, name_scope):
    #     """Do what is needed to finish the update.
    #
    #     This is called with the `name_scope` using the "name" that
    #     users have chosen for the application of gradients.
    #
    #     Args:
    #       update_ops: List of `Operation` objects to update variables.  This list
    #         contains the values returned by the `_apply_dense()` and
    #         `_apply_sparse()` calls.
    #       name_scope: String.  Name to use for the returned operation.
    #
    #     Returns:
    #       The operation to apply updates.
    #     """
    #     with tf.control_dependencies([tf.Print(tf.zeros([1]), [self._cost])]):
    #         return control_flow_ops.group(*update_ops, name=name_scope)


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

#
# class MyTrainer(SimpleTrainer):
#
#     def run_step(self):
#         print('step')
#         with TowerContext('', is_training=True):
#             print(self.hooked_sess.run(self.tower_func()))
#             print(self.hooked_sess.run(self.tower_func()))
#         self.hooked_sess.run(self.train_op)
#         with TowerContext('', is_training=True):
#             print(self.hooked_sess.run(self.tower_func()))


if __name__ == '__main__':
    trainer = SimpleTrainer()
    config = TrainConfig(
        model=TestModel(),
        dataflow=FakeData(shapes=[[3, 5]]),
        steps_per_epoch=500,
        max_epoch=1
    )
    launch_train_with_config(config, trainer)

