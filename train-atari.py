#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu

import numpy as np
import sys
import os
import uuid
import argparse

import cv2
import tensorflow as tf
import six
from six.moves import queue


from tensorpack import *
from tensorpack.tfutils import optimizer
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient, FilterNoneGrad
from tensorpack.utils.gpu import get_num_gpu


import gym
from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorflow.python.ops import control_flow_ops, state_ops
from tensorpack.utils.argtools import call_only_once, memoized
from tensorpack.tfutils.tower import TowerFuncWrapper
import functools
import tensorflow.contrib.slim as slim

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

GAMMA = 0.99
STATE_SHAPE = (4,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 100
EVAL_EPISODE = 5
BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
SIMULATOR_PROC = 8
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None

import trpo


def get_player(train=False, dumpdir=None):
    env = gym.make(ENV_NAME)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir, video_callable=lambda _: True)
    # env = FireResetEnv(env)
    # env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    # env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    return env


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def inputs(self):
        assert NUM_ACTIONS is not None
        return [tf.placeholder(tf.float32, (None,) + STATE_SHAPE, 'state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'futurereward'),
                tf.placeholder(tf.float32, (None,), 'action_prob'),
                ]

    @auto_reuse_variable_scope
    def _get_NN_prediction(self, state, action, futurereward, action_prob):
        # image = tf.cast(image, tf.float32) / 255.0
        with argscope(FullyConnected, activation=tf.nn.relu):
            l = state
            l = FullyConnected('fc', l, 64)
            for i in range(5):
                l = FullyConnected('fc%d' % i, l, 64)

        # l = FullyConnected('fc0', l, 64)
        # l = PReLU('prelu', l)
        policy = tf.nn.softmax(FullyConnected('fc-pi', l, NUM_ACTIONS), name='policy')    # unnormalized policy
        return policy

    def build_graph(self, state, action, futurereward, action_prob):
        self.policy = self._get_NN_prediction(state, action, futurereward, action_prob)
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        pi_a_given_s = tf.reduce_sum(self.policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10)

        policy_loss = -tf.reduce_sum(futurereward * importance, name='policy_loss')
        cost = policy_loss
        self.cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')

        # summary.add_moving_summary(advantage, cost, tf.reduce_mean(importance, name='importance'))
        return self.cost

    def optimizer(self):
        # opt = tf.train.AdamOptimizer()
        opt = trpo.ConjugateGradientOptimizer(self.policy, self.cost, delta=0.01)
        gradprocs = [SummaryGradient()]
        opt_proc = optimizer.apply_grad_processors(opt, gradprocs)
        return opt_proc, opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        num_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy'],
            self._gpus[k % num_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            try:
                distrib = outputs.result()[0]
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client.memory.append(TransitionExperience(
                state, action, reward=None, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, client, state, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarded
        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            if isOver:
                # should clear client's memory and put to queue
                self._parse_memory(0, client, True)
            # else:
            #     if len(client.memory) == LOCAL_TIME_MAX + 1:
            #         R = client.memory[-1].value
            #         self._parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)

    def _parse_memory(self, init_r, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = k.reward + GAMMA * R
            self.queue.put([k.state, k.action, R, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


class MyTrainer(SimpleTrainer):
    """
    Single-GPU single-cost single-tower trainer.
    """
    def __init__(self):
        super(MyTrainer, self).__init__()

    def setup_graph2(self, inputs_desc, input, get_cost_fn, get_policy_fn, get_opt_fn):
        get_cost_fn = TowerFuncWrapper(get_cost_fn, inputs_desc)
        get_policy_fn = TowerFuncWrapper(get_policy_fn, inputs_desc)
        get_opt_fn = memoized(get_opt_fn)
        self.tower_func = get_cost_fn

        # TODO setup may want to register monitor as well??
        input_callbacks = self._setup_input(inputs_desc, input)
        train_callbacks = self._setup_graph2(input, get_cost_fn, get_policy_fn, get_opt_fn)
        self.register_callback(input_callbacks + train_callbacks)

    def _make_get_grad_fn(self, input, get_cost_fn, get_opt_fn):
        """
        Returns:
            a get_grad_fn for GraphBuilder to use.
        """
        # internal use only
        assert input.setup_done()

        def get_grad_fn():
            ctx = get_current_tower_context()
            cost = get_cost_fn(*input.get_input_tensors())
            if not ctx.is_training:
                return None     # this is the tower function, could be called for inference

            if ctx.has_own_variables:
                varlist = ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                varlist = tf.trainable_variables()
            opt = get_opt_fn()[0]
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=self.GATE_GRADIENTS,
                colocate_gradients_with_ops=self.COLOCATE_GRADIENTS_WITH_OPS,
                aggregation_method=self.AGGREGATION_METHOD)
            grads = FilterNoneGrad().process(grads)
            return grads

        return get_grad_fn

    def _setup_graph2(self, input, get_cost_fn, get_policy_fn, get_opt_fn):
        logger.info("Building graph for a single training tower ...")
        with TowerContext('', is_training=True):
            grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
        opt_proc, self.opt = get_opt_fn()
        self.opt.cost_fn = functools.partial(get_cost_fn, *input.get_input_tensors())
        self.opt.policy_fn = functools.partial(get_policy_fn, *input.get_input_tensors())
        self.opt.cache_vars = [tf.Variable(v.initialized_value(), name=v.op.name + 'cache', trainable=False) for v in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        self.opt.var2cache = control_flow_ops.group([state_ops.assign(c, v) for c, v in zip(self.opt.cache_vars,
                                                                                        tf.get_collection(
                                                                                            tf.GraphKeys.TRAINABLE_VARIABLES))])
        self.opt.cache2var = control_flow_ops.group([state_ops.assign(v, c) for c, v in zip(self.opt.cache_vars,
                                                                                        tf.get_collection(
                                                                                            tf.GraphKeys.TRAINABLE_VARIABLES))])
        with TowerContext('', is_training=True):
            self.train_op = opt_proc.apply_gradients(grads, name='min_op')

        return []


def launch_train_with_config2(config, trainer):
    assert isinstance(trainer, SingleCostTrainer), trainer
    assert isinstance(config, TrainConfig), config
    assert config.model is not None
    assert config.dataflow is not None or config.data is not None

    model = config.model
    inputs_desc = model.get_inputs_desc()
    input = config.data or config.dataflow
    input = apply_default_prefetch(input, trainer)

    trainer.setup_graph2(
        inputs_desc, input,
        model._build_graph_get_cost, lambda *inputs: model._get_NN_prediction(*inputs), model.get_optimizer)
    trainer.train_with_defaults(
        callbacks=config.callbacks,
        monitors=config.monitors,
        session_creator=config.session_creator,
        session_init=config.session_init,
        steps_per_epoch=config.steps_per_epoch,
        starting_epoch=config.starting_epoch,
        max_epoch=config.max_epoch,
        extra_callbacks=config.extra_callbacks)


def train():
    dirname = os.path.join('train_log', 'train-atari-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    num_gpu = get_num_gpu()
    global PREDICTOR_THREAD
    if num_gpu > 0:
        if num_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(num_gpu))[-num_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    config = AutoResumeTrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=1),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=get_model_loader(args.load) if args.load else None,
        max_epoch=1000,
    )
    trainer = MyTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config2(config, trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', default='CartPole-v0')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'dump_video'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--episode', help='number of episode to eval', default=1, type=int)
    args = parser.parse_args()

    ENV_NAME = args.env
    logger.info("Environment Name: {}".format(ENV_NAME))
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['policy']))
        if args.task == 'play':
            play_n_episodes(get_player(train=False), pred,
                            args.episode, render=True)
        elif args.task == 'eval':
            eval_model_multithread(pred, args.episode, get_player)
        elif args.task == 'dump_video':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                pred, args.episode)
    else:
        train()