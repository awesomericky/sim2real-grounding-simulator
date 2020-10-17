#coding: utf-8
from collections import deque
from copy import deepcopy
import tensorflow as tf
import numpy as np
import itertools
import pickle
import random
import copy
import time
import os

EPS = 1e-10

class SGAT_net:
    def __init__(self, real_env, sim_env, args):
        self.real_env = real_env
        self.sim_env = sim_env
        self.real_name = args['agent_real_name']
        self.sim_name = args['agent_sim_name']
        self.real_checkpoint_dir='{}/forward_checkpoint'.format(args['env_name'])
        self.sim_checkpoint_dir = '{}/backward_checkpoint'.format(args['env_name'])
        self.real_state_dim = real_env.observation_space.shape[0]
        self.sim_state_dim = sim_env.observation_space.shape[0]
        try:
            self.real_action_dim = real_env.action_space.shape[0]
            self.real_action_bound_min = real_env.action_space.low
            self.real_action_bound_max = real_env.action_space.high
            self.sim_action_dim = sim_env.action_space.shape[0]
            self.sim_action_bound_min = sim_env.action_space.low
            self.sim_action_bound_max = sim_env.action_space.high
        except:
            self.real_action_dim = 1
            self.real_action_bound_min = - 1.0
            self.real_action_bound_max = 1.0
            self.sim_action_dim = 1
            self.sim_action_bound_min = - 1.0
            self.sim_action_bound_max = 1.0
        self.forward_hidden1_units = args['forward_hidden1']
        self.forward_hidden2_units = args['forward_hidden2']
        self.backward_hidden1_units = args['backward_hidden1']
        self.backward_hidden2_units = args['backward_hidden2']
        self.forward_lr = args['forward_lr']
        self.backward_lr = args['backward_lr']
        self.forward_epochs = args['forward_epochs']
        self.backward_epochs = args['backward_epochs']
        self.forward_batch_size = args['forward_batch_size']
        self.backward_batch_size = args['backward_batch_size']

        with tf.variable_scope(self.real_name):
            # placeholder
            self.real_states = tf.placeholder(tf.float32, [None, self.real_state_dim], name='Real_states')
            self.real_actions = tf.placeholder(tf.float32, [None, self.real_action_dim], name='Real_actions')
            self.real_targets = tf.placeholder(tf.float32, [None, self.real_state_dim], name='Real_targets')

            # forward model output
            self.mean, self.std = self.build_forward_model('forward')

            # forward model loss & optimizer
            fw_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/forward')
            self.log_prob = - tf.reduce_sum(tf.log(self.std + EPS) + 0.5*np.log(2*np.pi) + tf.squared_difference(self.real_targets, self.mean) / (2 * tf.square(self.std) + EPS), axis=1)
            self.fw_loss = - tf.reduce_mean(self.log_prob)
            fw_optimizer = tf.train.AdamOptimizer(learning_rate=self.forward_lr)
            self.fw_gradients = tf.gradients(self.fw_loss, fw_vars)
            self.fw_train_op = fw_optimizer.apply_gradients(zip(self.fw_gradients, fw_vars))

        with tf.variable_scope(self.sim_name):
            # placeholder
            self.sim_states = tf.placeholder(tf.float32, [None, self.sim_state_dim], name='Sim_states')
            self.sim_next_states = tf.placeholder(tf.float32, [None, self.sim_state_dim], name='Sim_next_states')
            self.sim_targets = tf.placeholder(tf.float32, [None, self.sim_action_dim], name='Sim_targets')

            # backward model output
            self.transformed_action = self.build_backward_model('backward')

            # backward model loss & optimizer
            bw_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/backward')
            self.bw_loss = 0.5*tf.square(self.sim_targets - self.transformed_action)
            self.bw_loss = tf.reduce_mean(self.bw_loss)
            bw_optimizer = tf.train.AdamOptimizer(learning_rate=self.backward_lr)
            self.bw_gradients = tf.gradients(self.bw_loss, bw_vars)
            self.bw_train_op = bw_optimizer.apply_gradients(zip(self.bw_gradients, bw_vars))
        
        # make session and load model
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.load()

    def build_forward_model(self, name='forward', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            inputs = tf.concat([self.real_states, self.real_actions], axis=1)
            model = tf.layers.dense(inputs, self.forward_hidden1_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.forward_hidden2_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            mean = tf.layers.dense(model, self.real_state_dim, activation='relu', bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            log_std = tf.get_variable('log_std', shape=(self.real_state_dim), initializer=tf.random_normal_initializer(mean=-1.0, stddev=0.01))
            std = tf.ones_like(mean)*tf.exp(log_std)
        return mean, std

    def build_backward_model(self, name='backward', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            inputs = tf.concat([self.sim_states, self.sim_next_states], axis=1)
            model = tf.layers.dense(inputs, self.backward_hidden1_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.backward_hidden2_units, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            outputs  = tf.layers.dense(model, self.sim_action_dim, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
        return outputs
    
    def forward_train(self, trajs):
        # data preprocessing
        states = np.array(trajs[0])  # (num_steps, state_dim)
        actions = np.array(trajs[1])  # (num_steps, action_dim)
        targets = np.array(trajs[2])  # (num_steps, state_dim)
        total_trajs = np.concatenate((states, actions, targets), axis=1)

        # data training
        for _ in range(self.forward_epochs):
            np.random.shuffle(total_trajs) # (num_steps, state_dim + action_dim + state_dim)
            states = total_trajs[:, :self.real_state_dim]
            actions = total_trajs[:, self.real_state_dim:self.real_state_dim+self.real_action_dim]
            targets = total_trajs[:, self.real_state_dim+self.real_action_dim:]

            start = 0
            while start < total_trajs.shape[0] - self.forward_batch_size:
                self.sess.run([self.fw_train_op], feed_dict={
                    self.real_states:states[start:start+self.forward_batch_size, :], 
                    self.real_actions:actions[start:start+self.forward_batch_size, :], 
                    self.real_targets:targets[start:start+self.forward_batch_size, :]})
                start += self.forward_batch_size
        fw_loss = self.sess.run(self.fw_loss, feed_dict={self.real_states:trajs[0], self.real_actions:trajs[1], self.real_targets:trajs[2]})
        return fw_loss
    
    def backward_train(self, trajs):
        # data preprocessing
        states = np.array(trajs[0])  # (num_steps, state_dim)
        next_states = np.array(trajs[1])  # (num_steps, state_dim)
        targets = np.array(trajs[2])  # (num_steps, action_dim)
        total_trajs = np.concatenate((states, next_states, targets), axis=1)

        # data training
        for _ in range(self.backward_epochs):
            np.random.shuffle(total_trajs)
            states = total_trajs[:, :self.sim_state_dim]
            next_states = total_trajs[:, self.sim_state_dim:self.sim_state_dim+self.sim_state_dim]
            targets = total_trajs[:, self.sim_state_dim+self.sim_state_dim:]

            start = 0
            while start < total_trajs.shape[0] - self.backward_batch_size:
                self.sess.run([self.bw_train_op], feed_dict={
                    self.sim_states:states[start:start+self.backward_batch_size, :], 
                    self.sim_next_states:next_states[start:start+self.backward_batch_size, :], 
                    self.sim_targets:targets[start:start+self.backward_batch_size, :]})
                start += self.backward_batch_size
        bw_loss = self.sess.run(self.bw_loss, feed_dict={self.sim_states:trajs[0], self.sim_next_states:trajs[1], self.sim_targets:trajs[2]})
        return bw_loss
    
    def forward_transform(self, state, action):
        state = state.reshape((1, len(state)))
        action = action.reshape((1, len(action)))
        [[mean], [std]] = self.sess.run([self.mean, self.std], feed_dict={self.real_states:state, self.real_actions:action})
        next_state = mean + np.multiply(np.random.normal(0, 1, np.shape(mean)), std)
        return next_state

    def backward_transform(self, state, next_state):
        state = state.reshape((1, len(state)))
        next_state = next_state.reshape((1, len(next_state)))
        [transformed_action] = self.sess.run(self.transformed_action, feed_dict={self.sim_states:state, self.sim_next_states:next_state})
        return transformed_action
    
    def save_forward_model(self):
        self.forward_saver.save(self.sess, self.real_checkpoint_dir+'/model.ckpt')
        print('Sucess to save forward model!')
    
    def save_backward_model(self):
        self.backward_saver.save(self.sess, self.sim_checkpoint_dir+'/model.ckpt')
        print('Sucess to save backward model!')
    
    def load(self):
        self.forward_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + self.real_name))
        self.backward_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + self.sim_name))

        if not os.path.isdir(self.real_checkpoint_dir):
            os.makedirs(self.real_checkpoint_dir)
        if not os.path.isdir(self.sim_checkpoint_dir):
            os.makedirs(self.sim_checkpoint_dir)
        
        forward_ckpt = tf.train.get_checkpoint_state(self.real_checkpoint_dir)
        backward_ckpt = tf.train.get_checkpoint_state(self.sim_checkpoint_dir)
        self.sess.run(tf.global_variables_initializer())

        if forward_ckpt and tf.train.checkpoint_exists(forward_ckpt.model_checkpoint_path):
            self.forward_saver.restore(self.sess, forward_ckpt.model_checkpoint_path)
            print('Sucess to load forward model!')
        else:
            print('Fail to load forward model...')

        if backward_ckpt and tf.train.checkpoint_exists(backward_ckpt.model_checkpoint_path):
            self.backward_saver.restore(self.sess, backward_ckpt.model_checkpoint_path)
            print('Sucess to load backward model!')
        else:
            print('Fail to load backward model...')