#coding: utf-8
# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

import sys
import os

from nets import GAT_net

import imp
nets = imp.load_source("nets", "/home/awesomericky/racecar_ws/src/RCcar/scripts/CPO/nets.py")
env_sim = imp.load_source("env", "/home/awesomericky/racecar_ws/src/RCcar/scripts/env/env.py")
env_real = imp.load_source("env", "/home/awesomericky/Lab_intern/Prof_Oh/Code/RC_car/env/env.py")

from collections import deque
import tensorflow as tf
import numpy as np
import random
import pickle
# import wandb
import rospy
import time
import sys

#for random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

env_name = 'RCcar-v0'

agent_name = 'CPO'
algo = '{}_{}'.format(agent_name, seed)
save_name = '_'.join(env_name.split('-')[:-1])
save_name = "{}_{}".format(save_name, algo)
agent_args = {'agent_name':agent_name,
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':512,
            'hidden2':512,
            'v_lr':1e-3,
            'cost_v_lr':1e-3,
            'value_epochs':80,
            'cost_value_epochs':80,
            'num_conjugate':10,
            'max_decay_num':10,
            'line_decay':0.8,
            'max_kl':0.01,
            'max_avg_cost':25.0/1000.0, #25/1000,
            'damping_coeff':0.01,
            'gae_coeff':0.97,
            }
GAT_args = {'agent_real_name':agent_name + '_real',
            'agent_sim_name':agent_name + '_sim',
            'env_name':save_name,
            'forward_hidden1':64,
            'forward_hidden2':64,
            'backward_hidden1':64,
            'backward_hidden2':64,
            'forward_lr':1e-3,
            'backward_lr':1e-3,
            'forward_epochs':80,
            'backward_epochs':80,
            'forward_batch_size':50,
            'backward_batch_size':50
            }

env_real = env_real.Env_real(False)
env_sim = env_sim.Env_sim(False)
grounding = GAT_net(env_real, env_sim, GAT_args)

## ROS setting for RCcar(real world) must be launched before!
def forward_model_train():
    global save_name, env_real, grounding
    agent = nets.Agent(env_real, agent_args)
    env_real.set_ps()  # setting publisher and subscriber

    max_steps = 20000
    max_ep_len = 1000
    episodes = int(max_steps/max_ep_len)

    # initialize backup data
    is_backup = False
    backup_name = '{}/forward_backup.pkl'.format(save_name)
    if os.path.isfile(backup_name):
        input_value = input('Backup file for forward model exists. Wanna continue the last work? ( y/n )')
        if input_value != 'n':
           is_backup = True
    if is_backup:
        with open(backup_name, 'rb') as f:
            backup_list = pickle.load(f)
        start_step = backup_list[0]
    else:
        start_step = 0
        backup_list = [start_step, []]
    
    # collect data to train forward model
    states = []
    actions = []
    next_states = []
    break_bool = False
    while start_step < max_steps:
        input_value = input('Ready? (y/n)')
        if input_value == 'n':
            break_bool = True
            break
        state = env_real.reset()
        step = 0
        done = False
        while True:
            if rospy.is_shutdown():
                sys.exit()
            step += 1
            _, clipped_action, _, _ = agent.get_action(state, True)
            next_state, _, done, _ = env_real.step(clipped_action)

            states.append(state)
            actions.append(clipped_action)
            next_states.append(next_state)

            state = next_state
            if done or step >= max_ep_len:
                break
        start_step += step
        print("Left steps: {}".format(max_steps-start_step))

    trajs = [states, actions, next_states]
    # train forward model
    if not break_bool:
        fw_loss = grounding.forward_train(trajs)
        print('Forward model train loss: {}'.format(fw_loss))
        grounding.save_forward_model()

    # backup trajectory data
    backup_list[0] = start_step
    backup_list[1] = trajs
    with open(backup_name, 'wb') as f:
        pickle.dump(backup_list, f)

## ROS setting for RCcar(simulator) must be launched before!
def backward_model_train():
    global save_name, env_sim, grounding
    agent = nets.Agent(env_sim, agent_args)
    env_sim.set_ps()  # setting publisher and subscriber

    max_steps = 20000
    max_ep_len = 1000
    episodes = int(max_steps/max_ep_len)

    # initialize backup data
    backup_name = '{}/backward_backup.pkl'.format(save_name)
    start_step = 0
    backup_list = [start_step, []]
    
    # collect data to train backward model
    states = []
    actions = []
    next_states = []
    while start_step < max_steps:
        state = env_sim.reset()
        step = 0
        done = False
        while True:
            if rospy.is_shutdown():
                sys.exit()
            step += 1
            _, clipped_action, _, _ = agent.get_action(state, True)
            next_state, _, done, _ = env_sim.step(clipped_action)

            states.append(state)
            actions.append(clipped_action)
            next_states.append(next_state)

            state = next_state
            if done or step >= max_ep_len:
                break
        start_step += step
        print("Left steps: {}".format(max_steps-start_step))

    trajs = [states, next_states, actions]
    # train forward model
    bw_loss = grounding.backward_train(trajs)
    print('Backward model train loss: {}'.format(bw_loss))
    grounding.save_backward_model()

    # backup trajectory data
    backup_list[0] = start_step
    backup_list[1] = trajs
    with open(backup_name, 'wb') as f:
        pickle.dump(backup_list, f)

if len(sys.argv) == 2:
    if sys.argv[1] == 'forward':
        forward_model_train() 
    elif sys.argv[1] == 'backward':
        backward_model_train()
    else:
        print('Wrong command')
else:
    print('Wrong command')
