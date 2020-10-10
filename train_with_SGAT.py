# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

import sys
import os

from graph_drawer import Graph
from logger import Logger
from nets import Agent

import imp
nets = imp.load_source("nets", "/home/awesomericky/racecar_ws/src/RCcar/scripts/SGAT/nets.py")
env_sim = imp.load_source("env", "/home/awesomericky/racecar_ws/src/RCcar/scripts/env/env.py")
env_real = imp.load_source("env", "/home/awesomericky/Lab_intern/Prof_Oh/Code/RC_car/env/env.py")

from collections import deque
import tensorflow as tf
import numpy as np
import random
import pickle
import rospy
import time
import sys
# import wandb

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
SGAT_args = {'agent_real_name':agent_name + '_real',
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

## ROS setting for RCcar(simulator) must be launched before!
def train():
    global env_name, save_name, agent_args, env_real, env_sim, nets
    env_real =  env_real.Env_real(False)
    env_sim = env_sim.Env_sim(True)
    SGAT_model = nets.SGAT_net(env_real, env_sim, SGAT_args)
    agent = Agent(env_sim, agent_args)

    # wandb.init(project=save_name)
    accum_step = 0
    avg_temp_cost = 0

    v_loss_logger = Logger(save_name, 'v_loss')
    cost_v_loss_logger = Logger(save_name, 'cost_v_loss')
    kl_logger = Logger(save_name, 'kl')
    score_logger = Logger(save_name, 'score')
    cost_logger = Logger(save_name, 'cost')
    max_steps = 2000
    max_ep_len = 1000
    episodes = int(max_steps/max_ep_len)
    epochs = 2 #50
    save_freq = 1

    log_length = 10
    p_objectives = deque(maxlen=log_length)
    c_objectives = deque(maxlen=log_length)
    v_losses = deque(maxlen=log_length)
    cost_v_losses = deque(maxlen=log_length)
    kl_divergence = deque(maxlen=log_length)
    scores = deque(maxlen=log_length*episodes)
    costs = deque(maxlen=log_length*episodes)

    is_backup = False
    backup_name = '{}/backup.pkl'.format(save_name)
    if os.path.isfile(backup_name):
        #input_value = raw_input('backup file exists. wanna continue the last work?( y/n )')
        #if input_value != 'n':
        #    is_backup = True
        is_backup = True
    if is_backup:
        with open(backup_name, 'rb') as f:
            backup_list = pickle.load(f)
        start_iter = backup_list[0]
    else:
        start_iter = 0
        backup_list = [start_iter]

    for epoch in range(start_iter, epochs):
        #continue?
        print("="*20)
        print("Epoch : {}".format(epoch+1))
        #input_value = raw_input("wanna continue episodes?( y/n )")
        #if input_value == 'n':
        #    break

        states = []
        actions = []
        targets = []
        cost_targets = []
        gaes = []
        cost_gaes = []
        avg_costs = []
        ep_step = 0
        while ep_step < max_steps:
            #input_value = raw_input("ready?")

            state = env_sim.reset()
            done = False
            score = 0
            cost = 0
            step = 0
            temp_rewards = []
            temp_costs = []
            values = []
            cost_values = []
            while True:
                if rospy.is_shutdown():
                    sys.exit()
                step += 1
                ep_step += 1
                action, clipped_action, value, cost_value = agent.get_action(state, True)
                # action transformer by SGAT
                transformed_next_state = SGAT_model.forward_transform(state, clipped_action)
                transformed_action = SGAT_model.backward_transform(state, transformed_next_state)
                next_state, reward, done, info = env_sim.step(transformed_action)

                predict_cost = info['continuous_cost']

                states.append(state)
                actions.append(action)
                temp_rewards.append(reward)
                temp_costs.append(predict_cost)
                values.append(value)
                cost_values.append(cost_value)

                state = next_state
                score += reward
                cost += info.get('cost', 0)

                if done or step >= max_ep_len:
                    break

            print("step : {}, score : {}".format(step, score))
            if step >= max_ep_len:
                action, clipped_action, value, cost_value = agent.get_action(state, True)
            else:
                value = 0
                cost_value = 0
                print("done before max_ep_len...") 
            next_values = values[1:] + [value]
            temp_gaes, temp_targets = agent.get_gaes_targets(temp_rewards, values, next_values)
            next_cost_values = cost_values[1:] + [cost_value]
            temp_cost_gaes, temp_cost_targets = agent.get_gaes_targets(temp_costs, cost_values, next_cost_values)
            avg_costs.append(np.mean(temp_costs))
            targets += list(temp_targets)
            gaes += list(temp_gaes)
            cost_targets += list(temp_cost_targets)
            cost_gaes += list(temp_cost_gaes)

            score_logger.write([step, score])
            cost_logger.write([step, cost])
            scores.append(score)
            costs.append(cost)

            accum_step += step
            avg_temp_cost = np.mean(temp_costs)
            # wandb.log({'step': accum_step, 'score':score, 'cost':cost, 'avg_temp_cost':avg_temp_cost})

        trajs = [states, actions, targets, cost_targets, gaes, cost_gaes, avg_costs]
        v_loss, cost_v_loss, p_objective, cost_objective, kl = agent.train(trajs)

        v_loss_logger.write([ep_step, v_loss])
        cost_v_loss_logger.write([ep_step, cost_v_loss])
        kl_logger.write([ep_step, kl])

        p_objectives.append(p_objective)
        c_objectives.append(cost_objective)
        v_losses.append(v_loss)
        cost_v_losses.append(cost_v_loss)
        kl_divergence.append(kl)

        print(np.mean(scores), np.mean(costs), np.mean(v_losses), np.mean(cost_v_losses), np.mean(kl_divergence), np.mean(c_objectives))
        if (epoch+1)%save_freq == 0:
            agent.save()
            v_loss_logger.save()
            cost_v_loss_logger.save()
            kl_logger.save()
            score_logger.save()
            cost_logger.save()

        #backup
        backup_list[0] = epoch + 1
        with open(backup_name, 'wb') as f:
            pickle.dump(backup_list, f)

## ROS setting for RCcar(simulator) must be launched before!
def sim_test():
    global env_name, save_name, agent_args, env_real, env_sim, nets
    env_real = env_real.Env_real(False)
    env_sim = env_sim.Env_sim(True)
    SGAT_model = nets.SGAT_net(env_real, env_sim, SGAT_args)
    agent = Agent(env_sim, agent_args)

    episodes = int(10)
    max_steps = 2000

    for episode in range(episodes):
        state = env_sim.reset()
        done = False
        score = 0
        step = 0
        while not done and step <= max_steps:
            action, clipped_action, value, cost_value = agent.get_action(state, False)
            # action transformer by SGAT
            transformed_next_state = SGAT_model.forward_transform(state, clipped_action)
            transformed_action = SGAT_model.backward_transform(state, transformed_next_state)
            state, reward, done, info = env_sim.step(transformed_action)
            print(reward, '\t', info.get('cost', 0))
            score += reward
            step += 1
        print("score :",score)

## ROS setting for RCcar(real world) must be launched before!
def real_test():
    global env_name, save_name, agent_args, env_real
    env_real = env_real.Env_real(True)
    agent = Agent(env_real, agent_args)

    episodes = int(10)
    max_steps = 2000

    for episode in range(episodes):
        input_value = input('Ready? (y/n)')
        if input_value == 'n':
            break
        state = env_real.reset()
        done = False
        score = 0
        step = 0
        while not done and step <= max_steps:
            action, clipped_action, value, cost_value = agent.get_action(state, False)
            state, reward, done, info = env_real.step(clipped_action)
            print(reward, '\t', info.get('cost', 0))
            score += reward
            step += 1
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'sim_test':
    sim_test()
elif len(sys.argv)== 2 and sys.argv[1] == 'real_test':
    real_test()
else:
    train()
