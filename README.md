# sim2real-grounding-simulator
Sim2real algorithm that grounds simulator as real world (for reinforcement learning)


# Contents
1) Grounded Action Transformation (GAT)
  - GAT/nets.py
  - GAT/grounding.py
  - train_with_GAT.py
2) Stochastic Grounded Action Transformation (SGAT)
  - SGAT/nets.py
  - SGAT/grounding.py
  - train_with_SGAT.py
3) data_example
  - toy data result after training agent using SGAT
  (just check how the result file structure will look like)


# Explanation
1) net.py
  - network needed in GAT / SGAT
2) grounding.py
  - training forward and backward model
3) train_with_GAT.py , train_with_SGAT.py: 
  - training agent using grounded forward and inverse model
  - action transformer is added to the optimization algorithm
  - optimization algorithm could be changed by the user 
  (In the uploaded code, optimization algorithm is CPO)


# Implementation
1) grounding.py
- python grounding.py forward 
(forward model training)
- python grounding.py backward
(backward model training)
2) train_with_GAT.py
- python train_with_GAT.py
(training agent using optimization algorithm)
- python train_with_GAT.py sim_test
(testing trained agent in simulation environment)
- python train_with_GAT.py real_test
(testing trained agent in real world environment)
3) train_with_SGAT.py
- python train_with_SGAT.py
(training agent using optimization algorithm)
- python train_with_SGAT.py sim_test
(testing trained agent in simulation environment)
- python train_with_SGAT.py real_test
(testing trained agent in real world environment)


# Experiment order
(0. Prepare agent model in '{Data folder}/checkpoint (ex) RCcar_CPO_1/checkpoint)' directory)
1. Train forward model, inverse model in '/GAT' or '/SGAT' directory (using 'grounding.py')
2. Move '{Data folder} (ex) RCcar_CPO_1)' in '/Optimize (ex) CPO)'
3. Train agent in '/Optimize (ex) CPO)' directory (using 'train_with_SGAT.py)
4. Move '{Data folder} (ex) RCcar_CPO_1)' in '/GAT' or '/SGAT' directory
5. Repeat 1~4

*'0' step is needed if behvior cloning is used for initialization

# Prerequisites
1) tensorflow 1.13.1
2) python 2.7.12
3) ros-kinetic
4) MIT racecar simulator


# Reference
1) 'Grounded Action Transformation for Robot Learning in Simulation'
https://www.cs.utexas.edu/users/AustinVilla/papers/AAAI17-Hanna.pdf
2) 'Stochastic Grounded Action Transformation for Robot Learning in Simulation'
https://arxiv.org/abs/2008.01281
3) 'Constrained Policy Optimization'
https://arxiv.org/abs/1705.10528
4) MIT racecar simulator
https://github.com/mit-racecar/racecar_simulator
