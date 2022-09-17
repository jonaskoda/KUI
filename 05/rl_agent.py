import kuimaze
import numpy as np
import sys
import os
import gym
import time


def evaluate_board(q_table):                              # extract the final policy
    policy = {}
    best_value = 0
    best_move = 0
    best_direction_coords = 0

    for i in range(len(q_table)):                         # loop through q-table and choose the best moves
        for j in range(len(q_table[i])):
            best_direction_coords = (i, j) 
            for direction, q_value in enumerate(q_table[i][j]):
                if q_value > best_value:
                    best_value = q_value
                    best_move = direction
            policy[best_direction_coords] = best_move     # assign each coordinates sign of the best move direction
    
    return policy


def get_next_best_action(epsilon, state, q_table):        # implementation of epsilon greedy policy
    best_action = 0 
    n = np.random.random()

    if n > epsilon:                                       # if a random number happens to be larger than epsilon 
        best_action = np.random.randint(4)                # choose random move
    else:
        for direction, q_value in enumerate(q_table[state[0]][state[1]]):  
            if q_value > best_action:                     # else choose the best move
                best_action = direction

    return best_action


def learn_policy(env):
    '''
    Define constants:
    '''
    # Maze size
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))

    # Number of discrete actions
    num_actions = env.action_space.n
    # Q-table:
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)   #initialize all q_values to 0

    start_time = time.time()
    obv = env.reset()
    epsilon = 0.1
    gamma = 0.999
    alpha = 0.6
    state = obv[0:2]
    MAX_T = 1000                                                     # max trials (for one episode)

    while abs(start_time - time.time()) < 18:
        alpha = alpha * 0.99                                         # alpha decay
        t = 0                                                        # trial counter
        obv = env.reset()
        state = obv[0:2]
        is_done = False

        while not is_done and t < MAX_T:
            t += 1
            action = get_next_best_action(epsilon, state, q_table)   # use a function to choose a best action up/down/left/right 

            obv, reward, is_done, _ = env.step(action)               # make the action, observe new info about the environment
            
            next_state = obv[0:2]                                    # first and second info tells the coordinates

            q_value = q_table[state[0]][state[1]][action] 
            best_next_reward = 0
            for i in q_table[next_state[0]][next_state[1]]:
                if i > best_next_reward:
                    best_next_reward = i
            
            trial = reward + gamma * best_next_reward
            q_value = q_value + alpha * (trial - q_value)

            q_table[state[0]][state[1]][action] = q_value            # update the q_value

            state = next_state

    policy = evaluate_board(q_table)
    return policy


