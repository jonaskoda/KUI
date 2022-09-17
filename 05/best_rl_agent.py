#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
A sandbox for playing with the HardMaze
@author: Tomas Svoboda
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''


import kuimaze
import numpy as np
import sys
import os
import gym
import time


# MAP = 'maps/normal/normal3.bmp'
MAP = 'maps/easy/easy2.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
# PROBS = [0.8, 0.1, 0.1, 0]
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
SKIP = False
VERBOSITY = 2

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 255, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

# MAP = GRID_WORLD3


def wait_n_or_s():

    def wait_key():
        """
        returns key pressed ... works only in terminal! NOT in IDE!
        """
        result = None
        if os.name == 'nt':
            import msvcrt
            # https://cw.felk.cvut.cz/forum/thread-3766-post-14959.html#pid14959
            result = chr(msvcrt.getch()[0])
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        return result

    '''
    press n - next, s - skip to end ... write into terminal
    '''
    global SKIP
    x = SKIP
    while not x:
        key = wait_key()
        x = key == 'n'
        if key == 's':
            SKIP = True
            break


def get_visualisation(table):
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append({'x': j, 'y': i, 'value': [table[j][i][0], table[j][i][1], table[j][i][2], table[j][i][3]]})
    return ret

def evaluate_board(q_table):
    policy = {}
    best_value = 0
    best_move = 0
    best_direction_coords = 0

    for i in range(len(q_table)):
        for j in range(len(q_table[i])):
            best_direction_coords = (i, j) 
            for direction, q_value in enumerate(q_table[i][j]):
                if q_value > best_value:
                    best_value = q_value
                    best_move = direction
            policy[best_direction_coords] = best_move  
    
    return policy


def get_next_best_action(epsilon, state, q_table):         # implementation of epsilon greedy policy
    best_action = 0 
    n = np.random.random()

    if n > epsilon:
        best_action = np.random.randint(4)
    else:
        for direction, q_value in enumerate(q_table[state[0]][state[1]]):
            if q_value > best_action:
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
    MAX_T = 1000  # max trials (for one episode)

    while abs(start_time - time.time()) < 18:
        alpha = alpha * 0.99
        t = 0
        total_reward = 0
        obv = env.reset()
        state = obv[0:2]
        is_done = False
        while not is_done and t < MAX_T:
            t += 1
            #action = env.action_space.sample()   #env.action_space = vsechny akce
           
            action = get_next_best_action(epsilon, state, q_table)   #use a function to choose a best action up/down/left/right 
            print(action) 

            obv, reward, is_done, _ = env.step(action)

            #total_reward += reward   
            #total_length += length
            print(obv)
            print(reward)
            print(is_done)
            
            next_state = obv[0:2]                             #first and second info tells the coordinates

            q_value = q_table[state[0]][state[1]][action] 
            best_next_reward = 0
            for i in q_table[next_state[0]][next_state[1]]:
                if i > best_next_reward:
                    best_next_reward = i
            
            trial = reward + gamma * best_next_reward
            q_value = q_value + alpha * (trial - q_value)

            q_table[state[0]][state[1]][action] = q_value
            print(q_table)
            '''
            if VERBOSITY > 0:
                print(state, action, next_state, reward)
                env.visualise(get_visualisation(q_table))
                env.render()
                wait_n_or_s()
            '''
            state = next_state


    if not is_done:
        print('Timed out')

    #print('total_reward:', total_reward)
    print("done")

    policy = evaluate_board(q_table)
    print(policy)
    return policy


if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)

    if VERBOSITY > 0:
        print('====================')
        print('works only in terminal! NOT in IDE!')
        print('press n - next')
        print('press s - skip to end')
        print('====================')
    
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
    if VERBOSITY > 0:
        env.visualise(get_visualisation(q_table))
        env.render()
    learn_policy(env)
    '''
    if VERBOSITY > 0:
        SKIP = False
        env.visualise(get_visualisation(q_table))
        env.render()
        wait_n_or_s()

        env.save_path()
        env.save_eps()
    '''