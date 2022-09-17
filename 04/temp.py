#!/usr/bin/env python3

from calendar import c
from tkinter import LEFT, ttk
from turtle import Turtle
from types import new_class
from xxlimited import new

#from certifi import where

#from numpy import _FlatIterSelf
import kuimaze
import random
import os
import time
import sys
import copy
import math

MAP = 'maps/easy/easy1.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
PROBS = [0.4, 0.3, 0.3, 0]
GRAD = (0, 0)
SKIP = False
SAVE_EPS = False
VERBOSITY = 0


GRID_WORLD4 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
               [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 0, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

REWARD_NORMAL_STATE = -0.04
REWARD_GOAL_STATE = 1
REWARD_DANGEROUS_STATE = -1

GRID_WORLD3_REWARDS = [[REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_GOAL_STATE],
                       [REWARD_NORMAL_STATE, 0, REWARD_NORMAL_STATE, REWARD_DANGEROUS_STATE],
                       [REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE]]


def wait_n_or_s():
    def wait_key():
        '''
        returns key pressed ... works only in terminal! NOT in IDE!
        '''
        result = None
        if os.name == 'nt':
            import msvcrt
            result = msvcrt.getch()
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




def get_visualisation_values(dictvalues):
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        # ret.append({'x': key[0], 'y': key[1], 'value': [value, value, value, value]})
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret

# the init functions are provided for your convenience, modify, use ...
def init_policy(problem):
    policy = dict()
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue
        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = random.choice(actions)
    return policy

def init_utils(problem):
    '''
    Initialize all state utilities to zero except the goal states
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of utilities, indexed by state coordinates
    '''
    utils = dict()
    x_dims = problem.observation_space.spaces[0].n
    y_dims = problem.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            utils[(x,y)] = 0

    for state in problem.get_all_states():
        utils[(state.x, state.y)] = state.reward # problem.get_state_reward(state)
    return utils


def find_policy_via_policy_iteration(problem, discount_factor):
    policy = init_policy(problem)
    return(policy)
#potrebujeme porovnat s predchozim
#jak to chces na tu desku zahrat? Je potreba mit vlastni desku?
#prepsat stavy a ulozit
def check_convergence(problem, epsilon, old_states, actual_states):
    idx = 0
    for key in actual_states.keys():
        if(abs(actual_states[key] - old_states[key]) > epsilon):
            return False
        else:
            idx += 1
    return True


def actual_states_init(actual_states):
    for item in actual_states:
        item[2] = 0
    
def make_grid_from_states(problem, states):
    length = len(states)
    rows = problem.observation_space.spaces[0].n
    cols = problem.observation_space.spaces[1].n
    grid = list()

    for i in range(rows):
        grid.append([])
        for j in range(cols):
            grid[i].append(0)
    
    for item in states:
        x,y, reward = item
        grid[x][y] = reward
    return grid

def extract_policy(problem, states, actions, end_states):
    dict = {}
    grid = make_grid_from_states(problem, states)
    #print(actions[0].value)
    
    #actions = sorted(actions, lambda x: x.value, reverse=True)
    move = [[0,-1], [0,1], [-1,0], [1,0]] # UP, DOWN, LEFT RIGHT
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            #UP ,...
            if((i,j) in end_states):
                dict[(i,j)] = None
                continue
            candidates = list()
            max_val = -math.inf
            best_coords = () # tohle neni potreba 
            action = None
            row = i; col = j-1 #UP
            if((i >= 0 and i < len(grid)) and (j >= 0 and j < len(grid[0]))):
                if(max_val < grid[row][col]):
                    action = 'UP'
                    max_val = grid[row][col]
                    best_coords = (row, col)
            row = i; col = j+1 #DOWN
            if((row >= 0) and (row < len(grid)) and (col >= 0) and (col < len(grid[0]))):
                if(max_val < grid[row][col]):
                    action = 'DOWN'
                    max_val = grid[row][col]
                    best_coords = (row, col)
            row = i-1; col = j #LEFT
            if((row >= 0 and row < len(grid)) and (col >= 0 and col < len(grid[0]))):
                if(max_val < grid[row][col]):
                    action = 'LEFT'
                    max_val = grid[row][col]
                    best_coords = (row, col)
            row = i+1; col = j #RIGHT
            if((row >= 0 and row < len(grid)) and (col >= 0 and col < len(grid[0]))):
                if(max_val < grid[row][col]):
                    action = 'RIGHT'
                    max_val = grid[row][col]
                    best_coords = (row, col)
            #print(actions)
            for item in actions:
                if(item.name == action):
                    action = item
                    break

            dict[(i,j)] = action

    return dict

def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    old_states = {}
    actual_states = {}
    policy = {}
    delta = -math.inf
    for item in problem.get_all_states():
        actual_states[(item[0], item[1])] = item[2]
    old_states = copy.deepcopy(actual_states)
    for key in old_states.keys():
        old_states[key] = 0
    cnt = 0
    while(cnt != 500):#dokud nesedi epsilon
        delta = 0
        for state in problem.get_all_states():
            if(problem.is_terminal_state(state)):
                policy[(state[0], state[1])] = None
                continue # pokud je to end state, pokracujeme s jinym stavem, policy z end state nema smysl
            
            current_max = -math.inf
            for action in problem.get_actions(state):
                biggest_eval = 0
                for next_state in problem.get_next_states_and_probs(state, action):
                    #prvni secist vsechny probs
                    biggest_eval += (next_state[1]*old_states[(next_state[0][0], next_state[0][1])])

                evaluation = state[2] + (discount_factor * biggest_eval)
                if(evaluation > current_max):
                    current_max = evaluation
                    policy[(state[0], state[1])] = action
                    delta = max(abs(actual_states[(state[0], state[1])] - old_states[(state[0], state[1])]), delta)
                    actual_states[(state[0], state[1])] = current_max

        if(delta < epsilon*(1-discount_factor)/discount_factor):
            break
        old_states = copy.deepcopy(actual_states)
        cnt +=1
    
    return policy

#jde nam o to spocitat value a dosadit ji zpatky do vsech stavu;
#potrebujem dve SVOJE DESKY z originalni bereme pouze hodnotu R pri prechodu
#jediny co se meni je discount_factor*tocojetamted
'''def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    old_states = list()
    actual_states = list()
    value_candidates = list()
    end_states = list()
    actions = list()
    for item in problem.get_all_states():
        old_states.append(list(item))
    actual_states = copy.deepcopy(old_states)
    actual_states_init(actual_states)
    new_value = 0
    while(check_convergence(problem, epsilon, old_states, actual_states) == False):#dokud nesedi epsilon
        idx = 0
        for state in problem.get_all_states():
            if(problem.is_terminal_state(state)):
                end_states.append((state[0], state[1]))
                continue # pokud je to end state, pokracujeme s jinym stavem, policy z end state nema smysl
            for action in problem.get_actions(state):
                if action not in actions : actions.append(action)
                new_value = 0
                for next_state in problem.get_next_states_and_probs(state, action):
                    #spocitej soucet pres vsechny probs, zjistime tim hodnotu tohoto stavu
                    #Bellman update
                    new_value += next_state[1]*(state[2] + discount_factor*old_states[idx][2])
                    new_value += state[2] *(next_state[1]*actual_states
                value_candidates.append(new_value)
            current_best = max(value_candidates)
            while(value_candidates) : value_candidates.pop(0)
            actual_states[idx][2] = current_best # update values
            idx += 1
        old_states = copy.deepcopy(actual_states)
    
    return extract_policy(problem, actual_states, actions, end_states)
    #greedy, find policy and return
    #return policy'''

if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=GRID_WORLD3_REWARDS)
    # env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=None)
    # env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
    env.reset()
    print('====================')
    print('works only in terminal! NOT in IDE!')
    print('press n - next')
    print('press s - skip to end')
    print('====================')
    #check_convergence(env)
    #x = env.get_all_states()
    #utils = init_utils(env)
    test = find_policy_via_value_iteration(env, 0.9, 0.0001)
    
    #y = env.get_actions(x[0])
    #actions = [action for action in env.get_actions(x[0])]
    #z = list()
    #for item in actions:
    #    z.append(env.get_next_states_and_probs(x[0], item))
    #print(x[8][2]) #takhle dostanu reward
    #print(env.get_all_states())
    
    # policy1 = find_policy_via_value_iteration(env)
    policy = find_policy_via_policy_iteration(env,0.9999)
    print(type(policy.get((1,2))))
    env.visualise(get_visualisation_values(policy))
    env.render()
    wait_n_or_s()
    print('Policy:', policy)
    utils = init_utils(env)
    env.visualise(get_visualisation_values(utils))
    env.render()
    time.sleep(5)
