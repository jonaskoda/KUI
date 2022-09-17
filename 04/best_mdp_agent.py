#!/usr/bin/env python3

import math
from math import gamma
import kuimaze
import random
import os
import time
import sys
import copy


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
        utils[(state.x, state.y)] = state.reward                # problem.get_state_reward(state)
    return utils

def assign_values(problem, states):                             # initialize dictionaries to work with
    policy, values, new_values = dict(), dict(), dict()
    for state in states:                                        # fill the dictionaries with appropriate values
        if problem.is_terminal_state(state):  
            policy[state.x, state.y] = None
            values[state.x, state.y] = state.reward
            new_values[state.x, state.y] = state.reward
        else:
            actions = [action for action in problem.get_actions(state)]
            policy[state.x, state.y] = random.choice(actions)   # initialize random policy
            values[state.x, state.y] = 0
            new_values[state.x, state.y] = 0
    return policy, values, new_values

def find_max(problem, state, values, value_iter):               # find the best action with the highest value (UP/DOWN/RIGHT/LEFT)
    direction = None 
    max = -math.inf
    for action in problem.get_actions(state):
        sum = 0
        for prob in problem.get_next_states_and_probs(state, action):
            sum += (prob[1] * values[(prob[0][0], prob[0][1])]) # multiply value at coords (x,y) by probability and add it to the sum
        if sum > max:                                           # if the new action is the best one, set it that way
            max = sum
            direction = action
    if value_iter:                                              # if the function is called by value iteration function - return both
        return max, direction                                   # value and action
    else:
        return direction                                        # in case of policy iteration - return only action

def find_policy_via_value_iteration(problem, discount_factor, epsilon):

    states = [state for state in problem.get_all_states()]
    policy, values, new_values = assign_values(problem, states)
  
    while True:
        delta = 0
        values = copy.deepcopy(new_values)                      # create a copy of the changed values for future work
        for state in states:
            if problem.is_terminal_state(state):
                continue        
            biggest_val, best_act = find_max(problem, state, values, True) # find the best action for a given position

            new_val = state.reward + (discount_factor * biggest_val)       # calculate the value of the position

            difference = abs(new_val - values[state.x, state.y])           # calculate the abs. change of value
            if difference > delta:
                delta = difference
            new_values[state.x, state.y] = new_val
            policy[state.x, state.y] = best_act   

        if delta < epsilon:                                                # terminating condition
            break
    return policy

def eval_action(problem, state, action, values):                           # execute the given action, 
    direction = None                                                       # return value for the given state
    max = -math.inf
    sum = 0
    for new_state in problem.get_next_states_and_probs(state, action):
        sum += (new_state[1] * values[(new_state[0][0], new_state[0][1])]) # multiply probability by value at coords (x,y)
    if sum > max:
        max = sum
    return max

def policy_evaluation(problem, discount_factor, states, policy, values, new_values):   # given a policy, evaluate each grid on a board
    epsilon = 0.05                                                                     # static value for termination (not the best practice)
    while True:
        delta = 0                                                                    
        values = copy.deepcopy(new_values)                                             # create a copy of the changed values for future work
        for state in states:
            if problem.is_terminal_state(state):
                continue        
            result_val = eval_action(problem, state, policy[state.x, state.y], values) # execute the action given by policy
            new_val = state.reward + (discount_factor * result_val)                    # calculate the value of the position
            difference = abs(new_val - values[state.x, state.y])                       # calculate the abs. change of value
            if difference > delta:
                delta = difference
            new_values[state.x, state.y] = new_val  

        if delta < epsilon:                                                            # terminating condition
            break
    return new_values

def policy_improvement(problem, state, new_values, policy, unchanged):                 # update action with new values provided
    best_act = find_max(problem, state, new_values, False)                             # find the best action for a given grid
    if best_act != policy[state.x, state.y]:                                           # if the new action does not match -> update it
        unchanged = False                                                              # set the boolean controling change to false
        policy[state.x, state.y] = best_act                                            # assign the action to the policy
    return policy, unchanged

def find_policy_via_policy_iteration(problem,discount_factor):

    states = [state for state in problem.get_all_states()]
    policy, values, new_values = assign_values(problem, states)                        # initialize dictionaries to work with

    while True:
        new_values = policy_evaluation(problem, discount_factor, states, policy, values, new_values) # returns new values of grids to work with
        unchanged = True
        for state in states:                           
            if problem.is_terminal_state(state):                                       # in case of terminal state - skip
                continue 
            policy, unchanged = policy_improvement(problem, state, new_values, policy, unchanged) # returns a new policy + change (boolean) 
            if not unchanged:
                unchanged = False
        if unchanged:                                                                  # terminating codition
            break
    return policy


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

    print(env.get_all_states())
    # policy1 = find_policy_via_value_iteration(env)
    policy = find_policy_via_policy_iteration(env,0.95)
    env.visualise(get_visualisation_values(policy))
    env.render()
    wait_n_or_s()
    print()
    print('Policy:', policy)
    utils = init_utils(env)
    env.visualise(get_visualisation_values(utils))
    env.render()
    time.sleep(5)
