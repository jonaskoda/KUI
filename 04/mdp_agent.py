import math
import kuimaze
import random
import copy

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
    env = kuimaze.MDPMaze()  
    # env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=None)
    # env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
    env.reset()
    print(env.get_all_states())
    # policy1 = find_policy_via_value_iteration(env)
    policy = find_policy_via_policy_iteration(env,0.9999)
   
