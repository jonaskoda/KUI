#!/usr/bin/python3
'''
Very simple example how to use gym_wrapper and BaseAgent class for state space search 
@author: Zdeněk Rozsypálek, and the KUI-2019 team
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018, 2019
'''

import time
import kuimaze
import os
import random
import heapq
import math
import copy


class Agent(kuimaze.BaseAgent):
    '''
    Agent class that inherits kuimaze.BaseAgent class 
    '''
    def __init__(self, environment):
        self.environment = environment

    def calc_path_cost(self, positions, goal, g, path):    #method for calculating prices
        eval_pos = []
        for i in positions:                                #for each node

            coords, one_move_dist = i                      

            new_g = g + one_move_dist                      # calculate g(x) 

            x_dist = coords[0] - goal[0]
            y_dist = coords[1] - goal[1]
            h = math.sqrt((x_dist)**2 + (y_dist)**2)       # calculate h(x) 

            F = g + h                                      # calculate F(x) = g(x) + h(x)

            new_path = copy.deepcopy(path)
            new_path.append(coords)
            
            eval_pos.append((F, new_g, h, coords, new_path))                   # creation of evaluated node + adding to a list
        return eval_pos                                                        # returns a list with evaluated nodes


    def queue_adder(self, expanded_positions, expanded, visited, prior_queue):  
        cnt = 0                                                                                 
        for i in expanded_positions[0]:                            # method for processing expanded nodes   
                if i[3] in expanded:                               # eliminates number of needed checks  
                    continue                                       # by first checking whether a position 
                if i[3] not in visited:                            # was already expanded or not visited
                    heapq.heappush(prior_queue, i)
                    continue
                for index, value in enumerate(prior_queue): 
                    if i[3] == value[3]:
                        cnt +=1
                        if value[0] > i[0]:
                            prior_queue[index] = i                 # if the new value is smaller, update node             
                        elif value[0] == i[0]:                     # based on F(x) value and then on g(x)
                            if value[1] > i[1]:                       
                                prior_queue[index] = i
                if cnt == 0:    
                    heapq.heappush(prior_queue, i)
                cnt = 0
                visited[i[3]] = 1
       
        heapq.heapify(prior_queue)  

        return expanded, visited, prior_queue


    def find_path(self):
        '''
        returns a path_section as a list of positions [(x1, y1), (x2, y2), ... ].
        '''
        observation = self.environment.reset() 
        goal = observation[1][0:2]
        position = observation[0][0:2]                         # initial state (x, y) = position
        print("Starting random searching")
        no_sol = 0
        prior_queue = []
        expanded = {}
        visited = {}
        path = [position]
        full_position = (0, 0, 0, position, path)

        # A* algorithm
        while True:
            new_positions = self.environment.expand(position)  # returns [[(x1, y1), cost], [(x2, y2), cost], ... ]
            expanded[position] = 1                             # adding to a dictionary for fast searching of expanded coords
            visited[position] = 1                              # adding to a dictionary for fast searching of visited coords
            expanded_positions = [self.calc_path_cost(new_positions, goal, full_position[1], full_position[4])]   

            if not prior_queue:                                # starting case
                prior_queue.append(expanded_positions[0][0])
            
            self.queue_adder(expanded_positions, expanded, visited, prior_queue) 
                            
            for position in prior_queue:
                full_position = heapq.heappop(prior_queue)  
                if full_position[3] not in expanded:           # select first not expanded from priotity queue  
                    break                                       
            position = full_position[3]       
                                  
            if position == goal:                               # break the loop when the goal position is reached
                print("goal reached")
                break
            if len(expanded) == len(visited):
                no_sol = 1
                break
            self.environment.render()               # show enviroment's GUI          
            time.sleep(0.1)                         # sleep for demonstration     
        if no_sol == 0:
            return full_position[4]                 # prikaz pro vytisknuti cesty z baseagent.find_path
        elif no_sol == 1:
            return None


if __name__ == '__main__':

    MAP = 'maps/easy/easy3.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path = agent.find_path()
    print(path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(3)
