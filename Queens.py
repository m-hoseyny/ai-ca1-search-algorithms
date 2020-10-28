#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:16:56 2019

@author: Mohammad Hosseini
"""
import numpy as np
import queue
import time, sys
from functools import wraps

BOARD_SIZE = 8
FILE_PATH = 'test_a.csv'


start_program_time = time.time()


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print ('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

def make_actoin_space():
    a = [-1, 0, 1]
    b = [-1, 0, 1]
    data = []
    for i in range(len(a)):
        for j in range(len(b)):
            if (a[i], b[j]) == (0, 0):
                continue
            data.append((a[i], b[j]))
    return data

def print_grid(grid):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if grid[i, j] == 1:
                print('x', end=' ')
            else:
                print('o', end=' ')
        print()

def make_grid(queens_list):
    grid = np.zeros((8,8), dtype=int)
    for i in range(len(queens_list)):
        grid[queens_list[i][0], queens_list[i][1]] = 1
    return grid

def get_input():
    queens_list = []
    with open(FILE_PATH) as f:
        for i in range(BOARD_SIZE):
            row = f.readline().strip().split(',')
            row = np.asarray(list(map(int, row))) - 1
            queens_list.append(row)
    return np.asanyarray(queens_list)

def check_state(queens_list):
    if len(set(queens_list[:,0])) != len(queens_list[:,0]) or len(set(queens_list[:, 1])) != len(queens_list[:, 1]):
        return False
    for i in range(len(queens_list)):
        for j in range(len(queens_list)):
            if i == j:
                continue
            diff = abs(queens_list[i] - queens_list[j])  
            if diff.tolist() == [0, 0]:
                return False
            if np.array_equal(abs(diff)//max(diff), [1, 1]):
                return False
    return True
    
def make_an_action(queens_list, action, queen_number):   
    actions = np.zeros([8,2], dtype=int)
    actions[queen_number,:] = action
    new = queens_list + actions
    if -1 in new or BOARD_SIZE in new: 
        return None
    else:
        return new
action_space = make_actoin_space()

@timing
def BFS(queens_list):
    ql = np.copy(queens_list)
    q = queue.Queue()
    q.put(ql)
    itt = 0
    start = time.time()
    while not q.empty():
        current_state = q.get()
        if itt % 100000 == 0:     
            print("Number of Iteration : ", itt)
            print_grid(make_grid(current_state))
            print("BFS - Elapsed Time : ", time.time()-start)
            print("Fontier size : ", q.qsize())
            print("----------------")
        for i in range(len(ql)): # take actions for each queen
            for action in action_space:
                l_ql = np.copy(current_state)
                l_ql = make_an_action(l_ql, action, i)
                itt += 1
                if l_ql is not None:
                    q.put(l_ql)
                else:
                    continue
                if check_state(l_ql):
                    print("Solution find")
                    return l_ql  , itt                         
            
    print("ALgorithm does not have any solution")
    
    
def _dls(queens_list, depth, it):
    it[0] += 1
    if it[0] % 1000000 == 0:
        print('it : ', it[0])
    if queens_list is None:
        return None
    if check_state(queens_list):
        print("IDS solution found\n")
        return queens_list
    if depth <= 0:
        return None
    for i in range(len(queens_list)):
        for action in action_space:
            l_ql = np.copy(queens_list)
            l_ql = make_an_action(l_ql, action, i)
            l_ql = _dls(l_ql, depth-1, it) 
            if l_ql is not None:
                return l_ql
    return None


@timing
def IDS(queens_list, max_depth):
    it = [0]
    for i in range(max_depth):
        print("depth : ", i)
        res = _dls(queens_list, i, it)
        if res is not None:
            return res, it[0]
    if res is None:
        print("Solution not found")
    return res
    

def heuristic_free_cells(queens_list):
    grid = make_grid(queens_list)
    grid = 1 - grid
    print(grid)
    for queen in queens_list:
        x = queen[0]
        y = queen[1]
        grid[x,:] = 0
        grid[:,y] = 0
        for k in range(min(x, BOARD_SIZE-y)):
            grid[x-k][y+k] = 0
        for k in range(min(BOARD_SIZE-x,y)):
            grid[x + k][y - k] = 0
        for k in range(min(BOARD_SIZE - x, BOARD_SIZE - y)):
            grid[x + k][y + k] = 0
        for k in range(min(BOARD_SIZE + x, BOARD_SIZE + y)):
            grid[x - k][y - k] = 0
        print(grid)
    return sum(sum(grid)), np.random.choice([_ for _ in range(8)], 1)
        

def heuristic_attacked_queens(queens_list):
    attacked_count = np.zeros(BOARD_SIZE)
    for i in range(len(queens_list)):
        for j in range(len(queens_list)):
            if i == j:
                continue
            if queens_list[i,0] == queens_list[j,0] or queens_list[i,1] == queens_list[j,1]:
                attacked_count[i] += 1
            diff = abs(queens_list[i] - queens_list[j])  
            if diff.tolist() == [0, 0]:
                attacked_count[i] += 1
                continue
            if np.array_equal(abs(diff)//max(diff), [1, 1]):
                attacked_count[i] += 1
                continue
    return int(sum(attacked_count)//2), np.argmax(attacked_count)
#print(heuristic_attacked_queens(queens_list))

@timing 
def A_star(queens_list, f_heuristic):
    start = time.time()
    q = np.array([[queens_list, 0]])
    itt = 0
    print(q[:,1])
    while q.size != 0:
        q = q[q[:,1].argsort()]
        current_state, total_cost = q[0]
        q = np.delete(q, 0, 0)
        if itt % 1000 == 0:     
            print("Number of Iteration : ", itt)
            print_grid(make_grid(current_state))
            print("A* - Elapsed Time : ", time.time()-start)
            print("Fontier size : ", q.size)
            print("----------------")
#        if itt == 9:
#            return
        cost, queen_number = f_heuristic(current_state)
#        print_grid(make_grid(current_state))
#        print(cost, ' ', queen_number)
        itt += 1
        for action in action_space:
            l_ql = np.copy(current_state)
            l_ql = make_an_action(l_ql, action, queen_number)
            if l_ql is not None:
                q = np.concatenate((q,[[l_ql, cost+total_cost]]))
            else:
                continue
            if check_state(l_ql):
                print("Solution find")
                return l_ql, itt                          
            
    print("ALgorithm does not have any solution")
    
queens_list = get_input()
print((queens_list))

######## BFS #########
bfs_solution = BFS(queens_list)
print(check_state(queens_list))
print(bfs_solution)
print(print_grid(make_grid(bfs_solution[0])))

###### IDS #########
sys.setrecursionlimit(1000000)
ids_solution = IDS(queens_list, 8)
print(ids_solution)
if ids_solution is not None:
    print(print_grid(make_grid(ids_solution[0])))

###### A start #######
astar_solution = A_star(queens_list, heuristic_attacked_queens)
print(astar_solution)


print("total run time : ", (time.time()-start_program_time))
