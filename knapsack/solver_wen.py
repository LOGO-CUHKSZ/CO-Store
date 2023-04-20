#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from queue import PriorityQueue


from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full

    # value = 0
    # weight = 0
    # taken = [0]*len(items)

    # for item in items:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight

    # value, taken = greedy_knapsack(items, capacity)
    # value, taken = dp_knapsack(items, capacity)
    value, taken = bb_knapsack(items, capacity)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_knapsack(items, capacity):
    taken = [0]*len(items)
    return taken


def greedy_knapsack(items, capacity):
    value = 0
    weight = 0
    taken = [0]*len(items)

    value_ratios = [item.value/item.weight for item in items]

    sort_index = [i for i, x in sorted(enumerate(value_ratios), reverse=True, key=lambda x: x[1])]
    for idx in sort_index:
        item = items[idx]
        if weight + item.weight <= capacity:
            taken[idx] = 1
            value += item.value
            weight += item.weight

    return value, taken


def dp_knapsack(items, capacity):
    value = 0
    taken = [0]*len(items)

    sorted(items, key=lambda x: x.value/x.weight)

    dp_table = [0] * (capacity + 1)
    item_select = [[] for i in range(capacity + 1)]

    for idx, item in enumerate(items):
        if item.weight > capacity:
            pass
        for l in range(capacity, item.weight - 1, -1):
            new_value = dp_table[l - item.weight] + item.value
            if new_value > dp_table[l]:
                dp_table[l] = new_value
                item_select[l] = item_select[l - item.weight] + [idx]

    for idx in item_select[-1]:
        taken[idx] = 1
    value = dp_table[-1]

    return value, taken


def bb_knapsack(items, capacity):
    items = sorted(items, key=lambda k: float(k.value)/k.weight, reverse = True)
    v = [item.value for item in items]
    w = [item.weight for item in items]
    item_count = len(items)

    # recursive dfs part

    # sys.setrecursionlimit(100000)
    # def recursive_dfs(state, best, level, value, weight):
    #     # feasible condition
    #     if weight > capacity:
    #         return
        
    #     # bound conditon
    #     bound = value
    #     wt = weight

    #     j = level
    #     while j < item_count and wt + w[j] <= capacity:
    #         bound += v[j]
    #         wt += w[j]
    #         j += 1
    #     # fill knapsack with fraction of a remaining item
    #     if j < item_count:
    #         bound += (capacity - wt) * float(v[j])/ w[j]
    #     if bound < best[0]:
    #         return 

    #     # update base
    #     if best[0] <= value:
    #         best[0] = value
    #         best[1] = state
        
    #     # branch
    #     if level < item_count:
    #         left = state.copy()
    #         left[level] = 1
    #         right = state.copy()
    #         right[level] = 0
    #         recursive_dfs(left, best, level + 1,  value + items[level].value, weight + items[level].weight)
    #         recursive_dfs(right, best, level + 1, value, weight)

    # init_state = [-1] * item_count
    # best = [0, []]
    # recursive_dfs(init_state, best, 0, 0, 0)
    # best_value, best_state = best


    # non recursive dfs part

    Node = namedtuple("Node", ['state', 'level', 'value', 'weight'])
    def non_recursive_dfs():
        stack = []
        best_value = 0
        best_state = []
        node = Node([-1] * item_count, 0, 0, 0)
        stack.append(node)
        while stack:
            node = stack.pop()

            # feasible condition
            if node.weight > capacity:
                continue

            # bound condition
            j = node.level
            bound = node.value
            wt = node.weight
            
            while j < item_count and wt + w[j] <= capacity:
                bound += v[j]
                wt += w[j]
                j += 1
            # fill knapsack with fraction of a remaining item
            if j < item_count:
                bound += (capacity - wt) * float(v[j])/ w[j]
            
            if bound < best_value:
                continue

            # update best
            if best_value <= node.value:
                best_value = node.value
                best_state = node.state

            # branch
            if node.level < item_count:
                right_state = node.state.copy()
                right_state[node.level] = 0
                right_node = Node(right_state, node.level + 1, node.value, node.weight)
                stack.append(right_node)
                left_state = node.state.copy()
                left_state[node.level] = 1
                left_node = Node(left_state, node.level + 1, node.value + v[node.level], node.weight + w[node.level])
                stack.append(left_node)
        
        return best_value, best_state
    
    best_value, best_state = non_recursive_dfs()

    # state to taken transform
    taken = [0] * item_count
    for idx, x in enumerate(best_state):
        if x == 1:
            taken[items[idx].index] = 1

    return best_value, taken


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

