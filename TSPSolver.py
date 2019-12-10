#!/usr/bin/python3

import itertools
import heapq
import copy
from TSPClasses import *
import numpy as np
import time
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario
        return self.defaultRandomTour()

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedyFirstFind(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        for startCity in cities:
            if foundTour or time.time() - start_time >= time_allowance:
                break
            remaining_cities = cities.copy()
            remaining_cities.pop(count)
            route = [startCity]
            self.find_greedy_route(startCity, remaining_cities, route)
            bssf = TSPSolution(route)
            if bssf.cost < np.inf:
                foundTour = True
            count += 1
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def find_greedy_route(self, source, remaining_cities, route):
        if len(remaining_cities) == 0:
            return
        min_distance = source.costTo(remaining_cities[0])
        next_city = remaining_cities[0]
        next_city_index = 0
        for i in range(1, len(remaining_cities)):
            prospective_city = remaining_cities[i]
            distance = source.costTo(prospective_city)
            if distance < min_distance:
                min_distance = distance
                next_city = prospective_city
                next_city_index = i
        remaining_cities.pop(next_city_index)
        route.append(next_city)
        self.find_greedy_route(next_city, remaining_cities, route)


    def greedyBestFind(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        count = 0
        bssf = None
        start_time = time.time()
        for startCity in cities:
            if time.time() - start_time >= time_allowance:
                break
            remaining_cities = cities.copy()
            remaining_cities.pop(count)
            route = [startCity]
            self.find_greedy_route(startCity, remaining_cities, route)
            current_solution = TSPSolution(route)
            if bssf is None or current_solution.cost < bssf.cost:
                bssf = current_solution
            count += 1
        end_time = time.time()
        results['cost'] = bssf.cost if bssf is not None else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        # init starting values
        self.cities = self._scenario.getCities()
        start_time = time.time()
        # greedy solution is basically O(1) time since it's only the edges plus nodes which
        # is not really scalable in this NP problem, so it'll always be linear
        # space is O(n) to store every node
        bssf = self.greedyFirstFind(time_allowance)['soln']

        lowest_cost = bssf.cost
        total_solutions = 0
        max_queue_size = 1
        total_states_created = 1
        total_pruned = 0
        pq = []
        count = 0
        depth = 0

        # get and push initial state
        # reduce initial matrix is O(n^2) time and space as explained by the function
        initial_matrix, lower_bound = self.reduce_matrix(
            self.init_matrix(self.cities))
        initial_state = tuple((
            lower_bound,            # sort by           [0]
            count,                  # tie breaker       [1]
            self.cities[0],         # from city         [2]
            self.cities[1:],        # cities to visit   [3]
            [self.cities[0]],       # route             [4]
            initial_matrix,         # the state's matrix[5]
            lower_bound))           # the lowerbound    [6]

        # insert is O(logn) to resort the heap after adding (bubble up)
        heapq.heappush(pq, initial_state)
        while time.time() - start_time < time_allowance and len(pq) > 0:
            # pop is also O(logn) but takes O(1) time to get min, then logn to resort the heap
            state = heapq.heappop(pq)
            # we know we run this k times (number of states)
            if (state[6] < lowest_cost):
                for city in state[3]:
                    count += 1
                    cost_to_city = state[5][state[2]._index][city._index]
                    if (cost_to_city != math.inf):
                        # this is O(n^2) time and space as explained near the function
                        new_matrix, reduction_cost = self.reduce_matrix_for_city(
                            city, state[2], state[5])

                        # formulate new state
                        new_lower_bound = state[6] + \
                            reduction_cost + cost_to_city
                        new_from_city = city
                        # remove a city by index is worst case O(n)
                        new_cities_to_visit = self.remove_city_by_index(
                            state[3], city._index)
                        new_route = state[4] + [city]
                        new_state = tuple((
                            new_lower_bound - 2000 * len(new_route),
                                                    # sort by           [0]
                            count,                  # tie breaker       [1]
                            new_from_city,          # from city         [2]
                            new_cities_to_visit,    # cities to visit   [3]
                            new_route,              # route             [4]
                            new_matrix,             # the state's matrix[5]
                            new_lower_bound))       # the lower bound   [6]

                        # check to see if this is a leaf
                        if (len(new_state[3]) == 0):
                            bssf = TSPSolution(new_state[4])
                            if (bssf.cost < lowest_cost):
                                lowest_cost = bssf.cost
                            total_solutions += 1
                        # check to see if this one is even worth going down
                        else:
                            if (new_state[6] < lowest_cost):
                                # push is again O(logn) since it might have to bubble all the way up
                                heapq.heappush(pq, new_state)
                            else:
                                total_pruned += 1
                            total_states_created += 1
            else:
                # this means another soln came up prior that now beats this soln
                total_pruned += 1
                total_states_created += 1

            # always make sure we're seeing if the queue size got bigger
            max_queue_size = max(max_queue_size, len(pq))

        end_time = time.time()
        results = {}
        results['cost'] = lowest_cost
        results['time'] = end_time - start_time
        results['count'] = total_solutions
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_states_created
        results['pruned'] = total_pruned

        return results

    def remove_city_by_index(self, cities, city_index):
        new_cities = cities[:]
        for i, city in enumerate(new_cities):
            if (city._index == city_index):
                del new_cities[i]
                return new_cities

    # O(n^2) for time because we visit every row and every column and copy every item
    # O(n^2) for space because we have to copy the values over
    def reduce_matrix_for_city(self, to_city, from_city, matrix):
        
        matrix = matrix.copy()
        cost_to_reduce = 0

        # set the row and column referring to these cities to inf
        matrix[from_city._index] = math.inf
        matrix[:, to_city._index] = math.inf
        # set the path back to from_city to inf since we can't go back
        matrix[to_city._index][from_city._index] = math.inf

        # reduce the matrix rows
        for row in range(matrix.shape[0]):
            minimum = np.min(matrix[row])
            if (minimum == math.inf):
                continue
            matrix[row] = matrix[row] - minimum
            cost_to_reduce += minimum

        # reduce the matrix cols
        for col in range(matrix.shape[1]):
            minimum = np.min(matrix[:, col])
            if (minimum == math.inf):
                continue
            matrix[:, col] = matrix[:, col] - minimum
            cost_to_reduce += minimum
        
        return matrix, cost_to_reduce

    # O(n^2) for time since we iterate over all cities twice
    # O(n^2) for space since we store n^2 values for distances
    def init_matrix(self, cities):
        # init the matrix with all distance values
        matrix = np.full((len(cities), len(cities)), fill_value=math.inf)
        for from_index, from_city in enumerate(cities):
            for to_index, to_city in enumerate(cities):
                if (from_index == to_index):
                    continue
                distance = from_city.costTo(to_city)
                matrix[from_index][to_index] = distance
        return matrix

    # O(n^2) for time because we visit every row and every column and copy every item
    # O(n^2) for space because we have to copy the values over
    def reduce_matrix(self, matrix):
        cost_to_reduce = 0

        # reduce rows
        for row in range(matrix.shape[0]):
            minimum = np.min(matrix[row])
            if (minimum == math.inf):
                cost_to_reduce = math.inf
                break
            matrix[row] = matrix[row] - minimum
            cost_to_reduce += minimum

        # reduce cols
        for col in range(matrix.shape[1]):
            minimum = np.min(matrix[:, col])
            if (minimum == math.inf):
                cost_to_reduce = math.inf
                break
            matrix[:, col] = matrix[:, col] - minimum
            cost_to_reduce += minimum

        return matrix, cost_to_reduce

    def print_matrix(self, matrix):
        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                         for row in matrix]))


    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0):
        results = {}
        count = 0
        start_time = time.time()
        greedy_solution = self.greedyBestFind()
        current_route = greedy_solution['soln'].route
        current_cost = greedy_solution['soln'].cost
        temperature = 1000
        best_route = current_route[:]
        best_cost = current_cost

        while temperature > 0 and time.time() - start_time < time_allowance:
            new_route = current_route[:]
            new_cost = math.inf
            # find two random cities
            indexes = random.sample(range(len(current_route)), 2)

            # swap cities
            temp = new_route[indexes[0]]
            new_route[indexes[0]] = new_route[indexes[1]]
            new_route[indexes[1]] = temp
            new_cost = TSPSolution(new_route).cost

            # If the new solution is better or acceptance probability is true, accept it
            if new_cost < current_cost or math.exp((current_cost - new_cost) / temperature) > random.randrange(0, 1):
                if (new_cost > current_cost):
                    print('actually randmoly switched!')
                current_route = new_route
                current_cost = new_cost

            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

            # decrease temperature
            count += 1
            new_temperature = temperature / (1 + math.log10(1 + count))
            temperature = new_temperature

        bssf = TSPSolution(best_route)
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results
        # return best results


# Overview
# ==============
# Set initial temperature

# Set cooling rate

# Create intial random pattern or path of our travelling sales
# ex
# ABCDEFG

# this distance is 100 total

# while temp > 1
# then we choose two cities randomly and swap them
# ex B and G

# we then get
# AGCDEFB
# this distance is 102

# get these distances and run them through the acceptance probability
# if it is acceptable then change the new to the current

# if its distance is less than the best, it becomes the best


# current, new, and best solutions

# cool down

# once temp <= 1
# the "metal" is set and we keep the best solution

# public static double acceptanceProbability(int energy, int newEnergy, double temperature) {
#         // If the new solution is better, accept it
#         if (newEnergy < energy) {
#             return 1.0;
#         }
#         // If the new solution is worse, calculate an acceptance probability
#         return Math.exp((energy - newEnergy) / temperature);
#     }
