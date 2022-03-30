# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
#Informed search Slide 58
    #loop do
        #if fringe is empty then return failure
        #remove front
    #if goal
        #return node
    #for node in available
        #insert into fringe
    from util import Stack
    stack = Stack()
    dfs = []

    #Push the starting point onto Stack
    stack.push((problem.getStartState(), []))

    while stack.isEmpty() == False:
        #Grab the top node in the stack.
        current, path = stack.pop()
        #if the current node isn't in the stack, then add it.
        if current not in dfs:
            dfs.append(current)
        #if current is in stack, then skip
        elif current in dfs:
            continue
        #if we reached the goal, return
        if problem.isGoalState(current) == True:
            return path

        for location, direction, distance in problem.getSuccessors(current):
            #stack.push(self, array of movement)
            pacman_direction = [direction]
            #order must be path and then direction otherwise illegal movements can occur.
            movement = path + pacman_direction
            stack.push((location, movement))


    #python pacman.py -l tinyMaze -p SearchAgent -a fn=depthFirstSearch
    return

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
# Informed search Slide 58
    # loop do
        # if fringe is empty then return failure
        # remove front
    # if goal
        # return node
    # for node in available
        # insert into fringe
    from util import Queue
    queue = Queue()
    bfs = []
    queue.push((problem.getStartState(), []))

    while queue.isEmpty() == False:
        # Grab the top node in the stack.
        current, path = queue.pop()
        if current not in bfs:
            bfs.append(current)
        # if current is in stack, then skip. If we don't add statement, it fails the last test case.
        elif current in bfs:
            continue
        if problem.isGoalState(current) == True:
            return path

        for location, direction, distance in problem.getSuccessors(current):
            #stack.push(self, array of movement)
            pacman_direction = [direction]
            #order must be path and then direction otherwise illegal movements can occur.
            movement = path + pacman_direction
            queue.push((location, movement))


    return

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #Expand the cheapest node first
    #Fringe priority queue -> cumulative cost
    from util import PriorityQueue
    pqueue = PriorityQueue()
    ucs = []
    #Gets starting location, pushes empty array with cost 0 because its the starting (cheapest) node.
    start_state = (problem.getStartState(), [], 0)
    #Give start state a priority of 0.
    pqueue.push((start_state), 0)

    while pqueue.isEmpty() == False:
        # Grab the top node in the stack.
        current, path, currentDistance = pqueue.pop()

        if current not in ucs:
            ucs.append(current)
        # if current is in stack, then skip. If we don't add statement, it fails the last test case.
        elif current in ucs:
            continue
        if problem.isGoalState(current) == True:
            return path

        for location, direction, distance in problem.getSuccessors(current):
            # pqueue.push(position, array of movement, distance/cost)
            pacman_direction = [direction]
            # order must be path and then direction otherwise illegal movements can occur.
            movement = path + pacman_direction
            #distance found
            new_distance = currentDistance + distance
            new_state = (location, movement, new_distance)
            pqueue.push(new_state, new_distance)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #Expand the cheapest node first
    #Fringe priority queue -> cumulative cost
    from util import PriorityQueue
    pqueue = PriorityQueue()
    ucs = []
    #Gets starting location, pushes empty array with cost 0 because its the starting (cheapest) node.
    start_state = (problem.getStartState(), [], 0)
    #Give start state a priority of 0.
    pqueue.push((start_state), 0)

    while pqueue.isEmpty() == False:
        # Grab the top node in the stack.
        current, path, currentDistance = pqueue.pop()
        if current not in ucs:
            ucs.append(current)
        # if current is in stack, then skip. If we don't add statement, it fails the last test case.
        elif current in ucs:
            continue
        if problem.isGoalState(current) == True:
            return path

        for location, direction, distance in problem.getSuccessors(current):
            # pqueue.push(position, array of movement, distance/cost)
            pacman_direction = [direction]
            # order must be path and then direction otherwise illegal movements can occur.
            movement = path + pacman_direction
            #distance found
            new_distance = currentDistance + distance

            new_state = (location, movement, new_distance)
            #-------------------NOTE: This value has to be less than or equal to the actual distance. Otherwise this won't work. ---------------------------------
            pqueue.push(new_state, new_distance+heuristic(location, problem))

    return

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

