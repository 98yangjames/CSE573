# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        new_vals = []
        for i in range(self.iterations):
            new_vals = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    #if its not the terminal, get the action and q value and put it into the array of vals.
                    action = self.getAction(state)
                    qvalue = self.computeQValueFromValues(state, action)
                    new_vals[state] = qvalue
                else: #if it is the terminal state, skip
                    continue
            self.values = new_vals

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        #Q = Sum(T(s,a,s')[R(s,a,s') + gamma*Max(Q(s',a'))

        for newState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, newState)
            gamma = self.discount
            q_newState = self.values[newState]
            q = q + probability * (reward + gamma * q_newState)
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        vals = util.Counter()
        possible_actions = self.mdp.getPossibleActions(state)
        for action in possible_actions:
            vals[action] = self.computeQValueFromValues(state, action)

        policy = vals.argMax()
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        getStates = self.mdp.getStates()
        for i in range(self.iterations):
            index = i % len(getStates)
            state = getStates[index]
            terminal_node = self.mdp.isTerminal(state)
            if terminal_node:
                continue
            else:
                action = self.getAction(state)
                q = self.getQValue(state, action)
                self.values[state] = q


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # compute predecessors of all states
        predecessors = {}
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            else:
                for action in self.mdp.getPossibleActions(s):
                    for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                        if next in predecessors:
                            predecessors[next].add(s)
                        else:
                            predecessors[next] = {s}

        #Initialize an empty priority queue
        pq = util.PriorityQueue()


        #for each non-terminal state s
        for s in self.mdp.getStates():
            if(self.mdp.isTerminal(s) == False):
                #find the absolute value of the diff between current s in self.values and highest q-value across all possible actions from s.
                current_value = self.values[s]
                values = []
                for action in self.mdp.getPossibleActions(s):
                    potentialQ = self.getQValue(s, action)
                    values.append(potentialQ)

                diff = abs(current_value - max(values))
                print(diff)
                pq.update(s, -diff)

        #for iteration in 0,1,2,...self.,iterations -1)
        for i in range(self.iterations):
            #if pq is empty then terminate
            if pq.isEmpty():
                break

            #pop a state s off the pq
            thisState = pq.pop()
            #update s value if not terminal state in self.values
            if (self.mdp.isTerminal(thisState) == False):
                #for each predecessor p of s
                values = []
                for action in self.mdp.getPossibleActions(thisState):
                    q = self.computeQValueFromValues(thisState, action)
                    values.append(q)
                self.values[thisState] = max(values)

        #     #find absolute value of the difference between current value of p in self.values and highest q

            for pred in predecessors[thisState]:
                if (self.mdp.isTerminal(pred) == False):
                    #same thing find absolute value of predecessor in self.values and highest q.
                    values = []
                    for action in self.mdp.getPossibleActions(pred):
                        q = self.computeQValueFromValues(pred, action)
                        values.append(q)
                    diff = abs(max(values) - self.values[pred])
                    #if diff > theta, push into priority queue with priority -diff.
                    if(diff > self.theta):
                        pq.update(pred, -diff)
