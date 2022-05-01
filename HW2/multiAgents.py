# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        eval = 999999999
        foodDistance = []
        currentFood = currentGameState.getFood()
        "*** YOUR CODE HERE ***"
        if action == "Stop":
            return -999999999

        for food in currentFood.asList():
            distance = util.manhattanDistance(food, newPos)
            foodDistance.append(distance)
            eval = min(distance, eval)

        eval = -eval

        for state in newGhostStates:
            if state.scaredTimer != 0:
                continue
            else:
                if newPos == state.getPosition():
                    return -999999999

        return eval


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #start at the first node and keep going down.
        maximum = -999999999
        action = gameState.getLegalActions(0)[0]
        for agentAction in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, agentAction)
            val = self.minimax(1, 0, successor)
            if val > maximum:
                maximum = val
                action = agentAction
            elif maximum == -999999999:
                action = agentAction
                maximum = val
        return action

    def minimax(self, agent, depth, gameState):
        val = 0
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            ag = 1
            val = max(self.minimax(ag, depth, gameState.generateSuccessor(agent, newState)) for newState in
                      gameState.getLegalActions(agent))
            return val
        elif agent != 0:
            nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth = depth + 1
            val = min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in
                      gameState.getLegalActions(agent))
            return val
        else:
            return -1


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        utility = -99999999
        action = gameState.getLegalActions(0)
        alpha = -9999999999
        beta = 9999999999
        legal_actions = gameState.getLegalActions(0)
        for state in legal_actions:
            successor = gameState.generateSuccessor(0, state)
            if utility > beta:
                return utility
            if self.alphabetaprune(1, 0, successor, alpha, beta) > utility:
                utility = self.alphabetaprune(1, 0, successor, alpha, beta)
                action = state

            alpha = max(alpha, utility)

        return action

    def alphabetaprune(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        # ---------------MAX VALUE -------------------------
        # initialize v = -inf
        # for each successor of state:
        # v = max(v, value of successor (alpha,beta))
        # if v > beta return v
        # alpha = max(alpha, v)
        # return v
        if agent == 0:
            v = -9999999
            for state in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, state)
                abprune = self.alphabetaprune(1, depth, successor, alpha, beta)
                v = max(v, abprune)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        # --------------MIN VALUE ----------------------------
        # initialize v = inf
        # for each successor of state:
        # v = min(v, value of successor (alpha,beta))
        # if v < alpha return v
        # beta = max(beta, v)
        # return v
        elif agent != 0:
            v = 99999999

            next = agent + 1
            if gameState.getNumAgents() == next:
                next = 0
                depth = depth + 1

            for state in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, state)
                abprune = self.alphabetaprune(next, depth, successor, alpha, beta)
                v = min(v, abprune)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        else:
            return -1


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        maximum = -99999999
        action = gameState.getLegalActions(0)
        for state in action:
            successor = gameState.generateSuccessor(0, state)
            utility = self.expectimax(1, 0, successor)
            if utility > maximum:
                maximum = utility
                action = state
            if maximum == -99999999:
                action = state
                maximum = utility

        return action

    def expectimax(self, agent, depth, gameState):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        #-------Grab the max value ----------
        if agent == 0:
            ag = 1
            vals = []
            for newState in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, newState)
                vals.append(self.expectimax(ag, depth, successor))
            return max(vals)
        #-------Grab the expectimax value--------
        else:
            next = agent + 1
            vals = []
            if gameState.getNumAgents() == next:
                next = 0
            if next == 0:
                depth = depth + 1
            for state in gameState.getLegalActions(agent):
                n = float(len(gameState.getLegalActions(agent)))
                vals.append(self.expectimax(next, depth, gameState.generateSuccessor(agent, state)) / n)

            expectimaxVals = sum(vals)

            return expectimaxVals


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    # Started by grabbing the minimum distance of the food, then the minimum distance of the ghosts
    # then checking if the game state is a win or a loss, and how many capsules are left on the map
    # and then adding them together to get the evaluation score.
    """
    "*** YOUR CODE HERE ***"


    minFood = 999999999
    for food in currentGameState.getFood().asList():
        minFood = min(999999999, util.manhattanDistance(currentGameState.getPacmanPosition(), food))

    Ghost = 999999999
    for ghost in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(currentGameState.getPacmanPosition(), ghost)
        if(distance < 3):
            return -999999999
    gameScore = 0
    if currentGameState.isLose():
        gameScore = -100000
    if currentGameState.isWin():
        gameScore = 100000

    return currentGameState.getScore() + (1/minFood) - (1/Ghost) + gameScore - len(currentGameState.getCapsules())


# Abbreviation
better = betterEvaluationFunction
