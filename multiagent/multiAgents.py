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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostDistance = 1000
        foodDistance = 1000

        foodList = newFood.asList()
        for i in newGhostStates:
            if max(newScaredTimes) == 0:
                break
            tempDistance = manhattanDistance(i.getPosition(), newPos)
            ghostDistance = min(ghostDistance, tempDistance)


        for i in foodList:
            tempDistance = manhattanDistance(i, newPos)
            foodDistance = min(tempDistance, foodDistance)

        return childGameState.getScore() - 1/ (ghostDistance + 1) + 1/ (foodDistance + 1)

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    from operator import itemgetter

    def minimax(self, gameState, agentNum, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentNum == 0:
            max_value = -1e9
            for action in gameState.getLegalActions(0):
                max_value = max(self.minimax(gameState.getNextState(agentNum, action), agentNum + 1, depth), max_value)

            return max_value

        else:
            if agentNum == gameState.getNumAgents() - 1:
                min_value = 1e9
                for action in gameState.getLegalActions(agentNum):
                    min_value = min(self.minimax(gameState.getNextState(agentNum, action), 0, depth + 1), min_value)

                return min_value
            else:
                min_value = 1e9
                for action in gameState.getLegalActions(agentNum):
                    min_value = min(self.minimax(gameState.getNextState(agentNum, action), agentNum + 1, depth), min_value)

                return min_value


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        evaluations = [(action, self.minimax(gameState.getNextState(0, action), agentNum=1, depth=0))
                       for action in gameState.getLegalActions(0)]
        sorted_eval = sorted(evaluations, key=lambda x: x[1], reverse=True)

        return sorted_eval[0][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax_ab(self, gameState, agentNum, depth, a, b):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentNum == 0:
            max_value = -1e9
            for action in gameState.getLegalActions(0):
                max_value = max(self.minimax_ab(gameState.getNextState(agentNum, action), agentNum + 1, depth, a, b), max_value)
                if max_value > b:
                    return max_value
                a = max(a, max_value)

            return max_value

        else:
            if agentNum == gameState.getNumAgents() - 1:
                min_value = 1e9
                for action in gameState.getLegalActions(agentNum):
                    min_value = min(self.minimax_ab(gameState.getNextState(agentNum, action), 0, depth + 1, a, b), min_value)
                    if min_value < a:
                        return min_value
                    b = min(b, min_value)
                return min_value
            else:
                min_value = 1e9
                for action in gameState.getLegalActions(agentNum):
                    min_value = min(self.minimax_ab(gameState.getNextState(agentNum, action), agentNum + 1, depth, a, b), min_value)
                    if min_value < a:
                        return min_value
                    b = min(b, min_value)

                return min_value

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        a = -1e9
        b = 1e9
        max_value = -1e9

        evaluations = []
        for action in gameState.getLegalActions(0):
            max_value = max(self.minimax_ab(gameState.getNextState(0, action), 1, 0, a, b),
                            max_value)
            if max_value > b:
                return max_value
            a = max(a, max_value)
            evaluations.append((action, max_value))

        sorted_eval = sorted(evaluations, key=lambda x: x[1], reverse=True)

        return sorted_eval[0][0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, agentNum, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentNum == 0:
            max_value = max([self.expectimax(gameState.getNextState(agentNum, action), agentNum + 1, depth)
                             for action in gameState.getLegalActions(0)])
            #print('max_value: ', max_value)
            return max_value

        else:
            counter = 0
            expect_value = 0

            if agentNum == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentNum):
                    expect_value += self.expectimax(gameState.getNextState(agentNum, action), 0, depth + 1)
                    counter += 1
                #print('expect_value: ', expect_value, ' counter: ', counter)
                return expect_value / counter

            else:
                for action in gameState.getLegalActions(agentNum):
                    expect_value += self.expectimax(gameState.getNextState(agentNum, action), agentNum + 1, depth)
                    counter += 1
                #print('expect_value: ', expect_value, ' counter: ', counter)
                return expect_value / counter

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        evaluations = [(action, self.expectimax(gameState.getNextState(0, action), agentNum=1, depth=0))
                      for action in gameState.getLegalActions(0)]
        #print(evaluations)
        sorted_eval = sorted(evaluations, key=lambda x: x[1], reverse=True)

        return sorted_eval[0][0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
