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
from functools import partial
from math import inf
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Obtine starea jocului dupa o actiune data
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Obtine noua pozitie a Pacman-ului in starea urmatoare a jocului
        newPos = successorGameState.getPacmanPosition()

        # Obtine starea noilor alimente din noua stare a jocului
        newFood = successorGameState.getFood()

        # Obtine starea noilor fantome din noua stare a jocului
        newGhostStates = successorGameState.getGhostStates()

        # Obtine lista timpilor ramasi pentru care fiecare fantoma va fi inca speriata
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Functie auxiliara pentru a calcula distanta Manhattan intre doua pozitii
        distToPacman = partial(manhattanDistance, newPos)

        # Calculeaza scorul pentru fantome in functie de distanta fata de Pacman si starea lor (speriat sau nu)
        ghostScore = min(
            (
                inf if ghost.scaredTimer > distToPacman(ghost.getPosition()) else
                -inf if distToPacman(ghost.getPosition()) <= 1 else 0
            )
            for ghost in newGhostStates
        )

        # Calculeaza cea mai mica distanta pana la cel mai apropiat aliment
        distToClosestFood = min(
            map(distToPacman, newFood.asList()), default=inf
        )

        # Calculeaza o caracteristica bazata pe cea mai mica distanta pana la cel mai apropiat aliment
        closestFoodFeature = 1.0 / (1.0 + distToClosestFood)

        # Returneaza o valoare numerica care reprezinta evaluarea starii jocului
        # Include scorul actual, scorul pentru fantome si caracteristica pentru cel mai apropiat aliment
        return successorGameState.getScore() + ghostScore + closestFoodFeature


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        # Defineste functia max_value care evalueaza actiunile pentru a maximiza scorul pentru Pacman.
        def max_value(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                # Daca am ajuns la o stare finala sau la adancimea maxima, intoarce scorul evaluat si o actiune vida.
                return self.evaluationFunction(gameState), ""

            actions = gameState.getLegalActions(0)  # Obtine actiunile legale pentru Pacman (agentul 0)
            bestAction = None
            bestScore = float('-inf')

            for action in actions:
                # Pentru fiecare actiune legala, genereaza starea succesoare si calculeaza scorul minim pentru actiunile agentilor adversi.
                successor = gameState.generateSuccessor(0, action)
                score = min_value(successor, 1, depth)[0]  # Apeleaza min_value pentru a obtine scorul minim
                if score > bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction

        # Defineste functia min_value care evalueaza actiunile pentru a minimiza scorul pentru agentii adversi.
        def min_value(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), ""

            actions = gameState.getLegalActions(agentIndex)  # Obtine actiunile legale pentru agentul specificat
            bestAction = None
            bestScore = float('inf')

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # Daca am evaluat toti agentii adversi, calculeaza scorul maxim pentru Pacman
                    score = max_value(successor, depth + 1)[0]
                else:
                    # Altfel, continua cu urmatorul agent advers si calculeaza scorul minim
                    score = min_value(successor, agentIndex + 1, depth)[0]

                if score < bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction

        # Porneste algoritmul Minimax cu max_value pentru a determina actiunea optima pentru Pacman.
        _, result_action = max_value(gameState, 0)
        return result_action

        return 0

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Apelul catre getBestActionAndScore pentru a determina cea mai buna actiune si scorul asociat
        result = self.getBestActionAndScore(gameState, 0, 0, float("-inf"), float("inf"))

        # Returneaza actiunea din rezultat
        return result[0]

    def getBestActionAndScore(self, game_state, index, depth, alpha, beta):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Starile terminale:
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", game_state.getScore()

        # Agent maximizator: Pacman are index = 0
        if index == 0:
            return self.max_value(game_state, index, depth, alpha, beta)

        # Agent minimizator: Fantoma are index > 0
        else:
            return self.min_value(game_state, index, depth, alpha, beta)

    def max_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the max utility action-score for max-agent with alpha-beta pruning
        """
        legalMoves = game_state.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Actualizarea indexului si adancimii succesorului daca este Pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calcularea actiunii si scorului pentru succesorul curent
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Actualizarea max_value si max_action pentru agentul maximizator
            if current_value > max_value:
                max_value = current_value
                max_action = action

            # Actualizarea valorii alpha pentru maximizatorul curent
            alpha = max(alpha, max_value)

            # Taierea Alpha-Beta: Se intoarce max_value deoarece urmatoarele max_value-uri posibile
            # ale maximizatorului pot deveni mai rele pentru valoarea beta a minimizatorului cand se revine in sus
            if max_value > beta:
                return max_action, max_value

        return max_action, max_value

    def min_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the min utility action-score for min-agent with alpha-beta pruning
        """
        legalMoves = game_state.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Actualizarea indexului si adancimii succesorului daca este Pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calcularea actiunii si scorului pentru succesorul curent
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Actualizarea min_value si min_action pentru agentul minimizator
            if current_value < min_value:
                min_value = current_value
                min_action = action

            # Actualizarea valorii beta pentru minimizatorul curent
            beta = min(beta, min_value)

            # Taierea Alpha-Beta: Se intoarce min_value deoarece urmatoarele min_value-uri posibile
            # ale minimizatorului pot deveni mai rele pentru valoarea alpha a maximizatorului cand se revine in sus
            if min_value < alpha:
                return min_action, min_value

        return min_action, min_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth + 1  # Adancimea curenta a cautarii
        agent = self.index  # Indexul agentului curent

        # Actualizeaza adancimea pentru urmatorul agent in functie de starea curenta
        nextDepth = depth - 1 if agent == 0 else depth

        # Verifica daca suntem la nivelul maxim de adancime sau intr-o stare terminala
        if nextDepth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)  # Returneaza valoarea evaluata a starii

        nextAgent = (agent + 1) % gameState.getNumAgents()  # Urmatorul agent in joc
        legalMoves = gameState.getLegalActions(agent)  # Miscarile legale ale agentului curent

        bestVal = float('-inf')  # Initializeaza cea mai buna valoare cu infinit negativ
        bestAction = None  # Initializeaza cea mai buna actiune cu None

        # Parcurge fiecare actiune legala a agentului curent
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agent,
                                                         action)  # Genereaza starea succesoare pentru actiunea curenta
            expVal = self.getActionValue(successorState, nextDepth,
                                         nextAgent)  # Obtine valoarea asteptata pentru actiunea curenta

            # Actualizeaza cea mai buna valoare si actiunea corespunzatoare
            if max(bestVal, expVal) == expVal:
                bestVal = expVal
                bestAction = action

        return bestAction  # Returneaza cea mai buna actiune

    def getActionValue(self, state, depth, agent):
        # Actualizeaza adancimea pentru urmatorul agent in functie de starea curenta
        nextDepth = depth - 1 if agent == 0 else depth

        # Verifica daca suntem la nivelul maxim de adancime sau intr-o stare terminala
        if nextDepth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)  # Returneaza valoarea evaluata a starii

        nextAgent = (agent + 1) % state.getNumAgents()  # Urmatorul agent in joc
        legalMoves = state.getLegalActions(agent)  # Miscarile legale ale agentului curent

        # Daca agentul nu este agentul principal (agentul 0)
        if agent != 0:
            prob = 1.0 / float(len(legalMoves))  # Calculeaza probabilitatea pentru actiunile disponibile
            value = 0.0  # Initializeaza valoarea cu 0

            # Parcurge fiecare actiune legala a agentului curent
            for action in legalMoves:
                successorState = state.generateSuccessor(agent,
                                                         action)  # Genereaza starea succesoare pentru actiunea curenta
                expVal = self.getActionValue(successorState, nextDepth,
                                             nextAgent)  # Obtine valoarea asteptata pentru actiunea curenta
                value += prob * expVal  # Actualizeaza valoarea ponderata

            return value  # Returneaza valoarea totala calculata

        bestVal = float('-inf')  # Initializeaza cea mai buna valoare cu infinit negativ

        # Parcurge fiecare actiune legala a agentului curent
        for action in legalMoves:
            successorState = state.generateSuccessor(agent,
                                                     action)  # Genereaza starea succesoare pentru actiunea curenta
            expVal = self.getActionValue(successorState, nextDepth,
                                         nextAgent)  # Obtine valoarea asteptata pentru actiunea curenta

            # Actualizeaza cea mai buna valoare
            if max(bestVal, expVal) == expVal:
                bestVal = expVal

        return bestVal  # Returneaza cea mai buna valoare


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Obtine pozitia lui Pacman si pozitiile fantomelor
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    # Obtine lista de boabe de mancare si calculeaza diverse contoare
    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    # Obtine scorul jocului
    game_score = currentGameState.getScore()

    # Calculeaza distantele pana la boabele de mancare
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

    # Determina cea mai apropiata boaba de mancare
    if food_count > 0:
        closest_food = min(food_distances)

    # Verifica proximitatea fantomelor si ajusteaza cea mai apropiata boaba de mancare
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        if ghost_distance < 2:
            closest_food = 99999

    # Definirea caracteristicilor si greutatilor corespunzatoare
    features = [1.0 / closest_food, game_score,food_count,capsule_count]

    weights = [10,200,-100,-10]

    # Calculul scorului final de evaluare
    return sum([feature * weight for feature, weight in zip(features, weights)])


# Abbreviation
better = betterEvaluationFunction
