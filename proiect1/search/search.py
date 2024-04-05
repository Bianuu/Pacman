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

# def depthFirstSearch(problem):
#     """
#     Search the deepest nodes in the search tree first.
#
#     Your search algorithm needs to return a list of actions that reaches the
#     goal. Make sure to implement a graph search algorithm.
#
#     To get started, you might want to try some of these simple commands to
#     understand the search problem that is being passed in:
#     """
#
#     visited = set()
#     actions = DFSutil(problem, problem.getStartState(), visited)
#     return actions
#     "*** YOUR CODE HERE ***"

class Node():
    def __init__(self, parent, state, action, cost):
        self.parent = parent
        self.state = state
        self. action = action
        self.cost = cost

    def __eq__(self, other):
         return self.state == other.state

def depthFirstSearch(problem: SearchProblem):
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

    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    ns, a, _ = problem.getSuccessors(problem.getStartState())[0]
    ns2, a2, _ = ns.getSuccessors(problem.getStartState())[0]
    ns23, a3, _ = problem.getSuccessors(problem.getStartState())[1]

    return [a]
    return [a2]
    return [a3]

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    "sa miscam o pozitie/2 pozitii .. Sud=0 sau Vest=1"
    (coord,directie,_) = problem.getSuccessors(problem.getStartState())[1]
    (coord1, directie1, _) = problem.getSuccessors(problem.getStartState())[1]
    return [directie,directie1]
    """
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()  # Se creeaza o stiva pentru nodurile de explorat
    frontier.push(Node(None, problem.getStartState(), None, 0))

    noduriExplorare = []

    while not frontier.isEmpty():
        currentNode = frontier.pop()  # Se extrage nodul curent

        if problem.isGoalState(currentNode.state):  # Verifica daca nodul curent este starea tinta
            path = []

            # Reconstruieste calea de la nodul de start la nodul tinta
            while currentNode.parent:
                path = path + [currentNode.action]
                currentNode = currentNode.parent

            path.reverse()  # Inverseaza calea pentru a o afisa in ordinea corecta
            return path  # Returneaza calea gasita catre starea tinta
        else:
            noduriExplorare.append(currentNode.state)  # Adauga starea curenta

            # Exploreaza succesorii starii curente
            for ns, a, _ in problem.getSuccessors(currentNode.state):
                if ns not in noduriExplorare:  # Verifica daca succesorul nu a fost deja explorat
                    frontier.push(Node(currentNode, ns, a, 0))  # Adauga succesorul in stiva pentru explorare ulterioara

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    frontier = util.Queue()  # Coada utilizata pentru a stoca nodurile ce urmeaza sa fie explorate
    frontier.push(Node(None, problem.getStartState(), None, 0))  # Adaugam nodul de start in coada cu costul 0
    noduriExplorare = []  # Lista pentru a urmari nodurile deja explorate

    while not frontier.isEmpty():  # Cat timp mai avem noduri de explorat in coada
        currentNode = frontier.pop()  # Extragem nodul curent din coada pentru a-l explora

        if problem.isGoalState(currentNode.state):  # Verificam daca nodul curent este nodul tinta
            path = []  # Initializam lista pentru a retine drumul catre nodul tinta

            # Reconstruim drumul de la nodul curent pana la nodul de start
            while currentNode.parent:
                path = path + [currentNode.action]  # Adaugam actiunea din nodul curent la drum
                currentNode = currentNode.parent  # Ne deplasam la nodul parinte

            path.reverse()  # Inversam lista pentru a obtine drumul de la nodul de start la nodul tinta
            return path  # Returnam drumul gasit catre nodul tinta
        else:
            noduriExplorare.append(currentNode.state)  # Adaugam starea nodului curent in lista de noduri explorate

            # Parcurgem succesoriile nodului curent si adaugam in coada nodurile neexplorate
            for ns, a, _ in problem.getSuccessors(currentNode.state):
                nm = Node(currentNode, ns, a, 0)  # Cream un nod pentru succesorul curent
                if (ns not in noduriExplorare) and (nm not in frontier.list):  # Verificam daca succesorul este neexplorat si nu este deja in coada
                    frontier.push(Node(currentNode, ns, a, 0))  # Adaugam succesorul in coada pentru a fi explorat ulterior

    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    noduriExplorare = []

    if problem.isGoalState(problem.getStartState()):  # Verifica daca starea initiala este deja starea finala
        return []  # Returneaza o lista goala daca am inceputul si finalul sunt identice

    frontierPriorityQ = util.PriorityQueue()  # Coada de prioritati pentru a explora nodurile
    frontierPriorityQ.push((problem.getStartState(), [], 0),0)  # Adauga starea initiala cu costul 0 in coada de prioritati

    while not frontierPriorityQ.isEmpty():  # Cat timp coada de prioritati nu este goala

        currentNode, directii, costmin = frontierPriorityQ.pop()  # Extrage urmatorul nod din coada

        if currentNode not in noduriExplorare:  # Verifica daca nodul curent nu a fost explorat inca
            noduriExplorare.append(currentNode)  # Adauga nodul curent in lista de noduri explorate

            if problem.isGoalState(currentNode):  # Verifica daca nodul curent este nodul final
                return directii  # Returneaza directiile pentru a ajunge la nodul final

            # Parcurge succesorii nodului curent
            for ns, a, next_cost in problem.getSuccessors(currentNode):
                # Adauga in coada informatiile despre succesor si costul estimat pana la acesta
                frontierPriorityQ.push((ns, directii + [a], costmin + next_cost), costmin + next_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    noduriExplorare = []

    # Verificam daca starea initiala este si o stare finala; daca da, returnam o lista goala, deoarece nu este nevoie de explorare suplimentara
    if problem.isGoalState(problem.getStartState()):
        return []

    frontierPriorityQ = util.PriorityQueue()  # Coada de prioritati pentru a gestiona starile in functie de costul estimat
    frontierPriorityQ.push((problem.getStartState(), [], 0),0)  # Adaugam starea initiala in coada de prioritati cu costul 0

    while not frontierPriorityQ.isEmpty():  # Iteram cat timp exista elemente in coada de prioritati
        currentNode, directii, costt = frontierPriorityQ.pop()  # Extragem nodul urmator si informatiile asociate

        if currentNode not in noduriExplorare:  # Verificam daca nodul curent nu a fost deja explorat
            noduriExplorare.append(currentNode)  # Adaugam nodul curent in lista de noduri explorate

            if problem.isGoalState(currentNode):  # Verificam daca nodul curent este o stare finala
                return directii  # Daca da, returnam directiile pana la acea stare, reprezentand solutia

            # Parcurgem succesorii nodului curent
            for ns, a, cost in problem.getSuccessors(currentNode):
                # Adaugam succesorii in coada de prioritati, impreuna cu directiile si costul asociat actualizat
                frontierPriorityQ.push((ns, directii + [a], costt + cost), costt + cost + heuristic(ns, problem))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch