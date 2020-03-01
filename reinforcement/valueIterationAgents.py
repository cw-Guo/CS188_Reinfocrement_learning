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
              mdp.getStates() # get all the states
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
        states = self.mdp.getStates() # get all the possible state
        lifetime = self.iterations
        curr_k = 1 # next ietration
        stack = {}
        while curr_k <= lifetime + 1:
            for key in stack:
                self.values[key] = stack[key]
            for curr_state in states:
                actions = self.mdp.getPossibleActions(curr_state) # available actionss
                if actions == (): # terminal
                    continue
                else:
                    optimal_action_bonus = -100000
                    for action in actions:
                        allPossibleList = self.mdp.getTransitionStatesAndProbs(curr_state, action) # it is a list of (nextState, prob) pairs
                        bonus = 0
                        for nextState, prob in allPossibleList:
                            reward = self.mdp.getReward(curr_state, action, nextState)
                            value_k_1 = self.values[nextState] # if not in the dict, default value = 0 --->refer to util.Counter
                            bonus += prob * (reward + self.discount * value_k_1) # ballmen Equation
                        if bonus > optimal_action_bonus:
                            optimal_action_bonus = bonus
                            optimal_action = action
                    stack[curr_state] = optimal_action_bonus
            curr_k += 1
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
        allPossibleList = self.mdp.getTransitionStatesAndProbs(state, action)
        bonus = 0
        for nextState, prob in allPossibleList:
            reward = self.mdp.getReward(state, action, nextState)
            value_k = self.values[nextState] # if not in the dict, default value = 0 --->refer to util.Counter
            bonus += prob * (reward + self.discount * value_k) # ballmen Equation
        return bonus

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state) # available actions
        if actions == (): # terminal
            return None
        optimal_action_bonus = -100000 # quite small
        optimal_action = None
        for action in actions:
            allPossibleList = self.mdp.getTransitionStatesAndProbs(state, action) # it is a list of (nextState, prob) pairs
            bonus = 0
            for nextState, prob in allPossibleList:
                reward = self.mdp.getReward(state, action, nextState)
                value_k = self.values[nextState] # if not in the dict, default value = 0 --->refer to util.Counter
                bonus += prob * (reward + self.discount * value_k) # ballmen Equation
            if bonus > optimal_action_bonus:
                optimal_action_bonus = bonus
                optimal_action = action
        return optimal_action
        util.raiseNotDefined()

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
              mdp.isTerminal(state)
              mdp.getReward(state, action, next_state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
    def runValueIteration(self):
        states = self.mdp.getStates()
        numstate = len(states)
        #each iterations only update one state's value
        # we have to re- implement our values pair : shouldn't involve  k in the pair
        
        curr_k = 1
        stack ={}

        for i  in range(self.iterations + 1):
            curr_state = states[i % numstate] # cycle
            actions = self.mdp.getPossibleActions(curr_state)
            for key in stack:
                self.values[key] = stack[key]

            if actions == ():
                continue
            else:
                optimal_bounus = - 1000000
                for action in actions:
                    allPossibleList = self.mdp.getTransitionStatesAndProbs(curr_state, action)
                    bonus = 0
                    for nextState, prob in allPossibleList:
                        reward = self.mdp.getReward(curr_state, action, nextState)
                        value_k_1 = self.values[nextState]
                        bonus += prob*(reward + self.discount * value_k_1)
                    if bonus > optimal_bounus:
                        optimal_bounus = bonus
                stack[curr_state] = optimal_bounus


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
        def getPredecessors(curr_state): # get the predecessors
            states = self.mdp.getStates()
            predecessors = set()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if actions == ():
                    continue    
                else:
                    for action in actions:
                        allPossibleList = self.mdp.getTransitionStatesAndProbs(state, action) # it is a list of (nextState, prob) pairs
                        for pair in allPossibleList:
                            if (pair[0] == curr_state) and (pair[1] != 0):
                                predecessors.add(state)
            return predecessors
            
        def getmaxQvalue(curr_state):
            actions = self.mdp.getPossibleActions(curr_state)
            maxQ = -1000000
            if actions ==(): # terminate
                return None
            for action in actions:
                temp = self.computeQValueFromValues(curr_state, action)
                if temp > maxQ:
                    maxQ = temp
            return maxQ

        queue = util.PriorityQueue() #initialize 
        states = self.mdp.getStates()
        for state in states:
            if state =='TERMINAL_STATE':
                continue
            maxQ = getmaxQvalue(state)
            diff = abs(self.values[state] - maxQ)
            queue.push(state, -diff) # we wanna higher  diff 
        
        lifetime = self.iterations
        for i in range(lifetime):
            if queue.isEmpty():
                return 
            currState = queue.pop()
            if currState != 'TERMINAL_STATE':
                self.values[currState]  = getmaxQvalue(currState)
                for preState in getPredecessors(currState):
                    maxpreQ = getmaxQvalue(preState) 
                    diff = abs(self.values[preState] - maxpreQ)
                    if diff > self.theta:
                        queue.update(preState, -diff)
            