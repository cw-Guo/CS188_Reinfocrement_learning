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
        while curr_k <= lifetime:
            for curr_state in states:
                actions = self.mdp.getPossibleActions(curr_state) # available actionss
                if actions == (): # terminal
                    new_pair = tuple((curr_state, curr_k))
                    self.values[new_pair] = self.values[curr_state,curr_k-1]
                else:
                    optimal_action_bonus = -100000
                    for action in actions:
                        allPossibleList = self.mdp.getTransitionStatesAndProbs(curr_state, action) # it is a list of (nextState, prob) pairs
                        bonus = 0
                        for nextState, prob in allPossibleList:
                            reward = self.mdp.getReward(curr_state, action, nextState)
                            value_k_1 = self.values[(nextState, curr_k -1)] # if not in the dict, default value = 0 --->refer to util.Counter
                            bonus += prob * (reward + self.discount * value_k_1) # ballmen Equation
                        if bonus > optimal_action_bonus:
                            optimal_action_bonus = bonus
                            optimal_action = action
                    new_pair = tuple((curr_state, curr_k))
                    self.values[new_pair] = optimal_action_bonus
            curr_k += 1
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[(state,self.iterations)]


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
            value_k = self.values[(nextState, self.iterations)] # if not in the dict, default value = 0 --->refer to util.Counter
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
                value_k = self.values[(nextState, self.iterations)] # if not in the dict, default value = 0 --->refer to util.Counter
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
        curr_k = 1
        for i  in range(self.iterations):
            curr_state = states[i % numstate]
            actions = self.mdp.getPossibleActions(curr_state)
            if actions == ():
                continue
            else:
                optimal_bounus = - 1000000
                for action in actions:
                    allPossibleList = self.mdp.getTransitionStatesAndProbs(curr_state, action)
                    bonus = 0
                    for nextState, prob in allPossibleList:
                        reward = self.mdp.getReward(curr_state, action, nextState)
                        value_k_1 = self.values[(nextState,i //numstate + 1)]
                        bonus += prob*(reward + self.discount * value_k_1)
                    if bonus > optimal_bounus:
                        optimal_bounus = bonus
                new_pair = tuple((curr_state, i+1))
            self.values[(curr_state,i + 1 - numstate)]
            self.values[new_pair] = optimal_bounus
        print(self.values)

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

