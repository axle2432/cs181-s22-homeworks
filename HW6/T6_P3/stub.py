# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent learns the optimal policy using Q-learning.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

        # Number of times the learner took action a from state s
        self.N = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

        # Q-learning parameters
        self.alpha = 0.25
        self.gamma = 0.9
        self.epsilon = 0.001

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """
        current_state = self.discretize_state(state)

        '''
        # Perform Q-learning update using current state and last state
        if self.last_state is not None:
            td_error = (self.last_reward + self.gamma *
                        np.max(self.Q[:, current_state[0], current_state[1]]) -
                        self.Q[self.last_action, self.last_state[0], self.last_state[1]])
            self.Q[self.last_action, self.last_state[0], self.last_state[1]] += self.alpha * td_error
        '''

        # Perform Q-learning update using decaying alpha
        if self.last_state is not None:
            N_tsa = self.N[self.last_action, self.last_state[0], self.last_state[1]]
            alpha_t = 1 / N_tsa
            td_error = (self.last_reward + self.gamma *
                        np.max(self.Q[:, current_state[0], current_state[1]]) -
                        self.Q[self.last_action, self.last_state[0], self.last_state[1]])
            self.Q[self.last_action, self.last_state[0], self.last_state[1]] += alpha_t * td_error

        '''
        # Choose next action using an epsilon-greedy policy
        if npr.rand() > self.epsilon:
            new_action = np.argmax(self.Q[:, current_state[0], current_state[1]])
        else:
            new_action = int(npr.rand() < 0.5)
        '''

        # Choose next action using decaying epsilon-greedy policy
        N_ts = np.sum(self.N[:, current_state[0], current_state[1]])
        epsilon_t = 1 / N_ts if N_ts > 0 else 1
        if npr.rand() > epsilon_t:
            new_action = np.argmax(self.Q[:, current_state[0], current_state[1]])
        else:
            new_action = int(npr.rand() < 0.5)

        self.N[new_action, current_state[0], current_state[1]] += 1
        self.last_action = new_action
        self.last_state = current_state

        return new_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)

    # Save history.
    np.save('hist', np.array(hist))
