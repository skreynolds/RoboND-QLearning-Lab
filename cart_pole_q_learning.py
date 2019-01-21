#!/usr/bin/env python3

#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()

# The typical imports
import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

# visualization helpers
#from IPython import display
import matplotlib
import matplotlib.pyplot as plt

# Set logger level
import logging
logging.basicConfig(level=logging.ERROR)


# copied from gym examples - a model for our agent
class RandAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done=None, mode=None):
        return self.action_space.sample()
    
    def init_episode(self, observation):
        # provided for compatibility with general learner
        return self.action_space.sample()


# This learner provides a general interface for agents and environments 
# optionally visualize within a Jupyter Notebook when visualize_plt is True
def learner(agent=None, env_id='CartPole-v0', episodes=100, max_length = 100, init_reward=0, 
            ignore_done=False, visualize_plt=True, mode=None):
    # load the environment 
    env = gym.make(env_id)
    # set the agent to random if none provided
    if agent is None:
        agent = RandAgent(env.action_space)

    # each episode runs until it is observed as finished, or exceeds max_length in time steps
    episode_count = episodes
    done = False
    n_steps = np.zeros((episode_count,))

    # run the episodes - use tqdm to track in the notebook
    for i in tqdm(range(episode_count), disable=visualize_plt):
        
        # Initialize environment for each episode
        ob = env.reset()  
        reward = init_reward
        if visualize_plt:
            img = plt.imshow(env.render(mode='rgb_array')) # only call this once, only for jupyter
        
        # initialize the agent
        agent.init_episode(ob)
        n_steps[i]=max_length
        
        # run the steps in each epsisode
        for t in range(max_length):
            # render the environment
            if visualize_plt:
                img.set_data(env.render(mode='rgb_array')) # just update the data
                plt.axis('off')
                #display.display(plt.gcf())
                #display.clear_output(wait=True)
            #else:
            #    env.render()
            
            # get agent's action
            action = agent.act(ob, reward, mode=mode)
            # take the action and get reward and updated observation
            ob, reward, done, _ = env.step(action)
            
            # terminate the steps if the problem is done
            if done and not ignore_done:
                n_steps[i] = t
                break
            if done and ignore_done and env_id=='MountainCar-v0' and ob[0]>= 0.5:
                # special case MountainCar
                # if have achieved the goal, then quit but otherwise keep going
                print("Episode {} done at step {}".format(i,t))
                print("Observations {}, Reward {}".format(ob, reward))
                n_steps[i] = t
                break
    env.close()
    return n_steps  # stats


class QLearningAgent:
    def __init__(self, 
                 learning_rate = 0.2, discount_factor = 1.0,
                 exploration_rate = 0.5, exploration_decay_rate = 0.99,
                 n_bins = 9, n_actions = 2, splits=None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate # initial epsilon
        self.exploration_decay_rate = exploration_decay_rate # decay factor for epsilon
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.splits = splits
        self.state = None
        self.action = None
        
        if self.splits is None: #CartPole default
            self.splits = [
                # Position
                np.linspace(-2.4, 2.4, self.n_bins)[1:-1],
                # Velocity
                np.linspace(-3.5, 3.5, self.n_bins)[1:-1],
                # Angle.
                np.linspace(-0.5, 0.5, self.n_bins)[1:-1],
                # Tip velocity
                np.linspace(-2.0, 2.0, self.n_bins)[1:-1]
            ]
        
        # Create Q-Table
        num_states = (self.n_bins+1) ** len(self.splits)
        self.q_table = np.zeros(shape=(num_states, self.n_actions))

    # Turn the observation into integer state
    def set_state(self, observation):
        state = 0
        for i, column in enumerate(observation):
            state +=  np.digitize(x=column, bins=self.splits[i]) * ((self.n_bins + 1) ** i)
        return state

    # Initialize for each episode
    def init_episode(self, observation):
        # Gradually decrease exploration rate
        self.exploration_rate *= self.exploration_decay_rate

        # Decide initial action
        self.state = self.set_state(observation)
        return np.argmax(self.q_table[self.state])

    # Select action and update
    def act(self, observation, reward=None, done=None, mode='train'):
        next_state = self.set_state(observation)
        
        if mode == 'test':
            # Test mode 
            next_action = np.argmax(self.q_table[next_state])
        else:
            # Train mode by default
            # Train by updating Q-Table based on current reward and 'last' action.
            self.q_table[self.state, self.action] += self.learning_rate * \
                (reward + self.discount_factor * max(self.q_table[next_state, :]) - self.q_table[self.state, self.action])
            # Exploration or exploitation
            do_exploration = (1 - self.exploration_rate) < np.random.uniform(0, 1)
            if do_exploration:
                #  Exploration
                next_action = np.random.randint(0, self.n_actions)
            else:
                # Exploitation
                next_action = np.argmax(self.q_table[next_state])

        self.state = next_state
        self.action = next_action
        return next_action

if __name__ == '__main__':
    
    #############################################
    # Instantiate the agent
    q_agent = QLearningAgent()

    num_episodes = 50
    max_length = 200
    initial_reward = 1
    #############################################

    #############################################
    # train the agent - execute this cell as many times as you wish
    # set the visualize_plt flag to True to see the cart in the notebook.  
    # note that this will run slower if visualized
    steps = learner(agent=q_agent, episodes=num_episodes, max_length = max_length, 
                        init_reward=initial_reward, visualize_plt=False)

    print("Minimum step count in {} episodes: {}".format(num_episodes, np.min(steps)))
    print("Average step count in {} episodes: {}".format(num_episodes, np.mean(steps)))
    print("Maximum step count in {} episodes: {}".format(num_episodes, np.max(steps)))
    print("Q-table size: ", q_agent.q_table.size)
    print("Q-table nonzero count: ", np.count_nonzero(q_agent.q_table))
    #############################################

    #############################################
    # Testing the agent - run this smaller sampling after the agent is achieving success and NOT exploring
    num_episodes = 5
    max_length = 200
    initial_reward = 1
    steps = learner(agent=q_agent, episodes=num_episodes, max_length = max_length,
                        init_reward=initial_reward, mode='test') # set mode to 'test' to avoid exploration

    print("Minimum step count in {} episodes: {}".format(num_episodes, np.min(steps)))
    print("Average step count in {} episodes: {}".format(num_episodes, np.mean(steps)))
    print("Maximum step count in {} episodes: {}".format(num_episodes, np.max(steps)))
    print("Q-table size: ", q_agent.q_table.size)
    print("Q-table nonzero count: ", np.count_nonzero(q_agent.q_table))
    #############################################