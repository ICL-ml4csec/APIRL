#from bs4 import BeautifulSoup
#import difflib
#import re
from dqn.dqn import DQN
import torch
import os
import numpy as np
import time
# The Agent class allows the agent to interact with the environment.

class Agent:

    # The class initialisation function.
    def __init__(self, state_space, action_space, epsilon = 1.0, gamma=0.9,
                 lr=0.0001, model='dqn', batch_size=32, hidden=None):
        # set the number of steps after which to update the target network
        # set epsilon
        self.epsilon = epsilon
        # set gamma
        self.gamma = gamma
        # set reward values

        # initalise the current state
        self.state = None

        self.num_actions = action_space

        # initalise reward for episodes
        self.total_reward = None


        self.model_type = 'dqn'
        # initalise the Q-Network
        self.dqn = DQN(gamma=self.gamma, lr=lr, batch_size=batch_size, input_dimension=state_space, output_dimension=action_space)
        


    def save_model(self):
        path = os.path.abspath(os.getcwd())
        if not os.path.exists(path + '/saved_models'):
            os.mkdir(path +'/saved_models')
        save_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.exists(path +'/saved_models/' + self.model_type + "_" + save_time):
            os.mkdir(path +'/saved_models/' + self.model_type + "_" + save_time)
        model_to_save = {'dqn_q_net_state_dict': self.dqn.q_network.state_dict(),
                         'dqn_target_state_dict': self.dqn.target_network.state_dict(),
                         'opt_state_dict':self.dqn.optimiser.state_dict()}

        torch.save(model_to_save, path + '/saved_models/' + self.model_type + "_" + save_time + '/dqn.pt')
        return 'saved_models/' + self.model_type + "_" + save_time

    def load_model(self, relative_path='./saved_models/dqn/dqn.pt'):
        try:
            if self.model_type == 'dqn':
                path = os.path.abspath(os.getcwd())
                check_point = torch.load(relative_path)
                self.dqn.q_network.load_state_dict(check_point['dqn_q_net_state_dict'])
                self.dqn.target_network.load_state_dict(check_point['dqn_target_state_dict'])
                self.dqn.optimiser.load_state_dict(check_point['opt_state_dict'])
                self.dqn.q_network.train()
                self.dqn.target_network.train()
            else:
                print('Loading model type: ' + self.model_type + ' is not supported')
        except:
             print(f'Error loading {relative_path}\nIs the path correct?\nExiting...')
             exit(-42)


    def reset_hiddens(self):
        self.dqn.reset_hidden()

    def get_hidden(self, x):
        return self.dqn.compute_hiddens(x)

    def update_network(self):
        self.dqn.update_target_network()

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.999
        else:
            self.epsilon = self.epsilon
        return

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(range(0, self.num_actions))
            self.dqn.predict_q_values(state)
        else:
            if self.model_type == 'state-action':
                state, action_representations = state
                state_q_values = []
                for action_rep in action_representations:
                    dqn_in = np.append(state, action_rep)
                    q = self.dqn.predict_q_values(dqn_in)
                    state_q_values.append(q)
                action = np.argmax(state_q_values)
                if state_q_values.count(state_q_values[action]) > 0:
                    action = np.random.choice(range(0, self.num_actions))
            else:
                state_q_values = self.dqn.predict_q_values(state)
                action = np.argmax(state_q_values)
        return action


    def get_action_shap(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.random_integers(low=0, high=self.num_actions, size=(1, state.shape[0]))
        else:
            new_state = []
            for i in range(state.shape[0]):
                new_state.append(np.append(state[i, :4], self.auto_features[int(state[i, -1])], axis=-1))
            state = np.array(new_state)
            state_q_values = self.dqn.predict_q_values(state)
            action = np.argmax(state_q_values, axis=-1)
        return action

