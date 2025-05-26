import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

import gin
import gym
from minigrid_basics.envs import mon_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.custom_wrappers import tabular_wrapper, mdp_wrapper
from minigrid_basics.custom_wrappers.coloring_wrapper import ColoringWrapper

class CoveringEigenoptions:
    def __init__(self, env_name, gamma_sr=0.9, gamma_o=0.9, eta_sr=0.1, eta_o=0.1, p_option=0.05, n_steps=1000, n_iter=10):
        pre_env = gym.make(env_name, agent_pos=(1, 1))
        pre_env = RGBImgObsWrapper(pre_env)
        pre_env = mdp_wrapper.MDPWrapper(pre_env)
        self.env = ColoringWrapper(pre_env, tile_size=32)
        
        self.gamma_sr = gamma_sr
        self.gamma_o = gamma_o
        self.eta_sr = eta_sr
        self.eta_o = eta_o
        self.p_option = p_option
        self.n_steps = n_steps
        self.n_iter = n_iter

        self.grid = self.env.unwrapped.grid
        self.width = self.grid.width
        self.height = self.grid.height
        self.valid_states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self._extract_valid_states()
        self.n_valid_states = len(self.valid_states)
        self.n_actions = len(self.env.actions)
        
        self.psi = np.zeros((self.n_valid_states, self.n_valid_states))
        
        self.discovered_options = []
        
        self.all_transitions = []

    def _extract_valid_states(self):
        idx = 0
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is None or (cell is not None and cell.type != 'wall'):
                    self.valid_states.append((i, j))
                    self.state_to_idx[(i, j)] = idx
                    self.idx_to_state[idx] = (i, j)
                    idx += 1

    def get_next_state(self, state, action):
        x, y = state
        
        if action == 0:  # right
            next_state = (x + 1, y)
        elif action == 1:  # down
            next_state = (x, y + 1)
        elif action == 2:  # left
            next_state = (x - 1, y)
        elif action == 3:  # up
            next_state = (x, y - 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if (next_state[0] < 0 or next_state[0] >= self.width or 
            next_state[1] < 0 or next_state[1] >= self.height):
            return state
        
        cell = self.grid.get(*next_state)
        if cell is not None and cell.type == 'wall':
            return state
        
        if next_state in self.state_to_idx:
            return next_state
        else:
            return state

    def execute_option(self, option, start_state):
        current_state = start_state
        transitions = []
        
        while True:
            current_state_pos = self.idx_to_state[current_state]
            state_idx = self.env.pos_to_state[current_state_pos[0] + current_state_pos[1] * self.width]
            
            if option['termination'][state_idx] == 1:
                break
            
            action = option['policy'][state_idx]
            next_state_pos = self.get_next_state(current_state_pos, action)
            next_state = self.state_to_idx[next_state_pos]
            transitions.append((current_state, action, next_state))
            current_state = next_state
        
        return transitions, current_state

    def collect_samples_with_options(self):
        transitions = []
        
        for step in range(self.n_steps):
            if step % 100 == 0:
                current_pos = self.env.unwrapped.agent_pos
                current_state = (current_pos[0], current_pos[1])
                current_state_idx = self.state_to_idx[current_state]
            
            if random.random() < (1 - self.p_option) or len(self.discovered_options) == 0:
                action = random.choice(range(self.n_actions))
                next_state = self.get_next_state(current_state, action)
                next_state_idx = self.state_to_idx[next_state]
                transitions.append((current_state_idx, action, next_state_idx))
                current_state = next_state
                current_state_idx = next_state_idx
                
            else:
                option = random.choice(self.discovered_options)

                if current_state in option['initiation_set']:   # option available from current state
                    option_transitions, final_state_idx = self.execute_option(option, current_state_idx)
                    transitions.extend(option_transitions)
                    current_state_idx = final_state_idx
                    current_state = self.idx_to_state[current_state_idx]
                else:   # take primitive action
                    action = random.choice(range(self.n_actions))
                    next_state = self.get_next_state(current_state, action)
                    next_state_idx = self.state_to_idx[next_state]
                    transitions.append((current_state_idx, action, next_state_idx))
                    current_state = next_state
                    current_state_idx = next_state_idx

        return transitions

    def learn_successor_representation(self, transitions):
        for _ in range(100): # needed?
            for step, (s, a, s_next) in enumerate(transitions):
                for i in range(self.n_valid_states):
                    indicator = 1.0 if s == i else 0.0
                    delta = indicator + self.gamma_sr * self.psi[s_next, i] - self.psi[s, i]
                    self.psi[s, i] += self.eta_sr * delta

    def get_top_eigenvector(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.psi)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        top_eigenvector = np.real(eigenvectors[:, idx[1]])
        top_eigenvalue = eigenvalues[idx[1]]
        
        return top_eigenvector, top_eigenvalue

    def q_learning_for_eigenoption(self, reward_function, transitions, iterations=1000):
        Q = np.zeros((self.n_valid_states, self.n_actions))
        
        transition_buffer = list(transitions)
        
        for iter in range(iterations):
            for s, a, s_next in transition_buffer:
                intrinsic_reward = reward_function[s_next] - reward_function[s]
                td_target = intrinsic_reward + self.gamma_o * np.max(Q[s_next, :])
                td_error = td_target - Q[s, a]
                Q[s, a] += self.eta_o * td_error
            
        return Q

    def create_option(self, Q, eigenvector, eigenvalue, option_id):
        option_policy = np.full(self.env.num_states, -1)
        termination = np.ones(self.env.num_states)
        initiation_set = set()
        
        for i, state in enumerate(self.valid_states):
            state_idx = self.env.pos_to_state[state[0] + state[1] * self.env.width]
            
            if np.max(Q[i, :]) > 0:
                initiation_set.add(state)
                best_action = np.argmax(Q[i, :])
                option_policy[state_idx] = best_action
                termination[state_idx] = 0
        
        option = {
            'id': option_id,
            'eigenvalue': eigenvalue,
            'eigenvector': eigenvector,
            'initiation_set': initiation_set,
            'policy': option_policy,
            'termination': termination,
            'Q_function': Q
        }
        
        return option

    def discover_covering_eigenoptions(self):
        self.all_transitions = []
        self.discovered_options = []
        
        # ROD cycle
        for iteration in range(self.n_iter):
            print(f"Iteration: {iteration + 1}/{self.n_iter}")
            
            new_transitions = self.collect_samples_with_options()
            self.all_transitions.extend(new_transitions)
            
            self.learn_successor_representation(self.all_transitions)
            
            top_eigenvector, top_eigenvalue = self.get_top_eigenvector()
            
            Q = self.q_learning_for_eigenoption(top_eigenvector, self.all_transitions)
            
            new_option = self.create_option(Q, top_eigenvector, top_eigenvalue, len(self.discovered_options))
            self.discovered_options.append(new_option)
        
        return self.discovered_options

    def visualize_option_policy(self, option_id, save_path_prefix=None):
        option = self.discovered_options[option_id]
        obs = self.env.reset()
        
        policy_path = f"{save_path_prefix}_eigenoption_policy_{option_id}.png" if save_path_prefix else f"covering_eigenoption_policy_{option_id}.png"
        self.env.render_option_policy(
            obs, option, policy_path,
            header=f"Covering EigenOption {option_id} Policy - ($\lambda$={option['eigenvalue']:.4f})"
        )

    def visualize_all_policies(self, save_path_prefix=None):
        for i, option in enumerate(self.discovered_options):
            self.visualize_option_policy(i, save_path_prefix)

def main():
    gin.parse_config_file('minigrid_basics/envs/classic_fourrooms.gin')
    env_name = mon_minigrid.register_environment()
    
    agent = CoveringEigenoptions(env_name, gamma_sr=0.99, gamma_o=0.99, eta_sr=0.1, eta_o=0.1, p_option=0.05, n_steps=100, n_iter=14)
    
    covering_eigenoptions = agent.discover_covering_eigenoptions()
    
    for i in range(min(10, len(covering_eigenoptions))):
        agent.visualize_option_policy(i, save_path_prefix="covering")
    
    return agent, covering_eigenoptions

if __name__ == "__main__":
    agent, options = main()