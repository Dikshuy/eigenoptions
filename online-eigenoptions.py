import numpy as np
import matplotlib.pyplot as plt
import random

import gin
import gym
from minigrid_basics.envs import mon_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.custom_wrappers import tabular_wrapper, mdp_wrapper
from minigrid_basics.custom_wrappers.coloring_wrapper import ColoringWrapper

class OnlineEigenoptions:
    def __init__(self, env_name, gamma_sr=0.9, gamma_o=0.9, eta_sr=0.1, eta_o=0.1, n_steps=10000):
        pre_env = gym.make(env_name)
        pre_env = RGBImgObsWrapper(pre_env)
        pre_env = mdp_wrapper.MDPWrapper(pre_env)
        self.env = ColoringWrapper(pre_env, tile_size=32)
        
        self.gamma_sr = gamma_sr
        self.gamma_o = gamma_o
        self.eta_sr = eta_sr
        self.eta_o = eta_o
        self.n_steps = n_steps

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
        
        self.eigenoption_q_functions = []
        self.eigenoptions = []
        
        self.transitions = []

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

    def collect_transitions(self):
        transitions = []
        
        for step in range(self.n_steps):
            if step % 1000 == 0:
                obs = self.env.reset()
                current_pos = self.env.unwrapped.agent_pos
                current_state = (current_pos[0], current_pos[1])
            
            action = random.choice(range(self.n_actions))
            next_state = self.get_next_state(current_state, action)
            
            if current_state in self.state_to_idx and next_state in self.state_to_idx:
                s_idx = self.state_to_idx[current_state]
                s_prime_idx = self.state_to_idx[next_state]
                transitions.append((s_idx, action, s_prime_idx))
            
            current_state = next_state
        
        self.transitions = transitions
        return transitions

    def learn_successor_representation(self):
        for step, (s, a, s_prime) in enumerate(self.transitions):
            for i in range(self.n_valid_states):
                indicator = 1.0 if s == i else 0.0
                delta = indicator + self.gamma_sr * self.psi[s_prime, i] - self.psi[s, i]
                self.psi[s, i] += self.eta_sr * delta
        
        return self.psi

    def q_learning_for_eigenoption(self, reward_function, max_episodes=1000):
        Q = np.zeros((self.n_valid_states, self.n_actions))
        
        for episode in range(max_episodes):
            start_state_idx = random.choice(range(self.n_valid_states))
            current_state_idx = start_state_idx
            
            for step in range(1000):    # Max steps per episode
                current_state = self.idx_to_state[current_state_idx]
                epsilon = max(0.1, 1.0 - episode / max_episodes)
                if random.random() < epsilon:
                    action = random.choice(range(self.n_actions))
                else:
                    action = np.argmax(Q[current_state_idx, :])
                
                next_state = self.get_next_state(current_state, action)
                next_state_idx = self.state_to_idx[next_state]
                
                intrinsic_reward = reward_function[next_state_idx] - reward_function[current_state_idx]
                
                td_target = intrinsic_reward + self.gamma_o * np.max(Q[next_state_idx, :])
                td_error = td_target - Q[current_state_idx, action]
                Q[current_state_idx, action] += self.eta_o * td_error
                
                current_state_idx = next_state_idx
        
        return Q

    def discover_eigenoptions_online(self, n_eigenoptions=10):
        self.collect_transitions()
        
        psi = self.learn_successor_representation()
        
        eigenvalues, eigenvectors = np.linalg.eig(psi)
        
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.eigenoptions = []
        
        for k in range(min(n_eigenoptions, len(eigenvalues))):
            e = np.real(eigenvectors[:, k])
            Q = self.q_learning_for_eigenoption(e, max_episodes=500)
            
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
            
            eigenoption = {
                'id': k,
                'eigenvalue': eigenvalues[k],
                'eigenvector': e,
                'initiation_set': initiation_set,
                'policy': option_policy,
                'termination': termination,
                'Q_function': Q
            }
            
            self.eigenoptions.append(eigenoption)
        
        return self.eigenoptions
    
    def visualize_eigenoption_policy(self, eigenoption_id, save_path_prefix=None):
        eigenoption = self.eigenoptions[eigenoption_id]
        obs = self.env.reset()
        
        policy_path = f"{save_path_prefix}_policy_{eigenoption_id}.png" if save_path_prefix else f"eigenoption_{eigenoption_id}_policy.png"
        self.env.render_option_policy(
            obs, eigenoption, policy_path,
            header=f"Eigenoption {eigenoption_id} Policy - ($\lambda$={eigenoption['eigenvalue']:.4f})"
        )

    def visualize_all_policies(self, save_path_prefix=None):
        for i, eigenoption in enumerate(self.eigenoptions):
            self.visualize_eigenoption_policy(i, save_path_prefix)

def main():
    gin.parse_config_file('minigrid_basics/envs/classic_fourrooms.gin')
    env_name = mon_minigrid.register_environment()
    
    agent = OnlineEigenoptions(env_name, gamma_sr=0.9, gamma_o=0.9, eta_sr=0.1, eta_o=0.1, n_steps=100000)
    
    eigenoptions = agent.discover_eigenoptions_online(n_eigenoptions=10)
    
    for i in range(1, min(4, len(eigenoptions))):
        agent.visualize_eigenoption_policy(i, save_path_prefix="eigenoption")
    
    return agent, eigenoptions

if __name__ == "__main__":
    agent, eigenoptions = main()