import numpy as np
import matplotlib.pyplot as plt

import gin
import gym
from minigrid_basics.envs import mon_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.custom_wrappers import tabular_wrapper, mdp_wrapper

class Eigenoptions:
    def __init__(self, env_name, gamma_sr=0.9, gamma_o=0.9, n_episodes=1000, learning_rate=0.1):
        pre_env = gym.make(env_name)
        pre_env = RGBImgObsWrapper(pre_env)
        pre_env = mdp_wrapper.MDPWrapper(pre_env)
        self.env = tabular_wrapper.TabularWrapper(pre_env, get_rgb = True)
        self.gamma_sr = gamma_sr
        self.gamma_o = gamma_o
        self.n_episodes = n_episodes
        self.lr = learning_rate

        self.grid = self.env.unwrapped.grid
        self.size = self.grid.width
        self.valid_states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self._extract_valid_states()
        self.n_valid_states = len(self.valid_states)
        self.n_actions = len(self.env.actions)
        
        self.P = self._build_transition_matrix()
        
        self.eigenoptions = []

    def _extract_valid_states(self):
        idx = 0
        for i in range(self.size):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)
                if cell is None or (cell is not None and cell.type != 'wall'):
                    self.valid_states.append((i, j))
                    self.state_to_idx[(i, j)] = idx
                    self.idx_to_state[idx] = (i, j)
                    idx += 1

    def get_next_state(self, state, action):
        x, y = state
        if action == 0:
            next_state = (x, y - 1)
        elif action == 1:
            next_state = (x, y + 1)
        elif action == 2:
            next_state = (x - 1, y)
        elif action == 3:
            next_state = (x + 1, y)
        else:
            raise ValueError("Invalid action")
        
        cell = self.grid.get(*next_state)
        if cell is not None and cell.type == 'wall':
            return state
        
        if next_state in self.state_to_idx:
            return next_state
        else:
            return state
        
    def _build_transition_matrix(self):
        P = np.zeros((self.n_valid_states, self.n_valid_states))
        
        for i, state in enumerate(self.valid_states):
            for action in range(self.n_actions):
                next_state = self.get_next_state(state, action)
                if next_state in self.state_to_idx:
                    j = self.state_to_idx[next_state]
                    P[i, j] += 1.0 / self.n_actions
        
        return P
    
    def learn_successor_representation(self):
        I = np.eye(self.n_valid_states)
        try:
            self.Psi = np.linalg.inv(I - self.gamma_sr * self.P)
        except np.linalg.LinAlgError:
            self.Psi = np.linalg.pinv(I - self.gamma_sr * self.P)
        
        return self.Psi
    
    def eigendecomposition(self, Psi):
        eigenvalues, eigenvectors = np.linalg.eig(Psi)
        
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def policy_iteration(self, reward_function, max_iterations=100):
        Q = np.zeros((self.n_valid_states, self.n_actions))
        
        for _ in range(max_iterations):
            Q_old = Q.copy()
            
            for i, state in enumerate(self.valid_states):
                for a, action in enumerate(self.env.actions):
                    next_state = self.get_next_state(state, action)
                    if next_state in self.state_to_idx:
                        j = self.state_to_idx[next_state]
                        reward = reward_function[j] - reward_function[i]
                        Q[i, a] = reward + self.gamma_o * np.max(Q[j, :])
            
            if np.max(np.abs(Q - Q_old)) < 1e-6:
                break
        
        return Q
    
    def discover_eigenoptions(self, n_eigenoptions=10):
        Psi = self.learn_successor_representation()
        eigenvalues, eigenvectors = self.eigendecomposition(Psi)
        
        self.eigenoptions = []
        
        for k in range(n_eigenoptions):
            e = eigenvectors[:, k]
            
            # intrinsic reward function: r(s,s') = e(s') - e(s)
            reward_function = e
            
            Q = self.policy_iteration(reward_function)
            
            option_policy = {}
            termination = {}
            initiation_set = set()
            
            for i, state in enumerate(self.valid_states):
                if np.max(Q[i, :]) > 0:
                    initiation_set.add(state)
                    best_action = np.argmax(Q[i, :])
                    option_policy[state] = best_action
                    termination[state] = 0
                else:
                    termination[state] = 1
            
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
    
    def visualize_eigenoption(self, eigenoption_id, save_path=None):        
        eigenoption = self.eigenoptions[eigenoption_id]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        eigenvector_grid = np.full((self.size, self.size), np.nan)
        for i, state in enumerate(self.valid_states):
            x, y = self.env.state_to_pos(state)
            eigenvector_grid[y, x] = eigenoption['eigenvector'][i]
        
        im1 = ax1.imshow(eigenvector_grid, cmap='RdBu', origin='upper')
        ax1.set_title(f'Eigenoption {eigenoption_id} - Eigenvector Values')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1)
        
        policy_grid = np.full((self.env.size, self.env.size), -1)
        for state, action_idx in eigenoption['policy'].items():
            x, y = self.env.state_to_pos(state)
            policy_grid[y, x] = action_idx
        
        arrow_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        
        ax2.imshow(np.zeros((self.size, self.size)), cmap='gray', alpha=0.3)
        
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) not in self.env.walls and policy_grid[y, x] >= 0:
                    dx, dy = arrow_dirs[int(policy_grid[y, x])]
                    ax2.arrow(x, y, dx*0.3, dy*0.3, head_width=0.1, 
                             head_length=0.1, fc='red', ec='red')
        
        ax2.set_title(f'Eigenoption {eigenoption_id} - Policy')
        ax2.set_xlim(-0.5, self.env.size-0.5)
        ax2.set_ylim(-0.5, self.env.size-0.5)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    gin.parse_config_file('minigrid_basics/envs/classic_fourrooms.gin')
    env_name = mon_minigrid.register_environment()
    
    agent = Eigenoptions(env_name, gamma_sr=0.9, gamma_o=0.9)
    
    eigenoptions = agent.discover_eigenoptions(n_eigenoptions=10)
    
    for i in range(min(4, len(eigenoptions))):
        agent.visualize_eigenoption(i)
    
    return agent, eigenoptions

if __name__ == "__main__":
    agent, eigenoptions = main()