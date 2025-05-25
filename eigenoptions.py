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
        self.env = tabular_wrapper.TabularWrapper(pre_env, get_rgb=True)
        self.gamma_sr = gamma_sr
        self.gamma_o = gamma_o
        self.n_episodes = n_episodes
        self.lr = learning_rate

        self.grid = self.env.unwrapped.grid
        self.width = self.grid.width
        self.height = self.grid.height
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
                for action in range(self.n_actions):
                    next_state = self.get_next_state(state, action)
                    if next_state in self.state_to_idx:
                        j = self.state_to_idx[next_state]         
                        reward = reward_function[j] - reward_function[i]
                        Q[i, action] = reward + self.gamma_o * np.max(Q[j, :])
            
            if np.max(np.abs(Q - Q_old)) < 1e-6:
                break
        
        return Q
    
    def discover_eigenoptions(self, n_eigenoptions=10):
        Psi = self.learn_successor_representation()
        eigenvalues, eigenvectors = self.eigendecomposition(Psi)
        
        self.eigenoptions = []
        
        for k in range(min(n_eigenoptions, len(eigenvalues))):
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
        
        eigenvector_grid = np.full((self.height, self.width), np.nan)
        
        for i, state in enumerate(self.valid_states):
            x, y = state
            eigenvector_grid[y, x] = eigenoption['eigenvector'][i]
        
        im1 = ax1.imshow(eigenvector_grid, cmap='RdBu', origin='upper')
        ax1.set_title(f'Eigenoption {eigenoption_id} - Eigenvector Values\n'
                     f'Eigenvalue: {eigenoption["eigenvalue"]:.4f}')
        ax1.grid(True, alpha=0.3)
        
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == 'wall':
                    ax1.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                              fill=True, color='black'))
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        policy_bg = np.zeros((self.height, self.width))
        ax2.imshow(policy_bg, cmap='gray', alpha=0.1, origin='upper')
        
        arrow_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        action_names = ['>', 'v', '<', '^']
        
        for i, state in enumerate(self.valid_states):
            x, y = state
            
            if state in eigenoption['policy']:
                action_idx = eigenoption['policy'][state]
                dx, dy = arrow_dirs[action_idx]
                
                # Color based on initiation set
                color = 'red' if state in eigenoption['initiation_set'] else 'orange'
                ax2.arrow(x, y, dx*0.3, dy*0.3, head_width=0.15, 
                         head_length=0.1, fc=color, ec=color, alpha=0.8)
                
                # Add action symbol
                ax2.text(x, y-0.15, action_names[action_idx], 
                        ha='center', va='center', fontsize=8, 
                        color='darkred', weight='bold')
            
            elif eigenoption['termination'].get(state, 0) == 1:
                # Terminal state - mark with X
                ax2.scatter(x, y, c='blue', s=100, marker='X', alpha=0.8)
                ax2.text(x, y+0.25, 'T', ha='center', va='center', 
                        fontsize=10, color='blue', weight='bold')
        
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == 'wall':
                    ax2.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                              fill=True, color='black'))
        
        ax2.set_title(f'Eigenoption {eigenoption_id} - Policy & Termination\n'
                     f'Red: Policy+Initiation, Orange: Policy only, Blue X: Terminal')
        ax2.set_xlim(-0.5, self.width-0.5)
        ax2.set_ylim(-0.5, self.height-0.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='>', color='red', label='Policy (in initiation set)', 
                   markersize=8, linestyle='None'),
            Line2D([0], [0], marker='>', color='orange', label='Policy (not in initiation set)', 
                   markersize=8, linestyle='None'),
            Line2D([0], [0], marker='X', color='blue', label='Terminal states', 
                   markersize=8, linestyle='None'),
            Line2D([0], [0], marker='s', color='black', label='Walls', 
                   markersize=8, linestyle='None')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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