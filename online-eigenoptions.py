import numpy as np
import matplotlib.pyplot as plt

import gin
import gym
from minigrid_basics.envs import mon_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.custom_wrappers import tabular_wrapper, mdp_wrapper
from minigrid_basics.custom_wrappers.coloring_wrapper import ColoringWrapper

class SuccessorRepresentation:
    def __init__(self, eta, gamma, D):
        self.eta = eta
        self.gamma = gamma
        self.D = D

    def SR(self):
        psi = np.zeros((self.D.shape[0], self.D.shape[0]))

        for (s, a, s_next) in self.D:
            for i in range(self.D.shape[0]):
                delta = 1 if i == s else 0
                psi[s, i] += self.eta * (delta + self.gamma * psi[s_next, i] - psi[s, i])

        return psi

class OnlineEigenoptions:
    def __init__(self, env_name, gamma_sr=0.9, gamma_o=0.9, n_episodes=1000, learning_rate=0.1):
        pre_env = gym.make(env_name)
        pre_env = RGBImgObsWrapper(pre_env)
        pre_env = mdp_wrapper.MDPWrapper(pre_env)
        self.env = ColoringWrapper(pre_env, tile_size=32)
        
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
    
    def successor_representation(self):
        I = np.eye(self.n_valid_states)
        try:
            psi = np.linalg.inv(I - self.gamma_sr * self.P)
        except np.linalg.LinAlgError:
            psi = np.linalg.pinv(I - self.gamma_sr * self.P)
        
        return psi
    
    def laplacian(self):
        degrees = np.sum(self.P, axis=1)
        D = np.diag(degrees)
        
        # Normalized Laplacian: L = I - D^(-1/2) * P * D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
        I = np.eye(len(self.P))
        return I - D_inv_sqrt @ self.P @ D_inv_sqrt
    
    def compute_eigenvalues(self, representation="Laplacian"):
        if representation == "Laplacian":
            L = self.laplacian()
            eigenvalues, eigenvectors = np.linalg.eig(L)
            
            idx = np.argsort(np.abs(eigenvalues))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            return eigenvalues, eigenvectors
        
        if representation == "SR":
            psi = self.successor_representation()
            eigenvalues, eigenvectors = np.linalg.eig(psi)
        
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
    
    def discover_eigenoptions(self, n_eigenoptions=10, representation="Laplacian"):
        eigenvalues, eigenvectors = self.compute_eigenvalues(representation)
        
        self.eigenoptions = []
        
        for k in range(min(n_eigenoptions, len(eigenvalues))):
            e = eigenvectors[:, k]
            
            # intrinsic reward function: r(s,s') = e(s') - e(s)
            reward_function = e
            
            Q = self.policy_iteration(reward_function)
            
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
    
    agent = OnlineEigenoptions(env_name, gamma_sr=0.9, gamma_o=0.9, n_episodes=1000, learning_rate=0.1)

    representation = "Laplacian" # or "SR"
    
    eigenoptions = agent.discover_eigenoptions(n_eigenoptions=10, representation=representation)
    
    for i in range(min(4, len(eigenoptions))):
        agent.visualize_eigenoption_policy(i, save_path_prefix="eigenoption")
    
    return agent, eigenoptions

if __name__ == "__main__":
    agent, eigenoptions = main()