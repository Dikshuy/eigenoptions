import numpy as np
import random
import gin

import gym
from minigrid_basics.envs import mon_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.custom_wrappers import mdp_wrapper
from minigrid_basics.custom_wrappers.coloring_wrapper import ColoringWrapper

import utils


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
        self.n_valid_states = self.env.num_states
        self.n_actions = self.env.num_actions
        self.stochasticity = self.env.unwrapped.stochasticity
        self.max_steps = self.env.unwrapped.max_steps

        self.psi = np.zeros((self.n_valid_states, self.n_valid_states))
        self.discovered_options = []
        self.all_transitions = []

    def sample_next_state(self, state, action):
        probs = self.env.transition_probs[state, action]
        if self.stochasticity == 0.:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_valid_states, p=probs))

    def execute_option(self, option, start_state, max_steps):
        current_state = start_state
        transitions = []

        for _ in range(max_steps):
            if option['termination'][current_state] == 1:
                break

            action = option['policy'][current_state]
            next_state = self.sample_next_state(current_state, action)
            transitions.append((current_state, action, next_state))
            current_state = next_state

        return transitions, current_state

    def collect_samples_with_options(self):
        transitions = []
        obs = self.env.reset()
        current_state = obs['state']

        while len(transitions) < self.n_steps:
            remaining_steps = self.n_steps - len(transitions)

            if random.random() < (1 - self.p_option) or len(self.discovered_options) == 0:
                action = random.choice(range(self.n_actions))
                next_state = self.sample_next_state(current_state, action)
                transitions.append((current_state, action, next_state))
                current_state = next_state

            else:
                option = random.choice(self.discovered_options)

                if current_state in option['initiation_set']:   # option available from current state
                    option_transitions, current_state = self.execute_option(option, current_state, remaining_steps)
                    transitions.extend(option_transitions)
                else:   # take primitive action
                    action = random.choice(range(self.n_actions))
                    next_state = self.sample_next_state(current_state, action)
                    transitions.append((current_state, action, next_state))
                    current_state = next_state

        return transitions

    def learn_successor_representation(self, transitions, epochs=100):
        identity = np.eye(self.n_valid_states)
        for _ in range(epochs):
            for s, a, s_next in transitions:
                target = identity[s] + self.gamma_sr * self.psi[s_next]
                self.psi[s] += self.eta_sr * (target - self.psi[s])

    def get_top_eigenvector(self):
        self.psi = (self.psi + self.psi.T)/2.0
        eigenvalues, eigenvectors = np.linalg.eigh(self.psi)
        idx = np.argmax(np.abs(eigenvalues))
        eigenvector = eigenvectors[:, idx]
        eigenvalue = eigenvalues[idx]

        # pick the direction such that sum_i e(i) < 0
        if np.sum(eigenvector) > 0:
            eigenvector = -eigenvector

        return eigenvector, eigenvalue

    def q_learning_for_eigenoption(self, reward_function, transitions, iterations=1000):
        Q = [[0.0] * self.n_actions for _ in range(self.n_valid_states)]
        intrinsic_rewards = [reward_function[s_next] - reward_function[s] for s, _, s_next in transitions]

        for _ in range(iterations):
            for (s, a, s_next), intrinsic_reward in zip(transitions, intrinsic_rewards):
                td_target = intrinsic_reward + self.gamma_o * max(Q[s_next])
                Q[s][a] += self.eta_o * (td_target - Q[s][a])

        return np.array(Q)

    def create_option(self, Q, eigenvector, eigenvalue, option_id):
        option_policy = np.full(self.n_valid_states, -1)
        termination = np.ones(self.n_valid_states)
        initiation_set = set()

        for state in range(self.n_valid_states):
            if np.max(Q[state, :]) > 0:
                initiation_set.add(state)
                option_policy[state] = np.argmax(Q[state, :])
                termination[state] = 0

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

    def run_rod_iteration(self):
        """execute ROD cycle"""
        new_transitions = self.collect_samples_with_options()
        self.all_transitions.extend(new_transitions)

        self.learn_successor_representation(new_transitions)

        eigenvector, eigenvalue = self.get_top_eigenvector()

        Q = self.q_learning_for_eigenoption(eigenvector, self.all_transitions)

        new_option = self.create_option(Q, eigenvector, eigenvalue, len(self.discovered_options))
        self.discovered_options.append(new_option)

        return new_transitions

    def discover_covering_eigenoptions(self):
        self.psi = np.zeros((self.n_valid_states, self.n_valid_states))
        self.all_transitions = []
        self.discovered_options = []

        for iteration in range(self.n_iter):
            print(f"Iteration: {iteration + 1}/{self.n_iter}")

            self.run_rod_iteration()

            utils.visualize_option_policy(self.env, self.discovered_options[-1], iteration, save_path_prefix="covering")

        utils.visualize_state_visitation(self.env, self.all_transitions, self.n_valid_states, save_path_prefix="covering")

        return self.discovered_options

    def estimate_random_walk_cover_time(self, n_seeds=100, save_heatmap_prefix=None):
        cover_times = []
        proportion_sum = np.zeros(self.n_valid_states)
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            obs = self.env.reset()
            start_state = obs['state']
            current_state = start_state
            visited = {current_state}
            visit_counts = np.zeros(self.n_valid_states)
            steps = 0
            episode_step = 0
            while len(visited) < self.n_valid_states:
                if episode_step >= self.max_steps:
                    current_state = start_state
                    episode_step = 0
                    continue

                action = random.choice(range(self.n_actions))
                current_state = self.sample_next_state(current_state, action)
                visit_counts[current_state] += 1
                visited.add(current_state)
                steps += 1
                episode_step += 1
            cover_times.append(steps)
            
            # per-seed proportion of interactions spent in each state
            proportion_sum += visit_counts / steps

        if save_heatmap_prefix is not None:
            utils.save_visitation_heatmap(self.env, 100 * proportion_sum / n_seeds, save_heatmap_prefix)

        return utils.cover_time_stats(cover_times)

    def estimate_cover_time_with_options(self, n_seeds=100, save_heatmap_prefix=None, max_iterations=1000):
        """estimate the cover time metric"""
        cover_times = []
        proportion_sum = np.zeros(self.n_valid_states)
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)

            self.psi = np.zeros((self.n_valid_states, self.n_valid_states))
            self.discovered_options = []
            self.all_transitions = []

            visited = set()
            visit_counts = np.zeros(self.n_valid_states)
            total_steps = 0
            cover_time = None

            for _ in range(max_iterations):
                new_transitions = self.run_rod_iteration()
                for s, a, s_next in new_transitions:
                    total_steps += 1
                    visit_counts[s_next] += 1
                    visited.add(s_next)
                    if cover_time is None and len(visited) == self.n_valid_states:
                        cover_time = total_steps

                if cover_time is not None:
                    break

            cover_times.append(cover_time)

            # per-seed proportion of interactions spent in each state
            proportion_sum += visit_counts / total_steps

        if save_heatmap_prefix is not None:
            utils.save_visitation_heatmap(self.env, 100 * proportion_sum / n_seeds, save_heatmap_prefix)

        return utils.cover_time_stats(cover_times)


def main():
    random.seed(42)
    np.random.seed(42)

    gin.parse_config_file('minigrid_basics/envs/classic_fourrooms.gin')
    env_name = mon_minigrid.register_environment()

    agent = CoveringEigenoptions(env_name, gamma_sr=0.99, gamma_o=0.99, eta_sr=0.1, eta_o=0.1, p_option=0.05, n_steps=100, n_iter=50)
    covering_eigenoptions = agent.discover_covering_eigenoptions()

    random_walk_stats = agent.estimate_random_walk_cover_time(n_seeds=100, save_heatmap_prefix="random_walk")
    ceo_stats = agent.estimate_cover_time_with_options(n_seeds=100, save_heatmap_prefix="ceo")
    print(f"Random-walk cover time: {random_walk_stats}")
    print(f"Covering eigenoptions (CEO) cover time: {ceo_stats}")

    return agent, covering_eigenoptions


if __name__ == "__main__":
    agent, options = main()