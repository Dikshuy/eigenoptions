import os

import matplotlib
import numpy as np

LOGS_DIR = "logs"
POLICIES_SUBDIR = "policies"
VISITATION_SUBDIR = "visitation"


def _save_path(subdir, filename):
    directory = os.path.join(LOGS_DIR, subdir)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def save_visitation_heatmap(env, visit_counts, save_path_prefix=None):
    filename = f"{save_path_prefix}_state_visitation.png" if save_path_prefix else "state_visitation.png"
    image_loc = _save_path(VISITATION_SUBDIR, filename)
    obs = env.reset()
    env.render_state_visits(obs, visit_counts, image_loc, cmap=matplotlib.colormaps["YlOrRd"])


def visualize_option_policy(env, option, option_id, save_path_prefix=None):
    obs = env.reset()

    filename = f"{save_path_prefix}_eigenoption_policy_{option_id}.png" if save_path_prefix else f"eigenoption_policy_{option_id}.png"
    policy_path = _save_path(POLICIES_SUBDIR, filename)
    env.render_option_policy(
        obs, option, policy_path,
        header=f"Covering EigenOption {option_id} Policy - ($\\lambda$={option['eigenvalue']:.4f})"
    )


def visualize_all_policies(env, options, save_path_prefix=None):
    for i, option in enumerate(options):
        visualize_option_policy(env, option, i, save_path_prefix)


def visualize_state_visitation(env, transitions, n_valid_states, save_path_prefix=None):
    visit_counts = np.zeros(n_valid_states)
    for s, _, s_next in transitions:
        visit_counts[s] += 1
        visit_counts[s_next] += 1

    save_visitation_heatmap(env, visit_counts, save_path_prefix)


def cover_time_stats(cover_times):
    cover_times = np.array(cover_times)
    return {
        'mean': float(np.mean(cover_times)),
        'std': float(np.std(cover_times, ddof=1)),
        'median': float(np.median(cover_times)),
        'min': float(np.min(cover_times)),
        'max': float(np.max(cover_times)),
    }
