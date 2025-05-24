import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import gymnasium as gym
from minigrid.envs import FourRoomsEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall
from scipy.linalg import eigh

class SpectralClustering:
    def __init__(self, env_name='MiniGrid-FourRooms-v0'):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.reset()
        
        self.grid = self.env.unwrapped.grid
        self.env_size = self.grid.width
        self.valid_states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        print(f"Environment size: {self.env_size}x{self.grid.height}")
        self._extract_valid_states()
        
    def _extract_valid_states(self):
        idx = 0
        for i in range(self.env_size):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)
                if cell is None or not isinstance(cell, Wall):
                    self.valid_states.append((i, j))
                    self.state_to_idx[(i, j)] = idx
                    self.idx_to_state[idx] = (i, j)
                    idx += 1
        
    def build_adjacency_matrix(self):
        n_states = len(self.valid_states)
        adjacency = np.zeros((n_states, n_states))
        
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for i, state in enumerate(self.valid_states):
            x, y = state
            for dx, dy in moves:
                next_x, next_y = x + dx, y + dy
                
                if (0 <= next_x < self.env_size and 
                    0 <= next_y < self.grid.height and 
                    (next_x, next_y) in self.state_to_idx):
                    
                    j = self.state_to_idx[(next_x, next_y)]
                    adjacency[i, j] = 1
                    
        return adjacency
    
    def compute_laplacian(self, adjacency, laplacian_type='normalized'):
        degrees = np.sum(adjacency, axis=1)
        D = np.diag(degrees)
        
        if laplacian_type == 'unnormalized':
            return D - adjacency
        
        elif laplacian_type == 'normalized':
            # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
            I = np.eye(len(adjacency))
            return I - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        elif laplacian_type == 'random_walk':
            # Random walk Laplacian: L = I - D^(-1) * A
            D_inv = np.diag(1.0 / (degrees + 1e-8))
            I = np.eye(len(adjacency))
            return I - D_inv @ adjacency
        
        else:
            raise ValueError("laplacian_type must be 'unnormalized', 'normalized', or 'random_walk'")
    
    def spectral_clustering(self, laplacian, n_clusters=4):
        eigenvalues, eigenvectors = eigh(laplacian)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # first n_clusters eigenvectors (excluding the first for normalized Laplacian)
        if np.abs(eigenvalues[0]) < 1e-8:  # first eigenvalue is approximately 0
            embedding = eigenvectors[:, 1:n_clusters+1]
        else:
            embedding = eigenvectors[:, :n_clusters]
        
        embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding_norm[embedding_norm == 0] = 1
        embedding = embedding / embedding_norm
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(embedding)
        
        return cluster_labels, eigenvalues, eigenvectors, embedding
    
    def analyze_eigenvalues(self, eigenvalues, plot=True):
        eigengaps = np.diff(eigenvalues)
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot eigenvalues
            ax1.plot(eigenvalues[:20], 'bo-')
            ax1.set_title('First 20 Eigenvalues')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Eigenvalue')
            ax1.grid(True)
            
            # Plot eigengaps
            ax2.plot(eigengaps[:20], 'ro-')
            ax2.set_title('Eigengaps')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Eigengap')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        largest_gap_idx = np.argmax(eigengaps[:10]) + 1  # +1 because we want the number of clusters
        print(f"largest eigengap suggests {largest_gap_idx} clusters")
        
        return eigengaps
    
    def visualize_clusters(self, cluster_labels, title="Spectral Clustering Results"):
        cluster_grid = np.full((self.env_size, self.grid.height), -1, dtype=int)  # -1 for walls
        
        for idx, (x, y) in enumerate(self.valid_states):
            cluster_grid[x, y] = cluster_labels[idx]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        n_clusters = len(np.unique(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        wall_color = 'black'
        
        for i in range(self.env_size):
            for j in range(self.grid.height):
                if cluster_grid[i, j] == -1:  # Wall
                    rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor=wall_color, 
                                       edgecolor='gray', 
                                       linewidth=0.5)
                    ax.add_patch(rect)
                else:  # Valid state
                    cluster_id = cluster_grid[i, j]
                    rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                       facecolor=colors[cluster_id], 
                                       edgecolor='white', 
                                       linewidth=1.0,
                                       alpha=0.8)
                    ax.add_patch(rect)
                    
                    ax.text(i, j, f'{cluster_id}', 
                           ha='center', va='center', fontsize=8,
                           color='black' if cluster_id != -1 else 'white')
        
        ax.set_xlim(-0.5, self.env_size-0.5)
        ax.set_ylim(-0.5, self.grid.height-0.5)
        ax.set_aspect('equal')

        ax.set_xticks(range(self.env_size))
        ax.set_yticks(range(self.grid.height))
        ax.grid(True, alpha=0.1, color='gray', linewidth=0.5)
        
        ax.set_title(title, fontsize=14)
        
        legend_elements = []
        for i in range(n_clusters):
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                               edgecolor='white', label=f'Cluster {i}'))
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=wall_color, 
                                           edgecolor='gray', label='Wall'))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.show()

    def visualize_eigenvectors(self, eigenvectors, eigenvalues, n_vecs=6, title_prefix="Eigenvector"):
        start_idx = 1 if np.abs(eigenvalues[0]) < 1e-8 else 0
        end_idx = start_idx + n_vecs
        
        n_cols = 3
        n_rows = int(np.ceil(n_vecs / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_vecs):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            vec_idx = start_idx + i
            if vec_idx >= eigenvectors.shape[1]:
                ax.axis('off')
                continue
                
            eigenvector = eigenvectors[:, vec_idx]
            eigenvalue = eigenvalues[vec_idx]
            
            vec_grid = np.full((self.env_size, self.grid.height), np.nan)
            
            for state_idx, (x, y) in enumerate(self.valid_states):
                vec_grid[x, y] = eigenvector[state_idx]
            
            im = ax.imshow(vec_grid.T, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
            
            ax.set_title(f'{title_prefix} {vec_idx+1}\n(λ = {eigenvalue:.4f})', fontsize=12)
            ax.grid(True, alpha=0.1, color='white', linewidth=0.5)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        for i in range(n_vecs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_clustering(self, embedding, cluster_labels):
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(embedding, cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
            return silhouette
        return None
    
    def run(self, n_clusters_range=range(2, 8), laplacian_type='normalized', random_seed=42):
        np.random.seed(random_seed)
        
        adjacency = self.build_adjacency_matrix()
        laplacian = self.compute_laplacian(adjacency, laplacian_type)

        eigenvalues, eigenvectors = eigh(laplacian)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.analyze_eigenvalues(eigenvalues)
        self.visualize_eigenvectors(eigenvectors, eigenvalues, n_vecs=6, title_prefix=f"{laplacian_type.title()} eigenvector")
        
        best_score = -1
        best_k = 4
        results = {}
        
        for k in n_clusters_range:
            print(f"Testing k={k}")
            cluster_labels, eigenvalues, eigenvectors, embedding = self.spectral_clustering(laplacian, n_clusters=k)
            
            score = self.evaluate_clustering(embedding, cluster_labels)
            results[k] = {
                'labels': cluster_labels,
                'score': score,
                'embedding': embedding,
                'eigenvalues': eigenvalues
            }
            
            if score and score > best_score:
                best_score = score
                best_k = k
        
        print(f"\nBest number of clusters: {best_k} (Silhouette Score: {best_score:.4f})")
        
        self.visualize_clusters(results[best_k]['labels'], f"Best Spectral Clustering (k={best_k}, score={best_score:.4f})")
        
        return results


if __name__ == "__main__":
    env_name = 'MiniGrid-FourRooms-v0'
    clustering = SpectralClustering(env_name)
    
    results = clustering.run(n_clusters_range=range(2, 8), laplacian_type='normalized')
    
    # # testing different Laplacian types
    # adjacency = clustering.build_adjacency_matrix()
    
    # for laplacian_type in ['unnormalized', 'normalized', 'random_walk']:
    #     print(f"\n--- {laplacian_type.upper()} LAPLACIAN ---")
    #     laplacian = clustering.compute_laplacian(adjacency, laplacian_type)
        
    #     eigenvalues, eigenvectors = eigh(laplacian)
    #     idx = np.argsort(eigenvalues)
    #     eigenvalues = eigenvalues[idx]
    #     eigenvectors = eigenvectors[:, idx]
        
    #     clustering.visualize_eigenvectors(eigenvectors, eigenvalues, n_vecs=4, title_prefix=f"{laplacian_type.title()} eigenvector")