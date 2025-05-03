import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def create_grid_graph_adjacency(n):
    num_nodes = n * n
    
    A = np.zeros((num_nodes, num_nodes))
    
    for i in range(n):
        for j in range(n):
            node_idx = i * n + j
            
            if j < n - 1:
                right_idx = i * n + (j + 1)
                A[node_idx, right_idx] = 1
                A[right_idx, node_idx] = 1
            
            if i < n - 1:
                bottom_idx = (i + 1) * n + j
                A[node_idx, bottom_idx] = 1
                A[bottom_idx, node_idx] = 1
    
    return A

def compute_laplacian(A, normalized=True):
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    
    L = D - A
    
    if normalized:
        # normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt
    
    return L

def compute_eigendecomposition(L, k=None):
    n = L.shape[0]
    
    if k is None:
        k = n
    else:
        k = min(k, n)
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]
    
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def plot_eigenvector_3d_surface(eigenvector, grid_size, eigenvalue, ax=None):
    n = grid_size
    eigvec_reshaped = eigenvector.reshape(n, n)
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        standalone = True
    else:
        standalone = False
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, eigvec_reshaped, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_title(f"λ = {eigenvalue:.4f}")
    ax.set_xlabel('X' if standalone else '')
    ax.set_ylabel('Y' if standalone else '')
    ax.set_zlabel('Value' if standalone else '')
    
    if not standalone:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    if standalone:
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        return fig
    return surf

def plot_top_eigenvector_3d_surfaces(eigenvectors, eigenvalues, grid_size, top_k=10):
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 3*n_rows))
    
    for i in range(top_k):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        eigvec = eigenvectors[:, i]
        eigenval = eigenvalues[i]
        plot_eigenvector_3d_surface(eigvec, grid_size, eigenval, ax)
        
    plt.tight_layout()
    return fig

def plot_spectrum(eigenvalues):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(eigenvalues, 'o-', markersize=8)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Spectrum of Laplacian Eigenvalues')
    ax1.grid(True)
    
    # spectral gaps
    eigenvalue_gaps = np.diff(eigenvalues)
    ax2.plot(eigenvalue_gaps, 'o-', markersize=8, color='red')
    ax2.set_xlabel('Gap Index')
    ax2.set_ylabel('Gap Size')
    ax2.set_title('Spectral Gaps')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def spectral_embedding_2d(eigenvectors, eigenvalues, adjacency, dims=[1, 2]):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = eigenvectors[:, dims[0]]
    y = eigenvectors[:, dims[1]]
    
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    
    scatter = ax.scatter(x, y, c=eigenvectors[:, 1], cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label=f'Fiedler Vector (λ = {eigenvalues[1]:.4f})')
    for i in range(len(adjacency)):
        for j in range(i+1, len(adjacency)):
            if adjacency[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.3)
    
    n = len(x)
    for i in range(n):
        ax.annotate(f"{i}", (x[i], y[i]), fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    ax.set_title(f'2D Spectral Embedding: Eigenvectors {dims[0]} and {dims[1]}')
    ax.set_xlabel(f'Eigenvector {dims[0]} (λ = {eigenvalues[dims[0]]:.4f})')
    ax.set_ylabel(f'Eigenvector {dims[1]} (λ = {eigenvalues[dims[1]]:.4f})')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def spectral_embedding_3d(eigenvectors, eigenvalues, adjacency, dims=[1, 2, 3]):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = eigenvectors[:, dims[0]]
    y = eigenvectors[:, dims[1]]
    z = eigenvectors[:, dims[2]]
    
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    z = z / np.max(np.abs(z))
    
    colors = cm.viridis(Normalize()(eigenvectors[:, 1]))
    scatter = ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
    for i in range(len(adjacency)):
        for j in range(i+1, len(adjacency)):
            if adjacency[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-', alpha=0.1)
    
    n = len(x)
    for i in range(n):
        ax.text(x[i], y[i], z[i], f"{i}", size=8, zorder=1, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    ax.set_title(f'3D Spectral Embedding: Eigenvectors {dims[0]}, {dims[1]}, and {dims[2]}')
    ax.set_xlabel(f'Eigenvector {dims[0]} (λ = {eigenvalues[dims[0]]:.4f})')
    ax.set_ylabel(f'Eigenvector {dims[1]} (λ = {eigenvalues[dims[1]]:.4f})')
    ax.set_zlabel(f'Eigenvector {dims[2]} (λ = {eigenvalues[dims[2]]:.4f})')
    
    plt.tight_layout()
    return fig

def plot_eigenvectors_heatmap(eigenvectors, eigenvalues, grid_size, indices=[1, 2, 3, 4]):
    n = grid_size
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 4))
    
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        eigvec = eigenvectors[:, idx].reshape(n, n)
        im = axes[i].imshow(eigvec, cmap='viridis')
        axes[i].set_title(f"λ = {eigenvalues[idx]:.4f}")
        plt.colorbar(im, ax=axes[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    return fig

def multi_dim_spectral_embedding(eigenvectors, eigenvalues, adjacency):
    fig = plt.figure(figsize=(16, 12))
    
    # 2D embeddings
    ax1 = fig.add_subplot(2, 2, 1)
    x1, y1 = eigenvectors[:, 1], eigenvectors[:, 2]
    scatter1 = ax1.scatter(x1, y1, c=eigenvectors[:, 1], cmap='viridis', s=50, alpha=0.8)
    
    n = len(x1)
    for i in range(n):
        ax1.annotate(f"{i}", (x1[i], y1[i]), fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    ax1.set_title(f'Eigenvectors 1-2 (λ = {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f})')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = fig.add_subplot(2, 2, 2)
    x2, y2 = eigenvectors[:, 2], eigenvectors[:, 3]
    scatter2 = ax2.scatter(x2, y2, c=eigenvectors[:, 1], cmap='viridis', s=50, alpha=0.8)
    
    for i in range(n):
        ax2.annotate(f"{i}", (x2[i], y2[i]), fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    ax2.set_title(f'Eigenvectors 2-3 (λ = {eigenvalues[2]:.4f}, {eigenvalues[3]:.4f})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3D embeddings
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    x3, y3, z3 = eigenvectors[:, 1], eigenvectors[:, 2], eigenvectors[:, 3]
    scatter3 = ax3.scatter(x3, y3, z3, c=eigenvectors[:, 1], cmap='viridis', s=50, alpha=0.8)
    
    for i in range(n):
        ax3.text(x3[i], y3[i], z3[i], f"{i}", size=8, zorder=1, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    ax3.set_title(f'Eigenvectors 1-2-3')
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    x4, y4, z4 = eigenvectors[:, 2], eigenvectors[:, 3], eigenvectors[:, 4]
    scatter4 = ax4.scatter(x4, y4, z4, c=eigenvectors[:, 1], cmap='viridis', s=50, alpha=0.8)
    
    for i in range(n):
        ax4.text(x4[i], y4[i], z4[i], f"{i}", size=8, zorder=1, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, pad=1))
            
    ax4.set_title(f'Eigenvectors 2-3-4')
    
    for i in range(len(adjacency)):
        for j in range(i+1, len(adjacency)):
            if adjacency[i, j] > 0:
                ax3.plot([x3[i], x3[j]], [y3[i], y3[j]], [z3[i], z3[j]], 'k-', alpha=0.05)
                ax4.plot([x4[i], x4[j]], [y4[i], y4[j]], [z4[i], z4[j]], 'k-', alpha=0.05)
    
    plt.tight_layout()
    return fig

def visualize_laplacian_spectrum(grid_size=10, normalized=True, top_k=10):
    adjacency = create_grid_graph_adjacency(grid_size)
    laplacian = compute_laplacian(adjacency, normalized=normalized)
    
    eigenvalues, eigenvectors = compute_eigendecomposition(laplacian)
    
    visualizations = {}
    
    # plot spectrum and spectral gaps
    visualizations['spectrum'] = plot_spectrum(eigenvalues)
    
    # plot top k eigenvectors as 3D surfaces
    visualizations['top_eigenvectors_3d'] = plot_top_eigenvector_3d_surfaces(
        eigenvectors, eigenvalues, grid_size, top_k=top_k)
    
    # plot eigenvectors as heatmaps
    visualizations['eigenvectors_heatmap'] = plot_eigenvectors_heatmap(
        eigenvectors, eigenvalues, grid_size, indices=list(range(1, min(9, eigenvectors.shape[1]))))
    
    # 2D spectral embeddings
    visualizations['spectral_embedding_2d_1_2'] = spectral_embedding_2d(
        eigenvectors, eigenvalues, adjacency, dims=[1, 2])
    
    visualizations['spectral_embedding_2d_2_3'] = spectral_embedding_2d(
        eigenvectors, eigenvalues, adjacency, dims=[2, 3])
    
    # 3D spectral embeddings
    visualizations['spectral_embedding_3d_1_2_3'] = spectral_embedding_3d(
        eigenvectors, eigenvalues, adjacency, dims=[1, 2, 3])
    
    visualizations['spectral_embedding_3d_2_3_4'] = spectral_embedding_3d(
        eigenvectors, eigenvalues, adjacency, dims=[2, 3, 4])
    
    # multiple spectral embeddings in one figure
    visualizations['multi_spectral_embedding'] = multi_dim_spectral_embedding(
        eigenvectors, eigenvalues, adjacency)
    
    return eigenvalues, eigenvectors, adjacency, visualizations

def main(grid_size=10, normalized=True, top_k=10, show_plots=True):
    eigenvalues, eigenvectors, adjacency, visualizations = visualize_laplacian_spectrum(
        grid_size, normalized, top_k)
    
    for i, ev in enumerate(eigenvalues):
        print(f"λ{i} = {ev:.6f}")
    
    if show_plots:
        for fig in visualizations.values():
            plt.figure(fig.number)
            plt.show()
    
    return eigenvalues, eigenvectors, adjacency, visualizations

if __name__ == "__main__":
    grid_size = 5
    normalized = True
    top_k = 10
    
    main(grid_size, normalized, top_k)