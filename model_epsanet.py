"""
MECMSL: Multi-scale EEG Classification Model with SPD Learning
A deep learning model for EEG signal classification using dynamic convolutions,
SPD (Symmetric Positive Definite) attention, and graph convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from graph.layers import GraphConvolution
from graph.utils import generate_cheby_adj


class DynamicPSAModule(nn.Module):
    """
    Dynamic Pyramid Squeeze and Attention Module
    Multi-scale feature extraction with channel attention mechanism
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=None, conv_groups=None):
        super(DynamicPSAModule, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]
        if conv_groups is None:
            conv_groups = [1, 4, 8, 16]

        self.num_branches = len(kernel_sizes)
        assert out_channels % self.num_branches == 0, \
            f"out_channels ({out_channels}) must be divisible by number of branches ({self.num_branches})"

        branch_channels = out_channels // self.num_branches

        # Multi-scale convolution branches
        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=conv_groups[i] if i < len(conv_groups) else 1
            )
            for i, k in enumerate(kernel_sizes)
        ])

        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 16, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 16, 1), out_channels, kernel_size=1),
            nn.Sigmoid()  # Changed from Softmax to Sigmoid for better gradient flow
        )

    def forward(self, x):
        # Multi-scale feature extraction
        branch_features = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_features, dim=1)

        # Apply channel attention
        attention_weights = self.channel_attention(concatenated)
        output = concatenated * attention_weights

        return output


class ChebyNet(nn.Module):
    """
    Chebyshev Graph Convolutional Network
    Graph convolution using Chebyshev polynomials for spectral filtering
    """

    def __init__(self, input_dim, K, output_dim, dropout=0.5):
        super(ChebyNet, self).__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout)

        # Chebyshev polynomial graph convolution layers
        self.graph_convs = nn.ModuleList([
            GraphConvolution(input_dim, output_dim)
            for _ in range(K)
        ])

    def forward(self, x, adjacency_matrix):
        """
        Forward pass through Chebyshev GCN

        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            adjacency_matrix: Graph adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            output: Graph convolution output
            chebyshev_adj: Chebyshev adjacency matrices
        """
        chebyshev_adj = generate_cheby_adj(adjacency_matrix, self.K)

        # Aggregate features from all Chebyshev polynomial orders
        output = None
        for i, conv_layer in enumerate(self.graph_convs):
            if output is None:
                output = conv_layer(x, chebyshev_adj[i])
            else:
                output += conv_layer(x, chebyshev_adj[i])

        output = F.relu(output)
        output = self.dropout(output)

        return output, chebyshev_adj


class SPDAttentionModule(nn.Module):
    """
    SPD (Symmetric Positive Definite) Attention Module
    Attention mechanism operating in the SPD manifold using Log-Cholesky parameterization
    """

    def __init__(self, input_dim, hidden_dim, num_heads=4, blend_factor=0.4, epsilon=1e-6):
        super(SPDAttentionModule, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.epsilon = epsilon

        assert hidden_dim % num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        # Query and Key projection layers
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Learnable parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.blend_factor = nn.Parameter(torch.tensor(blend_factor))

    def _log_cholesky_decomposition(self, spd_matrix):
        """Compute Log-Cholesky decomposition of SPD matrix"""
        # Add small diagonal regularization for numerical stability
        regularized_matrix = spd_matrix + torch.eye(
            spd_matrix.size(-1),
            device=spd_matrix.device
        ) * self.epsilon

        # Cholesky decomposition
        cholesky_factor = torch.linalg.cholesky(regularized_matrix)

        # Log of diagonal elements
        diagonal_elements = torch.diagonal(cholesky_factor, dim1=-2, dim2=-1)
        log_diagonal = torch.log(torch.clamp(diagonal_elements, min=self.epsilon))

        # Construct Log-Cholesky matrix
        log_cholesky = cholesky_factor.clone()
        diag_indices = torch.arange(cholesky_factor.size(-1))
        log_cholesky[..., diag_indices, diag_indices] = log_diagonal

        return log_cholesky

    def _exp_cholesky_reconstruction(self, log_cholesky):
        """Reconstruct SPD matrix from Log-Cholesky representation"""
        exp_cholesky = log_cholesky.clone()

        # Exponentiate diagonal elements
        diagonal_elements = torch.diagonal(log_cholesky, dim1=-2, dim2=-1)
        exp_diagonal = torch.exp(diagonal_elements)

        diag_indices = torch.arange(log_cholesky.size(-1))
        exp_cholesky[..., diag_indices, diag_indices] = exp_diagonal

        # Reconstruct SPD matrix: L @ L^T
        spd_matrix = exp_cholesky @ exp_cholesky.transpose(-1, -2)

        return spd_matrix

    def forward(self, spd_matrices):
        """
        Forward pass through SPD attention

        Args:
            spd_matrices: Input SPD matrices [batch_size, num_nodes, num_nodes]

        Returns:
            updated_spd: Updated SPD matrices after attention
            attention_weights: Computed attention weights
        """
        # Ensure input is symmetric
        spd_matrices = (spd_matrices + spd_matrices.transpose(-1, -2)) / 2

        # Transform to Log-Cholesky space
        log_cholesky = self._log_cholesky_decomposition(spd_matrices)

        # Multi-head attention in Log-Cholesky space
        batch_size, num_nodes, _ = log_cholesky.size()

        # Project to query and key spaces
        queries = self.query_proj(log_cholesky).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)

        keys = self.key_proj(log_cholesky).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.einsum("bhid,bhjd->bhij", queries, keys) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Ensure attention weights are symmetric
        attention_weights = (attention_weights + attention_weights.transpose(-1, -2)) / 2

        # Apply attention to queries
        attended_features = torch.einsum('bhij,bhjd->bhid', attention_weights, queries)
        attended_features = attended_features.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, -1
        )

        # Blend original and attended features
        blend_weight = torch.sigmoid(self.blend_factor)
        updated_log_cholesky = (1 - blend_weight) * attended_features + blend_weight * log_cholesky

        # Transform back to SPD space
        updated_spd = self._exp_cholesky_reconstruction(updated_log_cholesky)

        # Ensure output is symmetric
        updated_spd = (updated_spd + updated_spd.transpose(-1, -2)) / 2

        return updated_spd, attention_weights


class MECMSL(nn.Module):
    """
    MECMSL: Multi-scale EEG Classification Model with SPD Learning

    Architecture:
    1. Dynamic PSA Module for multi-scale feature extraction
    2. SPD Attention for adjacency matrix refinement
    3. Chebyshev GCN for graph-based feature learning
    4. Classification head
    """

    def __init__(self, num_classes=2, dropout=0.4, hidden_dim=64):
        super(MECMSL, self).__init__()

        self.hidden_dim = hidden_dim

        # Multi-scale convolutional feature extraction
        self.dynamic_psa = DynamicPSAModule(
            in_channels=hidden_dim,
            out_channels=hidden_dim
        )

        # Normalization and activation
        self.batch_norm = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # SPD attention for adjacency matrix processing
        self.spd_attention = SPDAttentionModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            blend_factor=0.69
        )

        # Graph convolutional network
        self.graph_conv = ChebyNet(
            input_dim=hidden_dim,
            K=5,
            output_dim=8,
            dropout=dropout
        )

        # Classification layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(8 * 8, num_classes)  # Adjusted for 3D output

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg_signal, adjacency_matrix):
        """
        Forward pass through MECMSL model

        Args:
            eeg_signal: EEG signal data [batch_size, channels, time_points]
            adjacency_matrix: Graph adjacency matrix [batch_size, channels, channels]

        Returns:
            logits: Classification logits [batch_size, num_classes]
            updated_adjacency: Updated adjacency matrix after SPD attention
        """
        # Multi-scale feature extraction
        # Add channel dimension for 2D convolution
        x = eeg_signal.unsqueeze(-1)  # [batch_size, channels, time_points, 1]

        # Dynamic PSA convolution
        x = self.dynamic_psa(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Remove the last dimension and prepare for GCN
        x = x.squeeze(-1)  # [batch_size, channels, reduced_time_points]

        # Ensure correct data types
        x = x.float()
        adjacency_matrix = adjacency_matrix.float()

        # SPD attention on adjacency matrix
        updated_adjacency, attention_weights = self.spd_attention(adjacency_matrix)

        # Graph convolution
        graph_features, chebyshev_adj = self.graph_conv(x, updated_adjacency)

        # Global pooling and classification
        # graph_features = graph_features.unsqueeze(-1)  # Add dimension for adaptive pooling
        pooled_features = self.adaptive_pool(graph_features)
        flattened_features = self.flatten(pooled_features)
        flattened_features = self.dropout(flattened_features)

        logits = self.classifier(flattened_features)

        return logits, updated_adjacency


def create_mecmsl_model(num_classes=2, dropout=0.4, hidden_dim=64):
    """
    Factory function to create MECMSL model

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate for regularization
        hidden_dim: Hidden dimension size

    Returns:
        MECMSL model instance
    """
    return MECMSL(num_classes=num_classes, dropout=dropout, hidden_dim=hidden_dim)


# Alias for backward compatibility
def epsanet50_spd():
    """Legacy function name for creating MECMSL model"""
    return create_mecmsl_model(num_classes=2, dropout=0.4, hidden_dim=64)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_mecmsl_model()

    # Create dummy input
    batch_size = 2
    channels = 64
    time_points = 128

    eeg_data = torch.randn(batch_size, channels, time_points)
    adjacency = torch.randn(batch_size, channels, channels)

    # Make adjacency matrix symmetric and positive definite
    adjacency = adjacency @ adjacency.transpose(-1, -2)
    adjacency = adjacency + torch.eye(channels) * 0.1

    # Forward pass
    with torch.no_grad():
        logits, updated_adj = model(eeg_data, adjacency)

    print(f"Model created successfully!")
    print(f"Input EEG shape: {eeg_data.shape}")
    print(f"Input adjacency shape: {adjacency.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Updated adjacency shape: {updated_adj.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")