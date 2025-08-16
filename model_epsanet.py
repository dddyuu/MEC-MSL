from graph.utils import *
import torch.linalg

class DynamicPSAModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9],conv_groups=[1, 4, 8, 16]):
        super(DynamicPSAModule, self).__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=conv_groups[((k-1) // 2-1)]
            )
            for k in kernel_sizes
        ])
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)

        attention_weights = self.channel_attention(x)
        out = x * attention_weights
        return out


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        self.dp = nn.Dropout(dropout)
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])

            else:
                result += self.gc1[i](x, adj[i])


        result = F.relu(result)
        return result, adj

def matrix_log(spd_matrix):
    eigvals, eigvecs = torch.linalg.eigh(spd_matrix)
    log_eigvals = torch.log(torch.clamp(eigvals, min=1e-6))  # 防止 log(0)
    log_matrix = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)
    return log_matrix


# 矩阵计算几何平均
def geometric_mean(log_matrices):
    mean_log = log_matrices.mean(dim=0)  # 对 Log 空间中的平均
    return mean_log

# 使用矩阵指数操作
def matrix_exp(log_matrix):
    eigvals, eigvecs = torch.linalg.eigh(log_matrix)
    exp_eigvals = torch.exp(eigvals)
    exp_matrix = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2)
    return exp_matrix


class mySPDAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, B=0.4, epsilon=1e-3):
        super(mySPDAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_w = nn.Linear(input_dim, hidden_dim)
        self.k_w = nn.Linear(input_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Learnable temperature
        self.B = nn.Parameter(torch.tensor(B))  # Trainable blending factor
        self.epsilon = epsilon
        self.out_proj = nn.Linear(hidden_dim, input_dim)  # Project back to input_dim

    def log_cholesky(self, A):
        """Compute the Log-Cholesky of SPD matrix A."""
        L = torch.linalg.cholesky(A + torch.eye(A.size(-1), device=A.device) * self.epsilon)
        log_L_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
        log_L = L.clone()
        log_L = log_L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(log_L_diag)
        return log_L

    def exp_cholesky(self, log_L):
        """Compute the SPD matrix from the Log-Cholesky matrix."""
        exp_L = log_L.clone()
        exp_L_diag = torch.exp(torch.diagonal(log_L, dim1=-2, dim2=-1))
        exp_L = exp_L - torch.diag_embed(torch.diagonal(log_L, dim1=-2, dim2=-1)) + torch.diag_embed(exp_L_diag)
        return exp_L @ exp_L.transpose(-1, -2)

    def forward(self, spd_matrices):
        # Ensure the matrix is symmetric
        spd_matrices = (spd_matrices + spd_matrices.transpose(-1, -2)) / 2

        # Step 1: Log-Cholesky
        log_spd = self.log_cholesky(spd_matrices)
        # print(log_spd.shape)
        # Step 2: Multi-head attention mechanism in Log-Cholesky space
        B, N, _ = log_spd.size()

        Q = self.q_w(log_spd).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_w(log_spd).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.einsum("bhid,bhjd->bhij", Q, K) / self.temperature  # Scaled by temperature

        weight = F.softmax(attention_scores, dim=-1)
        weight = (weight + weight.transpose(-1, -2)) / 2

        # print(weight)
        # Step 3: Weighted sum in Log-Cholesky space
        weighted_log_spd = torch.einsum('bhij,bhjd->bhid', weight, Q)
        weighted_log_spd = weighted_log_spd.transpose(1, 2).contiguous().view(B, N, -1)
        # print(weighted_log_spd.shape)
        # print(log_spd.shape)
        updated_log_spd = (1 - torch.sigmoid(self.B)) * weighted_log_spd + torch.sigmoid(self.B) * log_spd

        # Step 4: Exponentiate the result back to SPD space
        updated_spd = self.exp_cholesky(updated_log_spd)

        # Ensure the matrix is symmetric
        updated_spd = (updated_spd + updated_spd.transpose(-1, -2)) / 2
        return updated_spd, weight

class MECMSL(nn.Module):
    def __init__(self, dropout, num_classes=2):
        super(MECMSL, self).__init__()
        self.dropout = dropout
        self.DynamicPSAconv = DynamicPSAModule(64, 64, kernel_sizes=[3, 5, 7, 9])
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.SPDAttention = mySPDAttention(input_dim=64, hidden_dim=64, B=0.69)
        self.gcn = Chebynet([32, 64, 64], 5, 8, 0.4)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(64, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
    def forward(self, x, adj):
        x = x.unsqueeze(3)
        x = self.DynamicPSAconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.squeeze(3)
        x = x.float()
        adj = adj.float()
        adj, weight = self.SPDAttention(adj)
        out, adj_weight = self.gcn(x, adj)
        out = self.avgpool(out)
        out = self.flatten(out)
        x = self.fc2(out)
        return x, adj


def epsanet50_spd():
    model = MECMSL(0.4, num_classes=2)
    return model



