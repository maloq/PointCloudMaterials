import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------
# Stability constants
# ---------------------------
EPS = 1e-6
FEATURE_CLAMP_MAX = 50.0  # Clamp feature magnitudes to prevent explosion


# ---------------------------
# Helpers: kNN + gathering
# ---------------------------

def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)
    returns idx: (B, N, k) of k nearest neighbors (excluding self)
    NOTE: O(N^2) via torch.cdist. For large N, swap in torch_cluster.knn.
    """
    _, N, _ = x.shape
    if N <= 1:
        raise ValueError("kNN requires at least 2 points per batch.")
    k = min(k, N - 1)
    # Compute kNN in float32 for stable neighbor selection under mixed precision.
    x_f = x
    if x_f.dtype in (torch.float16, torch.bfloat16):
        x_f = x_f.float()
    with torch.no_grad():
        dist = torch.cdist(x_f, x_f)  # (B, N, N)
        # exclude self by setting diagonal huge
        eye = torch.eye(N, device=x_f.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
        dist = dist.masked_fill(eye, float("inf"))
        idx = dist.topk(k, largest=False).indices  # (B, N, k)
    return idx


def gather_neighbors(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbor entries along dim=1 (point dimension).
    t:  (B, N, ...)
    idx:(B, N, k)
    out:(B, N, k, ...)
    """
    B, N, k = idx.shape
    batch = torch.arange(B, device=t.device)[:, None, None]  # (B,1,1)
    return t[batch, idx]  # advanced indexing


# ---------------------------
# VN building blocks
# ---------------------------

class VNBatchNorm4D(nn.Module):
    """
    VN Batch Normalization for tensors of shape (B, N, C, 3).
    Normalizes based on the norm of each vector channel, preserving direction.
    """
    def __init__(self, num_features: int, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, 3)
        B, N, C, _ = x.shape
        # Compute norms per channel: (B, N, C)
        norm = torch.linalg.norm(x, dim=-1).clamp_min(EPS)
        # Reshape for BatchNorm1d: (B*N, C)
        norm_flat = norm.view(B * N, C)
        norm_bn = self.bn(norm_flat)  # (B*N, C)
        norm_bn = norm_bn.view(B, N, C, 1)  # (B, N, C, 1)
        norm = norm.unsqueeze(-1)  # (B, N, C, 1)
        # Scale vectors by normalized magnitude, preserving direction
        return x / norm * norm_bn


class VNLayerNorm(nn.Module):
    """
    VN Layer Normalization for tensors of shape (B, N, C, 3).
    Normalizes across the channel dimension while preserving direction.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, 3)
        norm = torch.linalg.norm(x, dim=-1).clamp_min(EPS)  # (B, N, C)
        # Normalize across channel dimension
        mean = norm.mean(dim=-1, keepdim=True)  # (B, N, 1)
        std = norm.std(dim=-1, keepdim=True).clamp_min(EPS)  # (B, N, 1)
        norm_normalized = (norm - mean) / std  # (B, N, C)
        # Apply learnable scale and shift
        norm_scaled = norm_normalized * self.gamma + self.beta  # (B, N, C)
        # Reconstruct vectors
        norm = norm.unsqueeze(-1)  # (B, N, C, 1)
        norm_scaled = norm_scaled.unsqueeze(-1)  # (B, N, C, 1)
        return x / norm * norm_scaled.clamp_min(EPS)


def clamp_features(x: torch.Tensor, max_norm: float = FEATURE_CLAMP_MAX) -> torch.Tensor:
    """Clamp feature vector magnitudes to prevent explosion."""
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


class VNLinear(nn.Module):
    """
    VN linear: mixes vector channels with scalar weights.
    Input:  (B, N, C_in, 3)
    Output: (B, N, C_out, 3)
    """
    def __init__(self, c_in: int, c_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        # Use Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.weight, gain=0.5)
        if bias:
            # A *vector* bias would break SO(3)-equivariance, so we disallow it here.
            raise ValueError("Vector bias breaks rotation equivariance; keep bias=False.")

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B,N,Cin,3), W: (Cout,Cin) -> (B,N,Cout,3)
        return torch.einsum("bncd,oc->bnod", v, self.weight)

def radial_tanh(vec: torch.Tensor, eps: float = EPS, scale: float = 0.1) -> torch.Tensor:
    """
    Bounds displacement magnitudes using tanh.
    Scale parameter controls the maximum displacement magnitude.
    """
    # vec: (..., 3)
    n = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
    return vec * (torch.tanh(n) / n) * scale


class VNReLU(nn.Module):
    """
    VN-ReLU from the Vector Neurons paper:
      q = W v
      k = U v
      if <q,k> >= 0: output q
      else: output q - proj_k(q)

    Shapes:
      in:  (B,N,Cin,3)
      out: (B,N,Cout,3)
    """
    def __init__(self, c_in: int, c_out: Optional[int] = None, eps: float = EPS):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.q_lin = VNLinear(c_in, c_out, bias=False)
        self.k_lin = VNLinear(c_in, c_out, bias=False)
        self.eps = eps

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        q = self.q_lin(v)  # (B,N,Cout,3)
        k = self.k_lin(v)  # (B,N,Cout,3)

        # dot per vector-channel: (B,N,Cout,1)
        dot = (q * k).sum(dim=-1, keepdim=True)

        # safe normalize k
        k_norm = torch.linalg.norm(k, dim=-1, keepdim=True).clamp_min(self.eps)
        k_hat = k / k_norm  # (B,N,Cout,3)

        # projection of q onto k_hat
        proj = (q * k_hat).sum(dim=-1, keepdim=True) * k_hat  # (B,N,Cout,3)

        out = torch.where(dot >= 0.0, q, q - proj)
        return out


class VNMLP(nn.Module):
    """
    Simple VN-MLP: Linear -> VNReLU -> Linear
    """
    def __init__(self, c_in: int, c_hidden: int, c_out: int):
        super().__init__()
        self.lin1 = VNLinear(c_in, c_hidden, bias=False)
        self.act = VNReLU(c_hidden)
        self.lin2 = VNLinear(c_hidden, c_out, bias=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(v)))


# ---------------------------
# VN Equivariant "Skip-Transformer" attention
# ---------------------------

class ScalarEdgeMLP(nn.Module):
    """
    Small MLP for attention logits from scalar invariants.
    """
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B,N,k,in_dim) -> (B,N,k,1)
        return self.net(s)


class VNInvariantAttention(nn.Module):
    """
    Equivariant message passing:
      Q = Wq V, K = Wk V, U = Wv V
      e_{jℓ} = MLP( ||Q_j||, ||K_ℓ||, <Q_j,K_ℓ>, ||x_j-x_ℓ|| )
      a = softmax(e / temperature) over neighbors
      H_j = V_j + residual_scale * sum_ℓ a_{jℓ} U_ℓ

    All logits are scalars built from invariants, so a is invariant.
    Message is scalar-weighted sum of vectors, so equivariant.

    Input:  x (B,N,3), v (B,N,C,3)
    Output: h (B,N,C_out,3)
    """
    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        k: int = 16, 
        mlp_hidden: int = 32,
        temperature: float = 1.0,
        residual_scale: float = 0.5,
        use_layer_norm: bool = True,
        feature_clamp_max: float = FEATURE_CLAMP_MAX,
    ):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.residual_scale = residual_scale
        self.feature_clamp_max = feature_clamp_max
        
        self.q = VNLinear(c_in, c_out, bias=False)
        self.k_lin = VNLinear(c_in, c_out, bias=False)
        self.u = VNLinear(c_in, c_out, bias=False)
        self.edge_mlp = ScalarEdgeMLP(in_dim=4, hidden=mlp_hidden)
        
        # Optional layer norm for stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = VNLayerNorm(c_out)

    @staticmethod
    def _reduce_invariant(v: torch.Tensor) -> torch.Tensor:
        """
        v: (B,N,C,3) -> (B,N,1) scalar invariant: mean of channel norms (not sum, for stability)
        """
        # channelwise norms: (B,N,C)
        n = torch.linalg.norm(v, dim=-1)
        return n.mean(dim=-1, keepdim=True)  # (B,N,1) - use mean instead of sum

    @staticmethod
    def _reduce_dot(a: torch.Tensor, b: torch.Tensor, c_out: int) -> torch.Tensor:
        """
        a,b: (B,N,C,3) -> (B,N,1) scalar invariant: mean of channel dot products (normalized)
        """
        d = (a * b).sum(dim=-1)  # (B,N,C)
        return d.mean(dim=-1, keepdim=True)  # (B,N,1) - use mean instead of sum

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, C, _ = v.shape

        idx = knn_indices(x, self.k)                     # (B,N,k)
        x_nbr = gather_neighbors(x, idx)                 # (B,N,k,3)

        Q = self.q(v)                                    # (B,N,Cout,3)
        K = self.k_lin(v)                                # (B,N,Cout,3)
        U = self.u(v)                                    # (B,N,Cout,3)
        
        C_out = Q.shape[2]

        K_nbr = gather_neighbors(K, idx)                 # (B,N,k,Cout,3)
        U_nbr = gather_neighbors(U, idx)                 # (B,N,k,Cout,3)

        # invariants (normalized for stability)
        qn = self._reduce_invariant(Q)                   # (B,N,1)
        kn = self._reduce_invariant(K)                   # (B,N,1)
        kn_nbr = gather_neighbors(kn, idx)               # (B,N,k,1)

        dot_nbr = self._reduce_dot(Q.unsqueeze(2), K_nbr, C_out)  # (B,N,k,1)

        dist = torch.linalg.norm(x.unsqueeze(2) - x_nbr, dim=-1, keepdim=True)  # (B,N,k,1)

        # assemble scalar features for each edge (j,ℓ)
        edge_s = torch.cat([qn.unsqueeze(2).expand(-1, -1, self.k, -1),
                            kn_nbr,
                            dot_nbr,
                            dist], dim=-1)  # (B,N,k,4)

        logits = self.edge_mlp(edge_s)                   # (B,N,k,1)
        
        # Temperature scaling for numerical stability
        logits = logits / self.temperature
        # Clamp logits to prevent extreme values
        logits = logits.clamp(-10, 10)
        
        attn = torch.softmax(logits, dim=2)              # (B,N,k,1)

        # message: sum over neighbors of scalar * vector
        msg = (attn.unsqueeze(-1) * U_nbr).sum(dim=2)    # (B,N,Cout,3)

        # Scaled residual connection to prevent magnitude growth
        out = Q + self.residual_scale * msg
        
        # Optional layer normalization
        if self.use_layer_norm:
            out = self.layer_norm(out)
        
        # Clamp features to prevent explosion
        out = clamp_features(out, max_norm=self.feature_clamp_max)
        
        return out


# ---------------------------
# VN-equivariant SPD-like block
# ---------------------------

class VNSnowflakeDeconvBlock(nn.Module):
    """
    VN-equivariant "SPD-like" upsampling block:
      1) equivariant attention to get context H
      2) split into r children with r VNLinear heads
      3) displacement predicted by VN-MLP (vector), added to duplicated coords

    Input:
      x: (B,N,3)
      v: (B,N,C,3)
    Output:
      x_up: (B, rN, 3)
      v_up: (B, rN, C_child, 3)
    """
    def __init__(
        self,
        c_in: int,
        c_ctx: int,
        c_child: int,
        up_factor: int = 2,
        k: int = 16,
        attn_mlp_hidden: int = 32,
        disp_hidden: int = 64,
        temperature: float = 1.0,
        residual_scale: float = 0.5,
        disp_scale: float = 0.1,
        use_batch_norm: bool = True,
        feature_clamp_max: float = FEATURE_CLAMP_MAX,
    ):
        super().__init__()
        self.r = up_factor
        self.disp_scale = disp_scale
        self.feature_clamp_max = feature_clamp_max

        # "skip-transformer" analog with stability improvements
        self.ctx = VNInvariantAttention(
            c_in=c_in, 
            c_out=c_ctx, 
            k=k, 
            mlp_hidden=attn_mlp_hidden,
            temperature=temperature,
            residual_scale=residual_scale,
            use_layer_norm=True,
            feature_clamp_max=feature_clamp_max,
        )

        # r splitting heads (fixed, equivariant)
        self.split_heads = nn.ModuleList([VNLinear(c_ctx, c_child, bias=False) for _ in range(self.r)])
        
        # Batch normalization after splitting
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = VNBatchNorm4D(c_child)

        # displacement MLP: produce 1 vector-channel then squeeze to (B,rN,3)
        self.disp = VNMLP(c_in=c_child, c_hidden=disp_hidden, c_out=1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape

        # 1) context with attention
        h = self.ctx(x, v)  # (B,N,Cctx,3)

        # 2) split features into r children
        children = []
        for head in self.split_heads:
            children.append(head(h))  # each: (B,N,Cchild,3)
        v_child = torch.stack(children, dim=2)           # (B,N,r,Cchild,3)
        v_up = v_child.reshape(B, N * self.r, v_child.size(3), 3)  # (B,rN,Cchild,3)
        
        # Apply batch normalization for stability
        if self.use_batch_norm:
            v_up = self.bn(v_up)
        
        # Clamp features to prevent explosion
        v_up = clamp_features(v_up, max_norm=self.feature_clamp_max)

        # 3) duplicate coords and predict displacements
        x_dup = x.unsqueeze(2).expand(B, N, self.r, 3).reshape(B, N * self.r, 3)  # (B,rN,3)

        disp_v = self.disp(v_up)                         # (B,rN,1,3)
        delta = disp_v.squeeze(2)                        # (B,rN,3)

        # Use radial_tanh with configurable scale for bounded displacements
        x_up = x_dup + radial_tanh(delta, scale=self.disp_scale)
        return x_up, v_up


# ---------------------------
# Example: a multi-stage decoder
# ---------------------------

class VNSnowflakeDecoder(nn.Module):
    """
    Stacks multiple VNSnowflakeDeconvBlock blocks with stability improvements.
    """
    def __init__(
        self,
        c_in: int,
        stages: Tuple[Tuple[int, int, int, int], ...],
        k: int = 16,
        temperature: float = 1.0,
        residual_scale: float = 0.5,
        disp_scale: float = 0.1,
        use_batch_norm: bool = True,
        feature_clamp_max: float = FEATURE_CLAMP_MAX,
    ):
        """
        Args:
            c_in: Input channels
            stages: tuple of (c_ctx, c_child, up_factor, disp_hidden) per stage
            k: Number of neighbors for kNN attention
            temperature: Temperature for attention softmax (higher = smoother attention)
            residual_scale: Scale factor for residual connections (lower = more stable)
            disp_scale: Scale factor for displacement predictions
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.feature_clamp_max = feature_clamp_max
        blocks = []
        cur_c = c_in
        for i, (c_ctx, c_child, up_factor, disp_hidden) in enumerate(stages):
            # Use progressively smaller displacement scale in later stages
            stage_disp_scale = disp_scale / (1.0 + 0.5 * i)
            blocks.append(
                VNSnowflakeDeconvBlock(
                    c_in=cur_c,
                    c_ctx=c_ctx,
                    c_child=c_child,
                    up_factor=up_factor,
                    k=k,
                    disp_hidden=disp_hidden,
                    temperature=temperature,
                    residual_scale=residual_scale,
                    disp_scale=stage_disp_scale,
                    use_batch_norm=use_batch_norm,
                    feature_clamp_max=feature_clamp_max,
                )
            )
            cur_c = c_child
        self.blocks = nn.ModuleList(blocks)
        self.out_c = cur_c

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x, v = blk(x, v)
            # Clamp outputs at each stage for safety
            v = clamp_features(v, max_norm=self.feature_clamp_max)
        return x, v


# ---------------------------
# Optional: quick equivariance sanity check
# ---------------------------

def random_rotation(device=None, dtype=None) -> torch.Tensor:
    """
    Returns R in SO(3) using QR decomposition.
    """
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # enforce det(Q)=+1
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@torch.no_grad()
def sanity_check_equivariance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, C = 2, 64, 16
    x = torch.randn(B, N, 3, device=device)
    v = torch.randn(B, N, C, 3, device=device)

    dec = VNSnowflakeDecoder(
        c_in=C,
        stages=((32, 32, 2, 64), (64, 32, 2, 64)),
        k=63,
    ).to(device).eval()

    R = random_rotation(device=device, dtype=x.dtype)  # (3,3)
    with torch.no_grad():
        idx1 = knn_indices(x, k=8)
        idx2 = knn_indices(x @ R, k=8)
        same = (idx1 == idx2).all(dim=-1)  # (B,N) whether all k neighbors match
        print("fraction points with identical kNN:", same.float().mean().item())
    x1, v1 = dec(x, v)
    x2, v2 = dec(x @ R, v @ R)

    # equivariance means outputs rotate: x2 ≈ x1 R, v2 ≈ v1 R
    err_x = (x2 - (x1 @ R)).abs().max().item()
    err_v = (v2 - (v1 @ R)).abs().max().item()
    print("max |x2 - x1R|:", err_x)
    print("max |v2 - v1R|:", err_v)


@torch.no_grad()
def plot_equivariance_vs_k(k_values=None, num_trials=5, save_path=None):
    """
    Plot equivariance errors (err_x and err_v) for different k values.
    
    Args:
        k_values: List of k values to test. Defaults to range(1, 65).
        num_trials: Number of trials to average over for each k.
        save_path: If provided, save the plot to this path.
    """
    if k_values is None:
        k_values = list(range(1, 65))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, C = 2, 64, 16
    
    err_x_list = []
    err_v_list = []
    
    for k in k_values:
        print(f"Testing k={k}...")
        
        # Average over multiple trials
        trial_err_x = []
        trial_err_v = []
        
        for _ in range(num_trials):
            x = torch.randn(B, N, 3, device=device)
            v = torch.randn(B, N, C, 3, device=device)
            
            dec = VNSnowflakeDecoder(
                c_in=C,
                stages=((32, 32, 2, 64), (64, 32, 2, 64)),
                k=k,
            ).to(device).eval()
            
            R = random_rotation(device=device, dtype=x.dtype)
            
            x1, v1 = dec(x, v)
            x2, v2 = dec(x @ R, v @ R)
            
            err_x = (x2 - (x1 @ R)).abs().max().item()
            err_v = (v2 - (v1 @ R)).abs().max().item()
            
            trial_err_x.append(err_x)
            trial_err_v.append(err_v)
        
        # Average errors over trials
        err_x_list.append(sum(trial_err_x) / num_trials)
        err_v_list.append(sum(trial_err_v) / num_trials)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, err_x_list, 'b-o', label='err_x (max |x2 - x1R|)', markersize=4)
    ax.plot(k_values, err_v_list, 'r-s', label='err_v (max |v2 - v1R|)', markersize=4)
    
    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('Equivariance Error', fontsize=12)
    ax.set_title('Equivariance Error vs. Number of Neighbors (k)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often helps visualize errors
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return k_values, err_x_list, err_v_list


if __name__ == "__main__":
    plot_equivariance_vs_k(
        k_values=list(range(1, 65)),
        num_trials=10,  # more trials for smoother results
        save_path="equivariance_vs_k.png"
    )
