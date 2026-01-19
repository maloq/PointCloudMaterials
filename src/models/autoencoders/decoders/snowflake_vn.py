import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------
# Helpers: kNN + gathering
# ---------------------------

def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)
    returns idx: (B, N, k) of k nearest neighbors (excluding self)
    NOTE: O(N^2) via torch.cdist. For large N, swap in torch_cluster.knn.
    """
    B, N, _ = x.shape
    dist = torch.cdist(x, x)  # (B, N, N)
    # exclude self by setting diagonal huge
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
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

class VNLinear(nn.Module):
    """
    VN linear: mixes vector channels with scalar weights.
    Input:  (B, N, C_in, 3)
    Output: (B, N, C_out, 3)
    """
    def __init__(self, c_in: int, c_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            # A *vector* bias would break SO(3)-equivariance, so we disallow it here.
            raise ValueError("Vector bias breaks rotation equivariance; keep bias=False.")

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B,N,Cin,3), W: (Cout,Cin) -> (B,N,Cout,3)
        return torch.einsum("bncd,oc->bnod", v, self.weight)

def radial_tanh(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # vec: (..., 3)
    n = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
    return vec * (torch.tanh(n) / n)

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
    def __init__(self, c_in: int, c_out: Optional[int] = None, eps: float = 1e-8):
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
      a = softmax(e) over neighbors
      H_j = V_j + sum_ℓ a_{jℓ} U_ℓ

    All logits are scalars built from invariants, so a is invariant.
    Message is scalar-weighted sum of vectors, so equivariant.

    Input:  x (B,N,3), v (B,N,C,3)
    Output: h (B,N,C_out,3)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 16, mlp_hidden: int = 32):
        super().__init__()
        self.k = k
        self.q = VNLinear(c_in, c_out, bias=False)
        self.k_lin = VNLinear(c_in, c_out, bias=False)
        self.u = VNLinear(c_in, c_out, bias=False)
        self.edge_mlp = ScalarEdgeMLP(in_dim=4, hidden=mlp_hidden)

    @staticmethod
    def _reduce_invariant(v: torch.Tensor) -> torch.Tensor:
        """
        v: (B,N,C,3) -> (B,N,1) scalar invariant: sum of channel norms
        """
        # channelwise norms: (B,N,C)
        n = torch.linalg.norm(v, dim=-1)
        return n.sum(dim=-1, keepdim=True)  # (B,N,1)

    @staticmethod
    def _reduce_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a,b: (B,N,C,3) -> (B,N,1) scalar invariant: sum of channel dot products
        """
        d = (a * b).sum(dim=-1)  # (B,N,C)
        return d.sum(dim=-1, keepdim=True)  # (B,N,1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, _, _ = v.shape

        idx = knn_indices(x, self.k)                     # (B,N,k)
        x_nbr = gather_neighbors(x, idx)                 # (B,N,k,3)
        v_nbr = gather_neighbors(v, idx)                 # (B,N,k,C,3)

        Q = self.q(v)                                    # (B,N,Cout,3)
        K = self.k_lin(v)                                # (B,N,Cout,3)
        U = self.u(v)                                    # (B,N,Cout,3)

        K_nbr = gather_neighbors(K, idx)                 # (B,N,k,Cout,3)
        U_nbr = gather_neighbors(U, idx)                 # (B,N,k,Cout,3)

        # invariants
        qn = self._reduce_invariant(Q)                   # (B,N,1)
        kn = self._reduce_invariant(K)                   # (B,N,1)
        kn_nbr = gather_neighbors(kn, idx)               # (B,N,k,1)

        dot_nbr = self._reduce_dot(Q.unsqueeze(2), K_nbr)  # (B,N,k,1) via broadcasting

        dist = torch.linalg.norm(x.unsqueeze(2) - x_nbr, dim=-1, keepdim=True)  # (B,N,k,1)

        # assemble scalar features for each edge (j,ℓ)
        edge_s = torch.cat([qn.unsqueeze(2).expand(-1, -1, self.k, -1),
                            kn_nbr,
                            dot_nbr,
                            dist], dim=-1)  # (B,N,k,4)

        logits = self.edge_mlp(edge_s)                   # (B,N,k,1)
        attn = torch.softmax(logits, dim=2)              # (B,N,k,1)

        # message: sum over neighbors of scalar * vector
        msg = (attn.unsqueeze(-1) * U_nbr).sum(dim=2)    # (B,N,Cout,3)

        # residual
        return Q + msg


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
    ):
        super().__init__()
        self.r = up_factor

        # "skip-transformer" analog
        self.ctx = VNInvariantAttention(c_in=c_in, c_out=c_ctx, k=k, mlp_hidden=attn_mlp_hidden)

        # r splitting heads (fixed, equivariant)
        self.split_heads = nn.ModuleList([VNLinear(c_ctx, c_child, bias=False) for _ in range(self.r)])

        # displacement MLP: produce 1 vector-channel then squeeze to (B,rN,3)
        self.disp = VNMLP(c_in=c_child, c_hidden=disp_hidden, c_out=1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape

        # 1) context
        h = self.ctx(x, v)  # (B,N,Cctx,3)

        # 2) split features into r children
        children = []
        for head in self.split_heads:
            children.append(head(h))  # each: (B,N,Cchild,3)
        v_child = torch.stack(children, dim=2)           # (B,N,r,Cchild,3)
        v_up = v_child.reshape(B, N * self.r, v_child.size(3), 3)  # (B,rN,Cchild,3)

        # 3) duplicate coords and predict displacements
        x_dup = x.unsqueeze(2).expand(B, N, self.r, 3).reshape(B, N * self.r, 3)  # (B,rN,3)

        disp_v = self.disp(v_up)                         # (B,rN,1,3)
        delta = disp_v.squeeze(2)                        # (B,rN,3)

        x_up = x_dup + radial_tanh(delta)                 # tanh optional for stability
        return x_up, v_up


# ---------------------------
# Example: a multi-stage decoder
# ---------------------------

class VNSnowflakeDecoder(nn.Module):
    """
    Stacks multiple VNSnowflakeDeconvBlock blocks.
    """
    def __init__(
        self,
        c_in: int,
        stages: Tuple[Tuple[int, int, int, int], ...],
        k: int = 16,
    ):
        """
        stages: tuple of (c_ctx, c_child, up_factor, disp_hidden) per stage
        """
        super().__init__()
        blocks = []
        cur_c = c_in
        for (c_ctx, c_child, up_factor, disp_hidden) in stages:
            blocks.append(
                VNSnowflakeDeconvBlock(
                    c_in=cur_c,
                    c_ctx=c_ctx,
                    c_child=c_child,
                    up_factor=up_factor,
                    k=k,
                    disp_hidden=disp_hidden,
                )
            )
            cur_c = c_child
        self.blocks = nn.ModuleList(blocks)
        self.out_c = cur_c

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x, v = blk(x, v)
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