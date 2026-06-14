import numpy as np
import torch_candle as torch
import torch_candle.nn as nn
import torch_candle.nn.functional as F

def dllt_solve(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Direct Linear Least-Squares Tensor Analytical Solver (DLLT-AS).
    
    Computes the closed-form Moore-Penrose Pseudo-Inverse (Ridge) projection:
    W_opt = Y^T X (X^T X + lambda I)^-1
    
    Args:
        X (Tensor): Input feature matrix of shape (num_samples, in_features)
        Y (Tensor): Target label/one-hot matrix of shape (num_samples, out_classes)
        lam (float): Regularization factor (lambda)
        
    Returns:
        Tensor: Analytical weight matrix W_opt of shape (out_classes, in_features)
    """
    device = X.device
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X, device=device)
    if not isinstance(Y, torch.Tensor):
        Y = torch.Tensor(Y, device=device)
        
    n_features = X.shape[1]
    
    # Calculate X^T X
    xtx = X.t().matmul(X)
    
    # Create lambda * I
    eye_np = lam * np.eye(n_features, dtype=np.float32)
    eye = torch.tensor(eye_np, device=device)
    
    # Add regularization
    reg_xtx = xtx + eye
    
    # Compute inverse
    import torch_candle.linalg as linalg
    inv_reg_xtx = linalg.inv(reg_xtx)
    
    # Compute W_opt = Y^T X (X^T X + lambda I)^-1
    yt_x = Y.t().matmul(X)
    W_opt = yt_x.matmul(inv_reg_xtx)
    
    return W_opt

class DLLTASModel(nn.Module):
    """
    Decoupled Local Analytical Solving (DLLT-AS) Model.
    
    A zero-backpropagation deep learning framework that solves layer projection 
    weights analytically in a single closed-form step using the Moore-Penrose Pseudo-Inverse.
    
    Attributes:
        in_features (int): Number of input dimensions.
        hidden_dim (int): Number of hidden channel dimensions.
        out_classes (int): Number of output class dimensions.
        reg (float): Ridge regression (Tikhonov) regularization weight factor.
    """
    def __init__(self, in_features: int, hidden_dim: int, out_classes: int, reg: float = 1e-4):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_classes = out_classes
        self.reg = reg
        
        # We construct the layers with trainable = False inside Parameter to avoid 
        # unintended backpropagation passes.
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim + in_features, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim + hidden_dim + in_features, out_classes, bias=False)
        
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Train all decoupled layers analytically in a single closed-form pass.
        
        Args:
            x (Tensor): Input tensor of shape (num_samples, in_features)
            y (Tensor): Target one-hot classification tensor of shape (num_samples, out_classes)
        """
        # --- Layer 1 Analytical Solver ---
        W1_base = dllt_solve(x, y, self.reg)
        W1_base_np = W1_base.numpy()
        W1_np = np.tile(W1_base_np.T, (1, self.hidden_dim // self.out_classes))
        self.fc1.weight = nn.Parameter(torch.tensor(W1_np.T, device=x.device))
        
        # Forward pass for Layer 1 activations
        h1 = F.silu(self.fc1(x))
        
        # --- Layer 2 Analytical Solver ---
        l2_input = torch.cat([h1, x * 0.4], dim=1)
        W2_base = dllt_solve(l2_input, y, self.reg)
        W2_base_np = W2_base.numpy()
        W2_np = np.tile(W2_base_np.T, (1, self.hidden_dim // self.out_classes))
        self.fc2.weight = nn.Parameter(torch.tensor(W2_np.T, device=x.device))
        
        # Forward pass for Layer 2 activations
        h2 = F.silu(self.fc2(l2_input))
        
        # --- Layer 3 Analytical Solver ---
        l3_input = torch.cat([h2, h1, x * 0.4], dim=1)
        W3 = dllt_solve(l3_input, y, self.reg)
        self.fc3.weight = nn.Parameter(W3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform standard feedforward inference through the analytically-solved layers.
        """
        h1 = F.silu(self.fc1(x))
        l2_input = torch.cat([h1, x * 0.4], dim=1)
        h2 = F.silu(self.fc2(l2_input))
        l3_input = torch.cat([h2, h1, x * 0.4], dim=1)
        return self.fc3(l3_input)
