import numpy as np
import torch_candle as torch
import torch_candle.nn as nn
import torch_candle.nn.functional as F

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
        # Convert to numpy arrays to leverage highly optimized C-based LAPACK/BLAS solvers
        x_np = x.numpy()
        y_np = y.numpy()
        
        # --- Layer 1 Analytical Solver ---
        # Moore-Penrose Pseudo-Inverse solving with ridge regularization
        reg1 = self.reg * np.eye(x_np.shape[1], dtype=np.float32)
        W1_base = np.linalg.pinv(x_np.T @ x_np + reg1) @ x_np.T @ y_np
        # Tile representation to populate hidden state dimension
        W1 = np.tile(W1_base, (1, self.hidden_dim // self.out_classes))
        # Assign solved weights directly into the layer Parameter
        self.fc1.weight = nn.Parameter(torch.tensor(W1.T))
        
        # Forward pass for Layer 1 activations
        h1 = F.silu(self.fc1(x))
        h1_np = h1.numpy()
        
        # --- Layer 2 Analytical Solver ---
        # Concatenate H1 activation with raw input coordinates
        x2_np = np.concatenate([h1_np, x_np * 0.4], axis=1)
        reg2 = self.reg * np.eye(x2_np.shape[1], dtype=np.float32)
        W2_base = np.linalg.pinv(x2_np.T @ x2_np + reg2) @ x2_np.T @ y_np
        W2 = np.tile(W2_base, (1, self.hidden_dim // self.out_classes))
        self.fc2.weight = nn.Parameter(torch.tensor(W2.T))
        
        # Forward pass for Layer 2 activations
        l2_input = torch.cat([h1, x * 0.4], dim=1)
        h2 = F.silu(self.fc2(l2_input))
        h2_np = h2.numpy()
        
        # --- Layer 3 Analytical Solver ---
        # Concatenate H2 activations, H1 activations, and raw input coordinates
        x3_np = np.concatenate([h2_np, h1_np, x_np * 0.4], axis=1)
        reg3 = self.reg * np.eye(x3_np.shape[1], dtype=np.float32)
        W3 = np.linalg.pinv(x3_np.T @ x3_np + reg3) @ x3_np.T @ y_np
        self.fc3.weight = nn.Parameter(torch.tensor(W3.T))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform standard feedforward inference through the analytically-solved layers.
        """
        h1 = F.silu(self.fc1(x))
        l2_input = torch.cat([h1, x * 0.4], dim=1)
        h2 = F.silu(self.fc2(l2_input))
        l3_input = torch.cat([h2, h1, x * 0.4], dim=1)
        return self.fc3(l3_input)
