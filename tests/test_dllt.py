import pytest
import numpy as np
import torch_candle as torch
from torch_candle.nn import DLLTASModel, dllt_solve

def test_dllt_solve_correctness():
    # Set random seed
    np.random.seed(42)
    
    num_samples = 50
    num_features = 5
    num_classes = 2
    lam = 1e-4
    
    # Synthetic inputs
    X_np = np.random.randn(num_samples, num_features).astype(np.float32)
    Y_np = np.random.randn(num_samples, num_classes).astype(np.float32)
    
    X = torch.tensor(X_np)
    Y = torch.tensor(Y_np)
    
    # Solve using dllt_solve
    W_opt = dllt_solve(X, Y, lam)
    
    # Solve using NumPy analytical formula: W_opt = Y^T X (X^T X + lambda I)^-1
    xtx = X_np.T @ X_np
    reg = lam * np.eye(num_features, dtype=np.float32)
    expected_W_opt = Y_np.T @ X_np @ np.linalg.inv(xtx + reg)
    
    # Assert close matching
    np.testing.assert_array_almost_equal(W_opt.numpy(), expected_W_opt, decimal=4)

def test_dllt_as_model_fit():
    np.random.seed(42)
    
    num_samples = 40
    num_features = 6
    num_classes = 2
    
    X_np = np.random.randn(num_samples, num_features).astype(np.float32)
    y_np = (X_np[:, 0] + X_np[:, 1] > 0.0).astype(np.float32)
    
    y_one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i in range(num_samples):
        y_one_hot[i, int(y_np[i])] = 1.0
        
    X = torch.tensor(X_np)
    Y = torch.tensor(y_one_hot)
    
    model = DLLTASModel(in_features=num_features, hidden_dim=8, out_classes=num_classes)
    
    # Fit the model (should run without error)
    model.fit(X, Y)
    
    # Predict/forward pass (should run without error)
    preds = model(X)
    assert preds.shape == (num_samples, num_classes)
