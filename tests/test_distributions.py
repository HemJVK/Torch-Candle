import pytest
import numpy as np
import torch_candle as torch
import torch_candle.distributions as D

def test_normal():
    norm = D.Normal(0.0, 1.0)
    sample = norm.sample((1000,))
    assert sample.shape == (1000,)
    # approx mean 0, std 1
    assert abs(sample.mean().item()) < 0.1
    assert abs(sample.std().item() - 1.0) < 0.1
    
    lp = norm.log_prob(torch.tensor([0.0]))
    # log(1 / sqrt(2*pi)) ~ -0.9189
    assert abs(lp.item() - (-0.9189)) < 1e-3

def test_bernoulli():
    b = D.Bernoulli(probs=torch.tensor([0.3]))
    sample = b.sample((1000,))
    assert abs(sample.mean().item() - 0.3) < 0.05
    assert b.entropy().item() > 0

def test_categorical():
    c = D.Categorical(probs=torch.tensor([0.1, 0.2, 0.7]))
    s = c.sample((1000,))
    # 2 should be the most frequent
    counts = np.bincount(s.numpy().flatten().astype(int))
    assert counts[2] > 600

def test_uniform():
    u = D.Uniform(torch.tensor([0.0]), torch.tensor([10.0]))
    s = u.sample((1000,))
    assert s.min().item() >= 0.0
    assert s.max().item() <= 10.0
    assert abs(s.mean().item() - 5.0) < 0.5

def test_multivariate_normal():
    loc = torch.zeros(2)
    cov = torch.eye(2)
    mvn = D.MultivariateNormal(loc, covariance_matrix=cov)
    s = mvn.sample((500,))
    assert s.shape == (500, 2)
    assert abs(s.mean().item()) < 0.2

def test_dirichlet():
    d = D.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
    s = d.sample((10,))
    assert s.shape == (10, 3)
    # Sum across last dim should be 1
    assert np.allclose(s.sum(dim=-1).numpy(), 1.0)

def test_kl_divergence():
    # Placeholder for KL registry test
    with pytest.raises(NotImplementedError):
        p = D.Normal(0.0, 1.0)
        q = D.Normal(1.0, 1.0)
        D.kl_divergence(p, q)
