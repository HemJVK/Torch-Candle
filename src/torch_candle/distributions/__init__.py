"""torch_candle.distributions — Probability distributions matching torch.distributions."""
import numpy as np
from ..tensor import Tensor

def _w(x): return Tensor(np.array(x, dtype=np.float32))


class Distribution:
    has_rsample = False
    has_enumerate_support = False

    def sample(self, sample_shape=()):
        raise NotImplementedError
    def log_prob(self, value):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def cdf(self, value):
        raise NotImplementedError
    def icdf(self, value):
        raise NotImplementedError
    def rsample(self, sample_shape=()):
        raise NotImplementedError
    def mean(self):
        raise NotImplementedError
    def variance(self):
        raise NotImplementedError
    def stddev(self):
        return _w(np.sqrt(self.variance().numpy()))
    def expand(self, batch_shape):
        raise NotImplementedError


class Normal(Distribution):
    has_rsample = True
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc if isinstance(loc, np.ndarray) else np.array(getattr(loc, 'numpy', lambda: loc)(), dtype=np.float32)
        self.scale = scale if isinstance(scale, np.ndarray) else np.array(getattr(scale, 'numpy', lambda: scale)(), dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.loc.shape
        return _w(np.random.normal(self.loc, self.scale, shape).astype(np.float32))

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        return _w(-0.5 * ((v - self.loc) / self.scale) ** 2 - np.log(self.scale) - 0.5 * np.log(2 * np.pi))

    def entropy(self):
        return _w(0.5 + 0.5 * np.log(2 * np.pi * np.e * self.scale ** 2))

    def cdf(self, value):
        from scipy.special import ndtr
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        return _w(ndtr((v - self.loc) / self.scale).astype(np.float32))

    def mean(self): return _w(self.loc)
    def variance(self): return _w(self.scale ** 2)


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            self.probs = probs.numpy() if isinstance(probs, Tensor) else np.array(probs, dtype=np.float32)
        elif logits is not None:
            lg = logits.numpy() if isinstance(logits, Tensor) else np.array(logits, dtype=np.float32)
            self.probs = 1.0 / (1.0 + np.exp(-lg))
        else:
            raise ValueError("probs or logits required")

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.probs.shape
        return _w(np.random.binomial(1, self.probs, shape).astype(np.float32))

    def log_prob(self, value):
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        return _w(v * np.log(np.clip(self.probs, 1e-7, 1)) +
                  (1 - v) * np.log(np.clip(1 - self.probs, 1e-7, 1)))

    def entropy(self):
        p = self.probs
        return _w(-(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1))))

    def mean(self): return _w(self.probs)
    def variance(self): return _w(self.probs * (1 - self.probs))


class Categorical(Distribution):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            p = probs.numpy() if isinstance(probs, Tensor) else np.array(probs, dtype=np.float32)
            self.probs = p / p.sum(-1, keepdims=True)
        else:
            lg = logits.numpy() if isinstance(logits, Tensor) else np.array(logits, dtype=np.float32)
            self.probs = np.exp(lg - lg.max(-1, keepdims=True))
            self.probs /= self.probs.sum(-1, keepdims=True)

    def sample(self, sample_shape=()):
        n = self.probs.shape[-1]
        shape = tuple(sample_shape)
        p_flat = self.probs.flatten()
        p_flat /= p_flat.sum() # Ensure exact sum to 1.0
        indices = np.array([np.random.choice(n, p=p_flat) for _ in range(max(1, int(np.prod(shape))))]).reshape(shape or (1,))
        return _w(indices.astype(np.int64))

    def log_prob(self, value):
        v = int(value.numpy().flat[0]) if isinstance(value, Tensor) else int(value)
        return _w(np.log(np.clip(self.probs[..., v], 1e-7, 1)))

    def entropy(self):
        return _w(-np.sum(self.probs * np.log(np.clip(self.probs, 1e-7, 1)), axis=-1))

    def mean(self): return _w(np.arange(self.probs.shape[-1]).dot(self.probs.T))
    def variance(self): return _w(np.arange(self.probs.shape[-1]) ** 2 @ self.probs.T - self.mean().numpy() ** 2)


class Uniform(Distribution):
    has_rsample = True
    def __init__(self, low, high, validate_args=None):
        self.low = low.numpy() if isinstance(low, Tensor) else np.array(low, dtype=np.float32)
        self.high = high.numpy() if isinstance(high, Tensor) else np.array(high, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + np.broadcast(self.low, self.high).shape
        return _w(np.random.uniform(self.low, self.high, shape).astype(np.float32))

    def rsample(self, sample_shape=()): return self.sample(sample_shape)

    def log_prob(self, value):
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        within = (v >= self.low) & (v < self.high)
        lp = np.where(within, -np.log(self.high - self.low), -np.inf)
        return _w(lp)

    def entropy(self): return _w(np.log(self.high - self.low))
    def mean(self): return _w((self.low + self.high) / 2)
    def variance(self): return _w((self.high - self.low) ** 2 / 12)


class Exponential(Distribution):
    has_rsample = True
    def __init__(self, rate, validate_args=None):
        self.rate = rate.numpy() if isinstance(rate, Tensor) else np.array(rate, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.rate.shape
        return _w(np.random.exponential(1.0 / self.rate, shape).astype(np.float32))

    def rsample(self, sample_shape=()): return self.sample(sample_shape)
    def log_prob(self, value):
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        return _w(np.log(self.rate) - self.rate * v)
    def entropy(self): return _w(1 - np.log(self.rate))
    def mean(self): return _w(1.0 / self.rate)
    def variance(self): return _w(1.0 / self.rate ** 2)


class MultivariateNormal(Distribution):
    has_rsample = True
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        self.loc = loc.numpy() if isinstance(loc, Tensor) else np.array(loc, dtype=np.float32)
        if covariance_matrix is not None:
            cov = covariance_matrix.numpy() if isinstance(covariance_matrix, Tensor) else np.array(covariance_matrix, dtype=np.float32)
            self.scale_tril = np.linalg.cholesky(cov).astype(np.float32)
        elif scale_tril is not None:
            self.scale_tril = scale_tril.numpy() if isinstance(scale_tril, Tensor) else np.array(scale_tril, dtype=np.float32)
        else:
            raise ValueError("Must supply covariance_matrix or scale_tril")

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.loc.shape
        z = np.random.standard_normal(shape).astype(np.float32)
        return _w((self.loc + z @ self.scale_tril.T).astype(np.float32))

    def rsample(self, sample_shape=()): return self.sample(sample_shape)
    def log_prob(self, value):
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        diff = v - self.loc
        cov = self.scale_tril @ self.scale_tril.T
        logdet = 2 * np.sum(np.log(np.diag(self.scale_tril)))
        mahal = diff @ np.linalg.inv(cov) @ diff
        k = self.loc.shape[-1]
        return _w(-0.5 * (k * np.log(2 * np.pi) + logdet + mahal))
    def mean(self): return _w(self.loc)
    def variance(self): return _w(np.diag(self.scale_tril @ self.scale_tril.T))


class Dirichlet(Distribution):
    def __init__(self, concentration, validate_args=None):
        self.concentration = concentration.numpy() if isinstance(concentration, Tensor) else np.array(concentration, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.concentration.shape[:-1]
        return _w(np.random.dirichlet(self.concentration, shape).astype(np.float32))

    def mean(self):
        return _w(self.concentration / np.sum(self.concentration, axis=-1, keepdims=True))


class Beta(Distribution):
    has_rsample = True
    def __init__(self, concentration1, concentration0, validate_args=None):
        self.alpha = concentration1.numpy() if isinstance(concentration1, Tensor) else np.array(concentration1, dtype=np.float32)
        self.beta = concentration0.numpy() if isinstance(concentration0, Tensor) else np.array(concentration0, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + np.broadcast(self.alpha, self.beta).shape
        return _w(np.random.beta(self.alpha, self.beta, shape).astype(np.float32))

    def rsample(self, sample_shape=()): return self.sample(sample_shape)
    def mean(self): return _w(self.alpha / (self.alpha + self.beta))
    def variance(self): return _w(self.alpha * self.beta / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)))


class Gamma(Distribution):
    has_rsample = True
    def __init__(self, concentration, rate, validate_args=None):
        self.concentration = concentration.numpy() if isinstance(concentration, Tensor) else np.array(concentration, dtype=np.float32)
        self.rate = rate.numpy() if isinstance(rate, Tensor) else np.array(rate, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + np.broadcast(self.concentration, self.rate).shape
        return _w(np.random.gamma(self.concentration, 1.0 / self.rate, shape).astype(np.float32))

    def rsample(self, sample_shape=()): return self.sample(sample_shape)
    def mean(self): return _w(self.concentration / self.rate)
    def variance(self): return _w(self.concentration / self.rate ** 2)


class Poisson(Distribution):
    def __init__(self, rate, validate_args=None):
        self.rate = rate.numpy() if isinstance(rate, Tensor) else np.array(rate, dtype=np.float32)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self.rate.shape
        return _w(np.random.poisson(self.rate, shape).astype(np.float32))

    def log_prob(self, value):
        from math import lgamma
        v = value.numpy() if isinstance(value, Tensor) else np.array(value, dtype=np.float32)
        return _w(v * np.log(self.rate) - self.rate - np.vectorize(lgamma)(v + 1))

    def mean(self): return _w(self.rate)
    def variance(self): return _w(self.rate)


# kl_divergence registry
_KL_REGISTRY = {}

def kl_divergence(p, q):
    key = (type(p), type(q))
    if key in _KL_REGISTRY:
        return _KL_REGISTRY[key](p, q)
    raise NotImplementedError(f"No KL divergence registered for {type(p)}, {type(q)}")

def register_kl(p_cls, q_cls):
    def decorator(fn):
        _KL_REGISTRY[(p_cls, q_cls)] = fn
        return fn
    return decorator
