"""Loss functions."""

import torch
import torch.distributions as td
from torch import Tensor


def likelihood(
    prediction: Tensor,
    target: Tensor,
    event_ndims: int,
    scale: float = 1.0,
) -> Tensor:
    """Compute the negative log-likelihood."""
    dist = td.Independent(td.Normal(prediction, scale), event_ndims)
    return -dist.log_prob(target).mean()


def kl_between_standart_normal(target: td.Independent) -> Tensor:
    """Compute the KL divergence between a standard normal distribution."""
    mean = torch.zeros_like(target.mean)
    stddev = torch.ones_like(target.stddev)
    normal = td.Independent(td.Normal(mean, stddev), 1)
    kl_loss = td.kl_divergence(target, normal).mean()
    return kl_loss.div(target.batch_shape[0])
