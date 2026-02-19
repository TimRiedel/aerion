"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing.

Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Learning", ICML 2018.
"""

import torch
from torch import nn
from typing import Tuple

from torch.optim import Adam


class GradNormBalancer(nn.Module):
    """
    Learnable loss weights updated by matching gradient norms to inverse training rates.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        weight_lr: float = 0.025,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            alpha: Asymmetry parameter; higher values favor harder tasks more.
            weight_lr: Learning rate for the weight optimizer.
        """
        super().__init__()
        self.alpha = alpha
        self.weight_lr = weight_lr
        self.l0 = None
        self.weights = None
        self.T = None

    def is_initialized(self) -> bool:
        return self.l0 is not None or self.weights is not None

    def init_state_and_optimizer(self, loss: torch.Tensor) -> torch.optim.Optimizer:
        if self.is_initialized():
            raise ValueError("init_state should only be called once")

        self.weights = nn.Parameter(torch.ones_like(loss, device=loss.device, dtype=loss.dtype))
        self.T = self.weights.sum().detach()
        self.l0 = loss.detach()
        gradnorm_optimizer = Adam([self.weights], lr=self.weight_lr)
        return gradnorm_optimizer

    def compute_weighted_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.weights @ loss

    def compute_gradnorm_loss(
        self,
        loss: torch.Tensor,
        layer: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the GradNorm auxiliary loss (L1 between current and target gradient norms).
        Call after weighted_loss.backward(retain_graph=True).

        Args:
            loss: Tensor of shape [num_tasks] (same as passed to weighted_loss).
            layer: Module whose parameters are used for gradient norms (e.g. model.output_projection).

        Returns:
            gradnorm_loss: Scalar loss to backward and update weights.
            loss_ratio: loss.detach() / l0 for logging.
            weights: Current weights for logging (detached).
        """
        gw = []
        for i in range(len(loss)):
            dl = torch.autograd.grad(
                self.weights[i] * loss[i],
                layer.parameters(),
                retain_graph=True,
                create_graph=True,
            )[0]
            gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        loss_ratio = loss.detach() / self.l0
        rt = loss_ratio / loss_ratio.mean()
        gw_avg = gw.mean().detach()
        constant = (gw_avg * rt ** self.alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        return gradnorm_loss, self.weights.detach()

    def renormalize_weights(self) -> torch.optim.Optimizer:
        """
        Renormalize weights to sum to T, replace Parameter, and create new weight optimizer.
        Call after model_optimizer.step() and weight_optimizer.step().

        Returns:
            New Adam optimizer over the renormalized weights. Use this for the next step.
        """
        new_weights = (self.weights / self.weights.sum() * self.T).detach()
        self.weights = nn.Parameter(new_weights)
        gradnorm_optimizer = Adam([self.weights], lr=self.weight_lr)
        return gradnorm_optimizer
