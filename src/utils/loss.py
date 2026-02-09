import torch
import torch.nn.functional as F


def rollout_mse_loss(
    model,
    x,
    y,
    detach=True,
    weights=None,
):
    """
    Multi-step rollout MSE loss.

    Args:
        model:      neural network, f(x) -> y_hat
        x:          initial input, shape [B, C, H, W]
        y:          ground truth rollout targets, shape [B, K, 1, H, W]
        detach:     whether to detach state between rollout steps (truncated BPTT)
        weights:    optional list or tensor of length K for step-wise loss weighting

    Returns:
        total_loss: scalar tensor
        loss_per_step: list of per-step losses (for logging)
    """

    B, K, _, H, W = y.shape

    state = x
    total_loss = 0.0
    loss_per_step = []

    if weights is None:
        weights = [1.0] * K
    else:
        assert len(weights) == K, "Length of weights must equal rollout steps K"

    for k in range(K):
        # Predict next step
        pred = model(state)              # [B, 1, H, W]

        # Ground truth at step k: y(t+k+1)
        target = y[:, k]                 # [B, 1, H, W]

        # MSE loss
        step_loss = F.mse_loss(pred, target)
        total_loss = total_loss + weights[k] * step_loss
        loss_per_step.append(step_loss.detach())

        # Roll forward
        state = pred.detach() if detach else pred

    return total_loss, loss_per_step
