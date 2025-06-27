
import torch

class GhostTracker:
    """Lightweight accumulator for ghost dot‑product Gram matrices."""

    def __init__(self):
        self.total_G = None
        self.batch_size = None

    def reset(self, batch_size: int | None = None):
        """Clear previous accumulation. Optionally cache batch size."""
        self.total_G = None
        if batch_size is not None:
            self.batch_size = batch_size

    @torch.no_grad()
    def add_layer(self, a: torch.Tensor, b: torch.Tensor):
        """Accumulate <∇ℓ_i,∇ℓ_j> for one linear layer.

        Args:
            a: activations that flowed *into* the weight. Shape (B, …)
            b: back‑prop gradients that flowed *out* of the same node. Shape (B, …)
        """
        B = a.size(0)
        a_flat = a.reshape(B, -1)
        b_flat = b.reshape(B, -1)
        G_layer = (a_flat @ a_flat.T) * (b_flat @ b_flat.T)  # Hadamard product
        if self.total_G is None:
            self.total_G = G_layer
        else:
            self.total_G += G_layer

    def pairwise_dot(self) -> torch.Tensor | None:
        """Return the accumulated pairwise gradient dot‑products."""
        return None if self.total_G is None else self.total_G.clone()
