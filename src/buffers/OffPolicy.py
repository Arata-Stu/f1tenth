import numpy as np
from .base import BaseBuffer

class OffPolicyBuffer(BaseBuffer):
    """
    Replay buffer for off-policy algorithms (e.g., SAC, TD3).
    Supports random sampling of mini-batches.
    """
    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.
        Returns tensors on the specified device,
        excluding log_probs.
        """
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch_size)
        # log_probs を除外
        keys = ["states", "actions", "rewards", "next_states", "dones"]
        batch = {k: self.storage[k][idxs] for k in keys}
        return {k: self.to_tensor(v) for k, v in batch.items()}
    
    def get_batch(self, batch_size):
        return self.sample(batch_size)
