from .base import BaseBuffer


class OnPolicyBuffer(BaseBuffer):
    """
    Buffer for on-policy algorithms (e.g., PPO).
    Collects one batch of transitions and returns all at once.
    """
    
    def get(self):
        """
        Retrieve all stored transitions (tensors) and reset buffer.
        """
        data = self.get_tensor()
        self.reset()
        return data
    
    def get_batch(self, batch_size=None):
        return self.get()