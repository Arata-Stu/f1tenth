import numpy as np
import torch


class BaseBuffer:
    """
    Base buffer for storing transitions in reinforcement learning.
    Stores arrays for states, actions, rewards, next_states, and dones.
    Provides methods to reset buffer and convert to numpy / tensor.
    """
    def __init__(self, size, state_dim, action_dim, device=None):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reset()

    def reset(self):
        """Clear the buffer."""
        self.ptr = 0
        self.full = False
        self.storage = {
            "states": np.zeros((self.size, self.state_dim), dtype=np.float32),
            "actions": np.zeros((self.size, self.action_dim), dtype=np.float32),
            "rewards": np.zeros((self.size, 1), dtype=np.float32),
            "next_states": np.zeros((self.size, self.state_dim), dtype=np.float32),
            "dones": np.zeros((self.size, 1), dtype=np.float32),
            "log_probs":   np.zeros((self.size, 1), dtype=np.float32),
        }

    def __len__(self):
        """Return the current size of the buffer."""
        return self.size if self.full else self.ptr

    def to_numpy(self, array):
        """
        Convert a torch tensor or other array-like to a NumPy array of type float32.
        """
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(np.float32)
        return np.array(array, dtype=np.float32)

    def to_tensor(self, array):
        """
        Convert a NumPy array to a torch tensor on the specified device.
        """
        tensor = torch.from_numpy(array)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def add(self, state, action, reward, next_state, done, log_prob=None):
        """
        Add a transition to the buffer at current pointer.
        Overwrites oldest data if buffer is full.
        Converts inputs to NumPy to avoid type/device mismatches.
        """
        state = self.to_numpy(state)
        action = self.to_numpy(action)
        reward = self.to_numpy(reward)
        next_state = self.to_numpy(next_state)
        done = self.to_numpy(done).astype(np.float32)

        self.storage["states"][self.ptr] = state
        self.storage["actions"][self.ptr] = action
        self.storage["rewards"][self.ptr] = reward
        self.storage["next_states"][self.ptr] = next_state
        self.storage["dones"][self.ptr] = done

        # Store log_prob if provided
        if log_prob is not None:
            lp = self.to_numpy(log_prob)
            # ensure shape (1,) or scalar
            self.storage["log_probs"][self.ptr, 0] = lp

        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0

    def get_batch(self, batch_size=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_numpy(self):
        """
        Get all data currently in the buffer (up to full capacity or pointer).
        Returns a dict of NumPy arrays.
        """
        max_idx = self.size if self.full else self.ptr
        return {k: v[:max_idx] for k, v in self.storage.items()}

    def get_tensor(self):
        """
        Get all data as PyTorch tensors.
        """
        np_data = self.get_numpy()
        return {k: self.to_tensor(v) for k, v in np_data.items()}

