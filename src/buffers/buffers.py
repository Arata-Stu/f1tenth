import torch
from omegaconf import DictConfig
from .OffPolicy import OffPolicyBuffer
from .n_step_OffPolicy import NStepOffPolicyBuffer
from .OnPolicy import OnPolicyBuffer

def get_buffers(buffer_cfg: DictConfig, state_dim: int, action_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if buffer_cfg.type == "OffPolicy":
        return OffPolicyBuffer(size=int(buffer_cfg.size),
                               state_dim=state_dim,
                               action_dim=action_dim,
                               device=device)

    elif buffer_cfg.type == "N-Step-OffPolicy":
        return NStepOffPolicyBuffer(size=int(buffer_cfg.size),
                                    state_dim=state_dim,
                                    action_dim=action_dim,
                                    device=device,
                                    n_step=int(buffer_cfg.n_step),
                                    gamma=float(buffer_cfg.gamma))  # gamma を渡す必要あり

    elif buffer_cfg.type == "OnPolicy":
        return OnPolicyBuffer(size=int(buffer_cfg.size),
                              state_dim=state_dim,
                              action_dim=action_dim,
                              device=device)

    else:
        raise ValueError(f"Unexpected buffer type: {buffer_cfg.type}")