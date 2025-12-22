import torch

def create_mask(batch: torch.Tensor) -> torch.Tensor:
    return batch != 0
