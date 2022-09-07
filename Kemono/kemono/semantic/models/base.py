# --- built in ---
import abc
# --- 3rd party ---
import torch
from torch import nn
import pytorch_lightning as pl
from rlchemy import registry
# --- my module ---

__all__ = [
  'BaseSemanticModel'
]

class BaseSemanticModel(nn.Module):
  """An interface of semantic models
  """
  @abc.abstractmethod
  def forward(
    self,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    **kwargs
  ) -> torch.Tensor:
    pass
