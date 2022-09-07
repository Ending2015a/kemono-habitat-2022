# --- built in ---
from typing import Union
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
# --- my module ---
from kemono.utils import image as image_utils


def get_frontier_map(
  free_map: Union[np.ndarray, torch.Tensor],
  exp_map: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
  """Get frontier pixels
  Reference:
    https://github.com/srama2512/PONI/blob/main/poni/geometry.py#L174-L208

  Args:
    free_map (np.ndarray): walkable regions. np.bool
    exp_map (np.ndarray): explored regions. np.bool

  Returns:
    np.ndarray: frontier regions. np.bool
  """
  free_map = torch.as_tensor(free_map).to(dtype=torch.bool)
  exp_map = torch.as_tensor(exp_map).to(dtype=torch.bool)
  orig_shape = free_map.shape
  orig_ndims = len(orig_shape)
  free_map = image_utils.to_4D_tensor(free_map)
  exp_map = image_utils.to_4D_tensor(exp_map)
  
  unexp_map = ~exp_map
  # calculate frontiers (4D tensor)
  unexp_map_u = nn.functional.pad(unexp_map, (0, 0, 0, 1))[..., 1:, :]
  unexp_map_d = nn.functional.pad(unexp_map, (0, 0, 1, 0))[..., :-1, :]
  unexp_map_l = nn.functional.pad(unexp_map, (0, 1, 0, 0))[..., :, 1:]
  unexp_map_r = nn.functional.pad(unexp_map, (1, 0, 0, 0))[..., :, :-1]
  frontiers = (
    (free_map == unexp_map_u) |
    (free_map == unexp_map_d) |
    (free_map == unexp_map_l) |
    (free_map == unexp_map_r)
  ) & (free_map)

  frontiers = image_utils.from_4D_tensor(frontiers, orig_ndims)
  return frontiers.to(dtype=torch.bool)
