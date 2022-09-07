# --- built in ---
import enum
import random
from typing import (
  Any,
  Dict,
  Tuple,
  Union,
  Optional
)
import dataclasses
from dataclasses import dataclass
# --- 3rd party ---
import torch
from torch import nn
import numpy as np
# --- my module ---

__all__ = [
  'to_4D_tensor',
  'from_4D_tensor',
  'ImageNormalize',
  'resize_image',
  'RandomTransformState'
]

def to_4D_tensor(t: torch.Tensor) -> torch.Tensor:
  """Convert `t` to 4D tensors (b, c, h, w)
  Args:
    t (torch.Tensor): 0/1/2/3/4D tensor
      0D: ()
      1D: (c,)
      2D: (h, w)
      3D: (c, h, w)
      4D: (b, c, h, w)
  
  Returns:
    torch.Tensor: 4D image tensor
  """
  ndims = len(t.shape)
  if ndims == 0:
    # () -> (b, c, h, w)
    return torch.broadcast_to(t, (1, 1, 1, 1))
  elif ndims == 1:
    # (c,) -> (b, c, h, w)
    return t[None, :, None, None]
  elif ndims == 2:
    # (h, w) -> (b, c, h, w)
    return t[None, None, :, :] # b, c
  elif ndims == 3:
    # (c, h, w) -> (b, c, h, w)
    return t[None, :, :, :] # b
  else:
    return t

def from_4D_tensor(t: torch.Tensor, ndims: int) -> torch.Tensor:
  """Convert `t` to `ndims`-D tensors
  Args:
    t (torch.Tensor): 4D image tensors in shape (b, c, h, w).
    ndims (int): the original rank of the image.
  Returns:
    torch.Tensor: `ndims`-D tensors
  """
  _ndims = len(t.shape)
  assert _ndims == 4, f"`t` must be a 4D tensor, but {_ndims}-D are given."
  if ndims == 0:
    return t[0, 0, 0, 0]
  elif ndims == 1:
    return t[0, :, 0, 0]
  elif ndims == 2:
    return t[0, 0, :, :] # -b, -c
  elif ndims == 3:
    return t[0, :, :, :] # -b
  else:
    return t

class ImageNormalize(nn.Module):
  def __init__(
    self,
    mean: Union[np.ndarray, torch.Tensor],
    std: Union[np.ndarray, torch.Tensor],
  ):
    super().__init__()
    mean = to_4D_tensor(torch.as_tensor(mean))
    std = to_4D_tensor(torch.as_tensor(std))
    self.register_buffer('mean', mean, persistent=False)
    self.register_buffer('std', std, persistent=False)

  def forward(self, x):
    return (x - self.mean) / self.std

def resize_image(
  image: Union[np.ndarray, torch.Tensor],
  size: Tuple[int, int],
  mode: str = 'nearest',
  **kwargs
) -> torch.Tensor:
  """Resize image tensor

  Args:
      image (Union[np.ndarray, torch.Tensor]): expecting
        (h, w), (c, h, w), (b, c, h, w)
      size (Tuple[int, int]): target size in (h, w)
      mode (str, optional): resize mode. Defaults to 'nearest'.
  """
  size = torch.Size(size)
  t = torch.as_tensor(image)
  orig_shape = t.shape
  orig_dtype = t.dtype
  orig_ndims = len(orig_shape)
  _t = to_4D_tensor(t) # (b, c, h, w)
  # do nothing if the tensor shape is already matched to the target size
  if _t.shape[-len(size):] == size:
    return t
  t = _t
  t = t.to(dtype=torch.float32)
  t = nn.functional.interpolate(t, size=size, mode=mode, **kwargs)
  t = from_4D_tensor(t, orig_ndims)
  return t.to(dtype=orig_dtype)


class AugmentType(str, enum.Enum):
  crop_rate = 'crop_rate'
  hflip = 'hflip'
  vflip = 'vflip'
  move_x = 'move_x'
  move_y = 'move_y'
  rotate = 'rotate'
  shear_x = 'shear_x'
  shear_y = 'shear_y'

def affine_sample(image, value, type: AugmentType):
  assert torch.is_tensor(image)
  assert len(image.shape) == 4, \
    "Only accept 4D image tensor (b, c, h, w)"
  orig_dtype = image.dtype
  image = image.to(dtype=torch.float32)
  # borrowed from ProDA
  type = AugmentType(type)
  v = value
  if type == AugmentType.rotate:
    theta = np.array([
      [np.cos(v/180*np.pi), -np.sin(v/180*np.pi), 0],
      [np.sin(v/180*np.pi), np.cos(v/180*np.pi), 0]
    ]).astype(np.float32)
  elif type == AugmentType.shear_x:
    theta = np.array([[1, v, 0], [0, 1, 0]]).astype(np.float32)
  elif type == AugmentType.shear_y:
    theta = np.array([[1, 0, 0], [v, 1, 0]]).astype(np.float32)
  elif type == AugmentType.move_x:
    theta = np.array([[1, 0, v], [0, 1, 0]]).astype(np.float32)
  elif type == AugmentType.move_y:
    theta = np.array([[1, 0, 0], [0, 1, v]]).astype(np.float32)
  h, w = image.shape[-2:]
  theta[0,1] = theta[0,1]*h/w
  theta[1,0] = theta[1,0]*w/h
  if type == AugmentType.shear_x or type == AugmentType.shear_y:
    theta[0,2] = theta[0,2]*2/h + theta[0,0] + theta[0,1] - 1
    theta[1,2] = theta[1,2]*2/h + theta[1,0] + theta[1,1] - 1
  theta = torch.as_tensor(theta, device=image.device)
  theta = theta.unsqueeze(0)
  grid = nn.functional.affine_grid(theta, image.size())
  grid = grid.to(device=image.device)
  image = nn.functional.grid_sample(
    image, grid, mode='nearest', align_corners=True
  )
  image = image.to(dtype=orig_dtype)
  return image


def flip_image(
  image: torch.Tensor,
  hflip: bool = False,
  vflip: bool = False
):
  if not hflip and not vflip:
    return image
  assert torch.is_tensor(image)
  dims = []
  if hflip:
    dims.append(-2)
  if vflip:
    dims.append(-1)
  image = torch.flip(image, dims)
  return image

def crop_image(image, crop_rate):
  assert torch.is_tensor(image)
  h, w = image.shape[-2:]
  crop_h = crop_rate[0] * h
  crop_w = crop_rate[1] * w
  t = int(crop_h // 2)
  l = int(crop_w // 2)
  d = h - int(crop_h - t)
  r = w - int(crop_w - l)
  # random crop
  image = image[..., t:d, l:r]
  return image

@dataclass
class RandomTransformState:
  img_size: Optional[Tuple[int, int]] = None
  random_rate: float = 0.5
  random_k: int = 2
  crop_rate: Optional[Tuple[float, float]] = None
  hflip: Optional[bool] = None
  vflip: Optional[bool] = None
  move_x: Optional[float] = None
  move_y: Optional[float] = None
  rotate: Optional[float] = None
  shear_x: Optional[float] = None
  shear_y: Optional[float] = None

  def generate_random(self) -> "RandomTransformState":
    k = self.random_k
    if np.random.uniform() >= self.random_rate:
      # no random argumentation
      return RandomTransformState(
        img_size = self.img_size,
        random_rate = self.random_rate,
        random_k = k
      )
    # get valid options (non-None, True, non-zero) from self
    options = self._get_valid_options()
    if len(options) == 0:
      return self
    # random choice k random transformations
    self_dict = dataclasses.asdict(self)
    opts = random.choices(options, k=k)
    kwargs = {}
    for opt in opts:
      kwargs[opt.value] = self._generate_random_value(opt)
    return RandomTransformState(
      img_size = self.img_size,
      random_rate = self.random_rate,
      random_k = k,
      **kwargs
    )

  def _get_valid_options(self) -> Dict[str, Any]:
    self_dict = dataclasses.asdict(self)
    options = [
      t for t in AugmentType
      if self_dict[t.value]
    ]
    return options

  def _generate_random_value(self, opt):
    self_dict = dataclasses.asdict(self)
    opt_str = opt.value
    if opt is AugmentType.crop_rate:
      return (
        np.random.uniform(high=self.crop_rate[0]),
        np.random.uniform(high=self.crop_rate[1])
      )
    elif opt is AugmentType.hflip or opt is AugmentType.vflip:
      # random bool
      return np.random.randint(2, dtype=bool)
    else:
      v = np.abs(self_dict[opt_str])
      return np.random.uniform(-v, v)

  def apply(
    self,
    image: torch.Tensor,
    mode: str = 'bilinear'
  ) -> torch.Tensor:
    assert torch.is_tensor(image), \
      "the input image must be a torch.Tensor"
    orig_ndims = len(image.shape)
    image = to_4D_tensor(image)

    if self.crop_rate is not None:
      image = crop_image(image, self.crop_rate)
    image = flip_image(image, hflip=self.hflip, vflip=self.vflip)
    if self.move_x:
      image = affine_sample(image, self.move_x, AugmentType.move_x)
    if self.move_y:
      image = affine_sample(image, self.move_y, AugmentType.move_y)
    if self.rotate:
      image = affine_sample(image, self.rotate, AugmentType.rotate)
    if self.shear_x:
      image = affine_sample(image, self.shear_x, AugmentType.shear_x)
    if self.shear_y:
      image = affine_sample(image, self.shear_y, AugmentType.shear_y)
    image = resize_image(image, self.img_size, mode)
    image = from_4D_tensor(image, ndims=orig_ndims)
    return image
